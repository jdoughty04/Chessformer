"""
Benchmark a trained policy-only/chess-fusion adapter against Elo-limited UCI bots.

The bot policy is intentionally simple:
1) Prefer the move with highest model move-eval score among legal moves.
2) Fallback to highest policy-logit legal move when move-eval head is unavailable.

Usage example:
  python src/inference/benchmark_policy_elo.py ^
    --checkpoint checkpoints/pretrain_policy/epoch-10 ^
    --config configs/pretrain_policy_only.yaml ^
    --engine "C:\\engines\\stockfish\\stockfish.exe" ^
    --elos 800 1000 1200 1400 1600 1800 ^
    --games-per-elo 8 ^
    --movetime-ms 75
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine
import chess.pgn
import torch

# src/inference/benchmark_policy_elo.py -> add src/ for training imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import training.chess_fusion_model as chess_fusion_model
from training.config import ModelConfig, load_config
from training.maia_model import extract_maia_features


@dataclass
class GameRecord:
    opponent_elo: int
    bot_is_white: bool
    result: str
    score: float
    ply_count: int


@dataclass
class LoadMeta:
    fallback_mapping_used: bool
    missing_keys: int
    unexpected_keys: int
    missing_example: str
    unexpected_example: str
    checkpoint: str
    config: str
    device: str


class _FallbackMapping:
    def __init__(self, vocab: List[str]):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.idx_to_move = {i: m for i, m in enumerate(vocab)}
        self.move_to_idx = {m: i for i, m in enumerate(vocab)}

    def decode(self, idx: int) -> Optional[str]:
        return self.idx_to_move.get(idx, None)

    def encode(self, move_uci: str) -> int:
        return self.move_to_idx.get(move_uci, -1)


class PolicyMoveDecoder:
    """Decode policy vocab indices to legal chess.Move without maia2 dependency."""

    def __init__(self, adapter: chess_fusion_model.ChessFusionAdapter):
        perceiver = getattr(adapter, "perceiver", None)
        self.from_idx = getattr(perceiver, "policy_from_square_idx", None)
        self.to_idx = getattr(perceiver, "policy_to_square_idx", None)
        if self.from_idx is None or self.to_idx is None:
            raise RuntimeError(
                "Structured policy index buffers not found on adapter. "
                "This script currently supports structured policy checkpoints."
            )
        self.from_idx = self.from_idx.detach().cpu().tolist()
        self.to_idx = self.to_idx.detach().cpu().tolist()
        if len(self.from_idx) != len(self.to_idx):
            raise RuntimeError("Invalid policy index buffers: from/to length mismatch.")
        self.vocab_size = len(self.from_idx)

    @staticmethod
    def _mirror_square(sq: int) -> int:
        # Mirror by rank only: a1<->a8, b1<->b8, ...
        return sq ^ 56

    def _abs_from_to(self, vocab_idx: int, turn_white: bool) -> Tuple[int, int]:
        frm = int(self.from_idx[vocab_idx])
        to = int(self.to_idx[vocab_idx])
        if turn_white:
            return frm, to
        return self._mirror_square(frm), self._mirror_square(to)

    @staticmethod
    def _pick_promotion(moves: List[chess.Move]) -> chess.Move:
        if len(moves) == 1:
            return moves[0]
        promo_pref = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        by_promo = {m.promotion: m for m in moves}
        for p in promo_pref:
            if p in by_promo:
                return by_promo[p]
        return moves[0]

    def decode_legal_move(self, board: chess.Board, vocab_idx: int) -> Optional[chess.Move]:
        frm, to = self._abs_from_to(vocab_idx, board.turn == chess.WHITE)
        matches = [m for m in board.legal_moves if m.from_square == frm and m.to_square == to]
        if not matches:
            return None
        return self._pick_promotion(matches)


def _resolve_checkpoint_dir(path: Path) -> Path:
    """Resolve to a directory that contains adapter.pt."""
    if (path / "adapter.pt").exists():
        return path

    candidates = sorted(
        [
            d
            for d in path.glob("*")
            if d.is_dir()
            and (d / "adapter.pt").exists()
            and (d.name.startswith("epoch-") or d.name.startswith("checkpoint-"))
        ],
        key=lambda d: d.name,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No adapter.pt found in {path} or child epoch/checkpoint dirs.")


def _find_config_path(checkpoint_dir: Path, explicit: Optional[Path]) -> Path:
    if explicit is not None:
        return explicit
    ckpt_cfg = checkpoint_dir / "config.yaml"
    if ckpt_cfg.exists():
        return ckpt_cfg
    raise FileNotFoundError(
        "Config file not provided and checkpoint has no config.yaml. "
        "Pass --config explicitly."
    )


def _resolve_engine_path(engine_path: Path) -> Path:
    if engine_path.exists():
        return engine_path
    if engine_path.suffix.lower() != ".exe":
        alt = engine_path.with_suffix(".exe")
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Engine executable not found: {engine_path}")


def _square_name(idx: int) -> str:
    return chess.square_name(int(idx))


def _maybe_patch_maia_mapping_from_state(state: Dict[str, torch.Tensor]) -> bool:
    """Patch chess_fusion_model.get_maia_mapping when maia2 is unavailable."""
    try:
        chess_fusion_model.get_maia_mapping()
        return False
    except Exception:
        pass

    from_key = None
    to_key = None
    for k in state.keys():
        if k.endswith("policy_from_square_idx"):
            from_key = k
        elif k.endswith("policy_to_square_idx"):
            to_key = k
    if from_key is None or to_key is None:
        raise RuntimeError(
            "maia2 is unavailable and adapter checkpoint lacks policy index buffers; "
            "cannot construct fallback move mapping."
        )

    from_idx = state[from_key].detach().cpu().tolist()
    to_idx = state[to_key].detach().cpu().tolist()
    if len(from_idx) != len(to_idx):
        raise RuntimeError("Invalid checkpoint buffers: policy_from/to length mismatch.")

    dup_counts: Dict[str, int] = {}
    promo_cycle = ["q", "r", "b", "n"]
    vocab: List[str] = []
    for frm, to in zip(from_idx, to_idx):
        base = f"{_square_name(frm)}{_square_name(to)}"
        n = dup_counts.get(base, 0)
        dup_counts[base] = n + 1
        if n == 0:
            vocab.append(base)
        else:
            suffix = promo_cycle[(n - 1) % len(promo_cycle)]
            vocab.append(base + suffix)

    fallback = _FallbackMapping(vocab)
    chess_fusion_model.get_maia_mapping = lambda: fallback
    print("[load] maia2 not available: using fallback move mapping from checkpoint buffers.")
    return True


def _build_adapter(
    model_cfg: ModelConfig,
    adapter_path: Path,
    config_path: Path,
    device: torch.device,
) -> Tuple[chess_fusion_model.ChessFusionAdapter, LoadMeta]:
    state = torch.load(adapter_path, map_location="cpu", weights_only=False)
    fallback_used = _maybe_patch_maia_mapping_from_state(state)
    adapter = chess_fusion_model.ChessFusionAdapter(model_cfg, llm_dim=2048).to(device)
    missing, unexpected = adapter.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] Missing keys: {len(missing)} (example: {missing[:5]})")
    if unexpected:
        print(f"[load] Unexpected keys: {len(unexpected)} (example: {unexpected[:5]})")
    adapter.eval()
    meta = LoadMeta(
        fallback_mapping_used=fallback_used,
        missing_keys=len(missing),
        unexpected_keys=len(unexpected),
        missing_example=(str(missing[0]) if missing else ""),
        unexpected_example=(str(unexpected[0]) if unexpected else ""),
        checkpoint=str(adapter_path.parent),
        config=str(config_path),
        device=str(device),
    )
    return adapter, meta


def _build_legal_vocab_candidates(board: chess.Board, decoder: PolicyMoveDecoder) -> List[Tuple[int, chess.Move]]:
    candidates: List[Tuple[int, chess.Move]] = []
    for idx in range(decoder.vocab_size):
        mv = decoder.decode_legal_move(board, idx)
        if mv is not None:
            candidates.append((idx, mv))
    return candidates


@torch.no_grad()
def _pick_bot_move(
    board: chess.Board,
    adapter: chess_fusion_model.ChessFusionAdapter,
    device: torch.device,
    decoder: PolicyMoveDecoder,
    top_k: int = 5,
) -> Tuple[chess.Move, str]:
    features = extract_maia_features(board.fen()).unsqueeze(0).to(device)
    stm = torch.tensor([board.turn == chess.WHITE], dtype=torch.bool, device=device)
    out = adapter(features, side_to_move=stm)

    legal_candidates = _build_legal_vocab_candidates(board, decoder)
    if not legal_candidates:
        # Should never happen on non-terminal positions, but keep a hard fallback.
        fallback = next(iter(board.legal_moves))
        return fallback, "model: no legal policy candidates; fallback first legal move"

    policy_logits = out["policy_logits"][0]
    move_eval_logits = out.get("move_eval_logits", None)
    move_eval_logits = move_eval_logits[0] if isinstance(move_eval_logits, torch.Tensor) else None

    def _fmt(rows: List[Tuple[chess.Move, float]]) -> str:
        if not rows:
            return "-"
        return ", ".join(f"{m.uci()}={s:.3f}" for m, s in rows)

    best_idx = None
    best_score = None
    policy_rows: List[Tuple[chess.Move, float]] = []
    eval_rows: List[Tuple[chess.Move, float]] = []
    for vocab_idx, _mv in legal_candidates:
        pol_s = float(policy_logits[vocab_idx].item())
        policy_rows.append((_mv, pol_s))
        if move_eval_logits is not None:
            ev_s = float(move_eval_logits[vocab_idx].item())
            eval_rows.append((_mv, ev_s))
            score = ev_s
        else:
            score = pol_s
        if move_eval_logits is not None:
            pass
        if best_score is None or score > best_score:
            best_score = score
            best_idx = vocab_idx

    move = decoder.decode_legal_move(board, int(best_idx))
    if move is None:
        fallback = next(iter(board.legal_moves))
        return fallback, "model: decode failed; fallback first legal move"
    if move not in board.legal_moves:
        # Last-resort fallback if decoding fails unexpectedly.
        fallback = next(iter(board.legal_moves))
        return fallback, "model: illegal decode; fallback first legal move"

    policy_rows.sort(key=lambda x: x[1], reverse=True)
    eval_rows.sort(key=lambda x: x[1], reverse=True)
    chosen_source = "move_eval" if move_eval_logits is not None else "policy"
    chosen_policy = next((s for m, s in policy_rows if m == move), None)
    chosen_eval = next((s for m, s in eval_rows if m == move), None) if eval_rows else None
    comment = (
        f"model: source={chosen_source}; chosen={move.uci()}; "
        f"chosen_policy={(f'{chosen_policy:.3f}' if chosen_policy is not None else 'n/a')}; "
        f"chosen_eval={(f'{chosen_eval:.3f}' if chosen_eval is not None else 'n/a')}; "
        f"legal_candidates={len(legal_candidates)}; "
        f"policy_top{top_k}=[{_fmt(policy_rows[:top_k])}]; "
        f"eval_top{top_k}=[{_fmt(eval_rows[:top_k])}]"
    )
    return move, comment


def _configure_elo_engine(engine: chess.engine.SimpleEngine, elo: int) -> None:
    options = engine.options
    if "UCI_LimitStrength" in options:
        engine.configure({"UCI_LimitStrength": True})
    if "UCI_Elo" in options:
        engine.configure({"UCI_Elo": int(elo)})
    elif "Skill Level" in options:
        # Approximate mapping 0..20 ~ [800..2800]
        skill = int(round((max(800, min(2800, elo)) - 800) / 100.0))
        skill = max(0, min(20, skill))
        engine.configure({"Skill Level": skill})


def _score_from_outcome(result: str, bot_is_white: bool) -> float:
    if result == "1/2-1/2":
        return 0.5
    if result == "1-0":
        return 1.0 if bot_is_white else 0.0
    if result == "0-1":
        return 0.0 if bot_is_white else 1.0
    return 0.5


def _play_single_game(
    adapter: chess_fusion_model.ChessFusionAdapter,
    device: torch.device,
    decoder: PolicyMoveDecoder,
    engine_path: Path,
    opponent_elo: int,
    bot_is_white: bool,
    movetime_ms: int,
    max_plies: int,
    random_opening_plies: int,
    opening_seed: Optional[int],
    annotate_pgn: bool,
    annotation_topk: int,
    verbose: bool,
    load_meta: LoadMeta,
) -> Tuple[GameRecord, chess.pgn.Game]:
    board = chess.Board()
    try:
        engine = chess.engine.SimpleEngine.popen_uci(str(engine_path))
    except (FileNotFoundError, PermissionError) as e:
        raise RuntimeError(
            f"Failed to start UCI engine at '{engine_path}'. "
            f"Check the path/executable permissions. Original error: {e}"
        ) from e
    _configure_elo_engine(engine, opponent_elo)

    if opening_seed is not None and random_opening_plies > 0:
        rnd = random.Random(opening_seed)
        for _ in range(random_opening_plies):
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            board.push(rnd.choice(moves))

    game = chess.pgn.Game()
    game.headers["White"] = "ModelBot" if bot_is_white else f"UCI_{opponent_elo}"
    game.headers["Black"] = f"UCI_{opponent_elo}" if bot_is_white else "ModelBot"
    game.headers["ModelCheckpoint"] = load_meta.checkpoint
    game.headers["ModelConfig"] = load_meta.config
    game.headers["ModelDevice"] = load_meta.device
    game.headers["ModelLoadFallbackMapping"] = str(load_meta.fallback_mapping_used)
    game.headers["ModelLoadMissingKeys"] = str(load_meta.missing_keys)
    game.headers["ModelLoadUnexpectedKeys"] = str(load_meta.unexpected_keys)
    if load_meta.missing_example:
        game.headers["ModelLoadMissingExample"] = load_meta.missing_example
    if load_meta.unexpected_example:
        game.headers["ModelLoadUnexpectedExample"] = load_meta.unexpected_example
    node = game

    limit = chess.engine.Limit(time=max(0.01, movetime_ms / 1000.0))
    try:
        for _ in range(max_plies):
            if board.is_game_over():
                break

            bot_to_move = (board.turn == chess.WHITE and bot_is_white) or (
                board.turn == chess.BLACK and not bot_is_white
            )
            move_comment = ""
            if bot_to_move:
                move, move_comment = _pick_bot_move(
                    board,
                    adapter,
                    device,
                    decoder,
                    top_k=annotation_topk,
                )
                if verbose:
                    print(f"    Model move {move.uci()} | {move_comment}")
            else:
                result = engine.play(board, limit)
                move = result.move
                if move is None:
                    break
                if annotate_pgn:
                    move_comment = f"engine: elo={opponent_elo}; move={move.uci()}"

            board.push(move)
            node = node.add_variation(move)
            if annotate_pgn and move_comment:
                node.comment = move_comment
    finally:
        engine.quit()

    outcome = board.outcome(claim_draw=True)
    if outcome is None:
        result_str = "1/2-1/2"
    else:
        result_str = outcome.result()

    game.headers["Result"] = result_str
    record = GameRecord(
        opponent_elo=opponent_elo,
        bot_is_white=bot_is_white,
        result=result_str,
        score=_score_from_outcome(result_str, bot_is_white),
        ply_count=len(board.move_stack),
    )
    return record, game


def _estimate_elo(records: Sequence[GameRecord], lo: float = 200.0, hi: float = 3200.0) -> float:
    """Estimate rating with simple MLE under logistic Elo expected-score model."""
    if not records:
        return float("nan")

    # Solve sum(score - expected) = 0 by bisection.
    def balance(rating: float) -> float:
        total = 0.0
        for rec in records:
            exp = 1.0 / (1.0 + 10.0 ** ((rec.opponent_elo - rating) / 400.0))
            total += (rec.score - exp)
        return total

    f_lo = balance(lo)
    f_hi = balance(hi)
    if f_lo > 0:
        return lo
    if f_hi < 0:
        return hi

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = balance(mid)
        if f_mid > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _print_summary(records: Sequence[GameRecord]) -> None:
    if not records:
        print("No games played.")
        return

    by_elo: Dict[int, List[GameRecord]] = {}
    for rec in records:
        by_elo.setdefault(rec.opponent_elo, []).append(rec)

    total_score = sum(r.score for r in records)
    total_games = len(records)
    print("\n=== Benchmark Summary ===")
    print(f"Games: {total_games}")
    print(f"Score: {total_score:.1f}/{total_games} ({100.0 * total_score / total_games:.1f}%)")
    print("Per-opponent:")
    for elo in sorted(by_elo):
        rows = by_elo[elo]
        s = sum(r.score for r in rows)
        n = len(rows)
        wins = sum(1 for r in rows if r.score == 1.0)
        draws = sum(1 for r in rows if r.score == 0.5)
        losses = sum(1 for r in rows if r.score == 0.0)
        print(f"  Elo {elo}: {s:.1f}/{n}  W/D/L={wins}/{draws}/{losses}")

    est = _estimate_elo(records)
    print(f"\nEstimated Elo (logistic MLE): {est:.0f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark policy-only adapter Elo vs UCI bots.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint dir (or root with epoch/checkpoint subdirs).")
    parser.add_argument("--config", type=Path, default=None, help="YAML config path. Defaults to checkpoint/config.yaml.")
    parser.add_argument("--engine", type=Path, required=True, help="UCI engine executable (e.g. stockfish).")
    parser.add_argument("--elos", type=int, nargs="+", default=[800, 1000, 1200, 1400, 1600, 1800], help="Opponent Elo levels.")
    parser.add_argument("--games-per-elo", type=int, default=6, help="Games at each Elo (colors alternate).")
    parser.add_argument("--movetime-ms", type=int, default=75, help="Opponent engine move time per move (ms).")
    parser.add_argument("--max-plies", type=int, default=220, help="Max plies per game before adjudicated draw.")
    parser.add_argument("--random-opening-plies", type=int, default=0, help="Optional random opening plies before both sides start.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed (for color/opening randomization).")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"), help="Torch device.")
    parser.add_argument("--pgn-out", type=Path, default=None, help="Optional PGN output path.")
    parser.add_argument("--annotate-pgn", action="store_true", default=True, help="Add verbose move annotations/comments to PGN.")
    parser.add_argument("--no-annotate-pgn", action="store_false", dest="annotate_pgn", help="Disable PGN move annotations.")
    parser.add_argument("--annotation-topk", type=int, default=5, help="Top-K moves to include in model annotations.")
    parser.add_argument("--verbose", action="store_true", help="Print per-move model decision logs during games.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint)
    config_path = _find_config_path(checkpoint_dir, args.config)
    engine_path = _resolve_engine_path(args.engine)
    training_cfg = load_config(str(config_path))
    model_cfg = training_cfg.model
    if model_cfg.mode not in ("chess_fusion", "policy_only"):
        model_cfg.mode = "chess_fusion"

    device = torch.device(args.device)
    print(f"Checkpoint: {checkpoint_dir}")
    print(f"Config: {config_path}")
    print(f"Device: {device}")
    print(f"Engine: {engine_path}")

    adapter, load_meta = _build_adapter(model_cfg, checkpoint_dir / "adapter.pt", config_path, device)
    decoder = PolicyMoveDecoder(adapter)

    records: List[GameRecord] = []
    pgn_games: List[chess.pgn.Game] = []

    for elo in args.elos:
        print(f"\n[Opponent Elo {elo}]")
        for g in range(args.games_per_elo):
            bot_is_white = (g % 2 == 0)
            record, game = _play_single_game(
                adapter=adapter,
                device=device,
                decoder=decoder,
                engine_path=engine_path,
                opponent_elo=int(elo),
                bot_is_white=bot_is_white,
                movetime_ms=args.movetime_ms,
                max_plies=args.max_plies,
                random_opening_plies=args.random_opening_plies,
                opening_seed=args.seed + (elo * 1000) + g,
                annotate_pgn=args.annotate_pgn,
                annotation_topk=max(1, args.annotation_topk),
                verbose=args.verbose,
                load_meta=load_meta,
            )
            records.append(record)
            pgn_games.append(game)
            side = "White" if bot_is_white else "Black"
            print(
                f"  Game {g+1}/{args.games_per_elo} as {side}: "
                f"{record.result} ({record.score:.1f}) plies={record.ply_count}"
            )

    _print_summary(records)

    if args.pgn_out is not None:
        args.pgn_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pgn_out.open("w", encoding="utf-8") as f:
            for game in pgn_games:
                print(game, file=f, end="\n\n")
        print(f"Wrote PGN: {args.pgn_out}")


if __name__ == "__main__":
    main()
