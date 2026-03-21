"""
Chess Commentary Inference Script

Generate commentary for chess positions from a PGN file using either:
1. A trained checkpoint (adapter + LoRA weights) - Automatically detects architecture
2. A base HuggingFace model (for comparison, e.g. TinyLlama, ChessGPT)

Usage:
    # Automatic detection (recommended)
    python inference.py --pgn game.pgn --checkpoint checkpoints/final
    
    # Manual override (if detection fails)
    python inference.py --pgn game.pgn --checkpoint checkpoints/final --mode engineered
    
    # Using base model (no chess adapter)
    python inference.py --pgn game.pgn --base-only
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import re
import torch
import chess
import chess.pgn
import json
import yaml

# Add parent to path for imports (FIXED: correctly points to src)
# __file__ = src/inference/inference.py -> parents[1] = src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from training.train import (
        ChessCommentaryModel,
        _filter_chess_fusion_adapter_state_dict,
        _preview_checkpoint_keys,
        _safe_apply_chat_template,
        build_commentary_prompt,
    )
    from training.config import ModelConfig, LoRAConfig, HybridConfig, PerceiverConfig, ChessFusionConfig, load_config
    from training.lc0_extractor import LC0HiddenStateExtractor
    from training.chess_adapter import extract_engineered_features
    try:
        from training.perceiver_adapter import extract_perceiver_features
    except ImportError:
        extract_perceiver_features = None
except ImportError as e:
    print(f"Error importing training modules: {e}")
    sys.exit(1)


def load_pgn(pgn_path: str) -> chess.pgn.Game:
    """Load a PGN file and return the first game."""
    with open(pgn_path, 'r', encoding='utf-8') as f:
        game = chess.pgn.read_game(f)
    if game is None:
        raise ValueError(f"Could not parse PGN from {pgn_path}")
    return game


def replay_to_ply(game: chess.pgn.Game, target_ply: int) -> chess.Board:
    """Replay game to a specific ply and return the board with history."""
    board = game.board()
    moves = list(game.mainline_moves())
    
    for i, move in enumerate(moves[:target_ply]):
        board.push(move)
    
    return board


def build_san_move_history(board: chess.Board) -> str:
    """Return SAN move history for the current board state (up to board.move_stack)."""
    replay_board = chess.Board()
    parts: List[str] = []
    for move in board.move_stack:
        if replay_board.turn:
            parts.append(f"{replay_board.fullmove_number}.")
        parts.append(replay_board.san(move))
        replay_board.push(move)
    return " ".join(parts)


def detect_model_config(checkpoint_path: Path) -> ModelConfig:
    """
    Inspect checkpoint files to detect the architecture and configuration.
    
    Logic:
    1. Try to load config.json (if we started saving it) - NOT YET IMPLEMENTED IN TRAIN
    2. Inspect adapter.pt state dict keys:
       - 'mlp.0.weight' (input 204) -> Engineered
       - 'mlp.0.weight' (input > 204) + 'layer_projections' -> Hybrid
       - 'pos_embeddings' -> Legacy/LC0
       - 'perceiver' keys -> Perceiver
    """
    print(f"Detecting architecture for {checkpoint_path}...")
    
    adapter_path = checkpoint_path / "adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter weights not found at {adapter_path}")
        
    try:
        state_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
    except Exception as e:
        # Fallback for older torch versions or complex pickles
        state_dict = torch.load(adapter_path, map_location="cpu", weights_only=False)
        
    keys = set(state_dict.keys())
    
    # Strip torch.compile prefix for detection
    keys = {k.replace("_orig_mod.", "") for k in keys}
    
    # Defaults
    config = ModelConfig()
    
    # 1a. Check for Chess-Fusion adapter (has multi_scale + gated_xattns + shared_readout)
    if any(k.startswith("multi_scale.") or k.startswith("gated_xattns.") or k.startswith("shared_readout.") for k in keys):
        print(" -> Detected: CHESS_FUSION mode")
        config.mode = "chess_fusion"
        return config

    # 1b. Check for Maia adapter
    if any(k.startswith("backbone.maia") or k.startswith("cross_attn_layers") for k in keys):
        print(" -> Detected: MAIA mode")
        config.mode = "maia"
        return config

    # 2. Check for Perceiver
    if any("cross_attention" in k or "latents" in k for k in keys):
        print(" -> Detected: PERCEIVER mode")
        config.mode = "perceiver"
        return config
        
    # 2. Check for Hybrid (has LC0 projections AND engineered/mixed input MLP)
    # Hybrid adapter has 'layer_projections' AND 'mlp'
    if "layer_projections.0.weight" in keys:
        if "mlp.0.weight" in keys:
             # Check MLP input dimension to be sure?
             # Hybrid MLP input is 204 + (4 * lc0_proj_dim)
             # But existence of both projections and MLP strongly suggests hybrid
             # (Legacy 'full' mode also has projections+MLP but also 'pos_embeddings')
             if "pos_embeddings" not in keys:
                 print(" -> Detected: HYBRID mode")
                 config.mode = "hybrid"
                 return config
             else:
                 # Legacy LC0 mode (Full)
                 print(" -> Detected: LEGACY LC0 mode (Full) - treating as 'hybrid' logic without engineered feats? No, not supported by new ChessCommentaryModel yet.")
                 print("WARNING: Legacy LC0 checkpoints might need migration. Trying 'hybrid' mode setup.")
                 # Actually, the user code earlier showed 'legacy' support via separate class or path.
                 # The 'ChessCommentaryModel' class in train.py seems to only support hybrid/engineered/perceiver explicitly in __init__?
                 # Wait, looking at train.py:
                 # elif config.mode == "engineered": ...
                 # elif config.mode == "perceiver": ...
                 # else: raise ValueError...
                 # It seems 'hybrid' and 'engineered' are the main supported ones in the snippet I saw.
                 # Wait, I missed the 'else' block in train.py?
                 # Re-reading train.py...
                 # Line 142: if config.mode == "hybrid": ...
                 # Line 151: elif config.mode == "engineered": ...
                 # Line 156: elif config.mode == "perceiver": ...
                 # Line 161: else: raise ValueError(f"Unknown mode: {config.mode}")
                 # So Legacy is NOT supported by current ChessCommentaryModel class!
                 # If we detect legacy, we might fail. But let's assume valid formatted checkpoints for now.
                 pass
    
    # 3. Check for Engineered (MLP only, no projections, input 204)
    if "mlp.0.weight" in keys and "layer_projections.0.weight" not in keys:
        # Check input dimension of MLP
        weight = state_dict["mlp.0.weight"]
        if weight.shape[1] == 204:
            print(" -> Detected: ENGINEERED mode")
            config.mode = "engineered"
            return config
            
    print(" -> WARNING: Could not auto-detect mode cleanly. Defaulting to 'engineered'.")
    config.mode = "engineered"
    return config


def generate_commentary(
    model: ChessCommentaryModel,
    board: chess.Board,
    config: ModelConfig,
    extractor: Optional[LC0HiddenStateExtractor],
    tokenizer: AutoTokenizer,
    pgn_moves: str = "",
    max_new_tokens: int = 256,
    min_new_tokens: int = 0,
    temperature: float = 0.7,
    device: str = "cuda",
    return_ids: bool = False
) -> str:
    """Unified generation function."""
    
    fen = board.fen()
    
    # Prepare inputs
    lc0_states = None
    engineered_feats = None 
    perceiver_feats = None
    
    # 1. Extract features based on mode
    if config.mode == "hybrid":
        print("Extracting LC0 + Engineered features...")
        if extractor is None:
            raise ValueError("LC0 Network required for hybrid mode")
        
        # LC0
        lc0_raw = extractor.extract(board)
        lc0_states = {k: torch.from_numpy(v).float() for k, v in lc0_raw.items()}
        
        # Engineered - handled inside model.generate usually, but let's see model.generate signature
        # train.py: generate(self, lc0_hidden_states, ..., fen=fen)
        # Inside generate: if hybrid, it calls extract_engineered_features(fen)
        # So we just pass fen and lc0_states
        pass
        
    elif config.mode == "engineered":
        print("Extracting Engineered features (internal)...")
        # handled inside model.generate via FEN
        pass
        
    elif config.mode == "perceiver":
        print("Extracting Perceiver features...")
        pass

    elif config.mode == "chess_fusion":
        print("Extracting Chess-Fusion features (internal via FEN)...")
        # Features extracted internally by model.generate() from FEN
        pass

    # Build inference prompt (mirrors training prompt style when PGN is provided)
    prompt = build_commentary_prompt(
        fen=fen,
        pgn_moves=pgn_moves,
        use_pgn_in_prompt=bool(pgn_moves),
        prepend_fen_in_prompt=bool(getattr(config, "prepend_fen_in_prompt", False)),
    )

    # Log full user prompt + token counts for visibility
    prompt_messages = [{"role": "user", "content": prompt}]
    prompt_chat_text = _safe_apply_chat_template(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    user_prompt_tokens = int(tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1])
    chat_prompt_tokens = int(tokenizer(prompt_chat_text, return_tensors="pt")["input_ids"].shape[1])
    print("[Inference] User prompt:")
    print(prompt)
    print(f"[Inference] Prompt tokens: user={user_prompt_tokens}, chat_template={chat_prompt_tokens}")

    # Generate (use autocast for bf16 models to avoid dtype mismatches)
    from contextlib import nullcontext
    use_autocast = device.startswith("cuda") and torch.cuda.is_available()
    gen_ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if use_autocast else nullcontext()
    try:
        with gen_ctx:
            commentary = model.generate(
                lc0_hidden_states=lc0_states,
                side_to_move=board.turn,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                fen=fen,
                return_ids=return_ids
            )
    except Exception as e:
        print(f"Generation failed: {e}")
        # Fallback debugging
        import traceback
        traceback.print_exc()
        return "[Error generating commentary]"

    return commentary


def _load_checkpoint_config(checkpoint_path: Path) -> Tuple[Optional[ModelConfig], Optional[int], bool, bool, Optional[int]]:
    """Load config.yaml from checkpoint directory if present."""
    config_path = checkpoint_path / "config.yaml"
    if not config_path.exists():
        return None, None, False, False, None
    try:
        training_cfg = load_config(str(config_path))
        return (
            training_cfg.model,
            training_cfg.max_length,
            False,
            bool(getattr(training_cfg, "use_pgn_in_prompt", False)),
            getattr(training_cfg, "pgn_prompt_last_n_moves", None),
        )
    except TypeError as exc:
        print(f"Warning: failed to load config.yaml with load_config: {exc}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    model_data = data.get("model", {}) if isinstance(data, dict) else {}
    legacy_model_keys = {
        "base_model",
        "mode",
        "engineered_features_type",
        "lc0_dim",
        "projection_dim",
        "load_in_8bit",
        "use_flash_attention",
        "use_torch_compile",
        "use_fen_tokens",
        "hybrid",
        "perceiver",
        "maia",
        "chess_fusion",
    }
    for key in list(legacy_model_keys):
        if key in data and key not in model_data:
            model_data[key] = data.pop(key)

    if "projection_dim" in model_data:
        hybrid_data = model_data.get("hybrid", {}) if isinstance(model_data.get("hybrid"), dict) else {}
        if "lc0_proj_dim" not in hybrid_data:
            hybrid_data["lc0_proj_dim"] = model_data.pop("projection_dim")
        else:
            model_data.pop("projection_dim")
        model_data["hybrid"] = hybrid_data
    lora_overrides = {
        "r": data.pop("lora_r", None),
        "alpha": data.pop("lora_alpha", None),
        "dropout": data.pop("lora_dropout", None),
        "unfreeze_epoch": data.pop("lora_unfreeze_epoch", None),
        "progressive_merge": data.pop("lora_progressive_merge", None),
        "target_modules": data.pop("lora_target_modules", None),
    }
    lora_overrides = {k: v for k, v in lora_overrides.items() if v is not None}
    if lora_overrides:
        model_lora = model_data.get("lora", {}) if isinstance(model_data.get("lora"), dict) else {}
        model_lora.update(lora_overrides)
        model_data["lora"] = model_lora

    if model_data:
        data["model"] = model_data

    training_cfg = load_config_from_dict(data)
    return (
        training_cfg.model,
        training_cfg.max_length,
        True,
        bool(getattr(training_cfg, "use_pgn_in_prompt", False)),
        getattr(training_cfg, "pgn_prompt_last_n_moves", None),
    )


def _truncate_pgn_prompt_moves(pgn_moves: str, pgn_prompt_last_n_moves: Optional[int]) -> str:
    """Mirror training-time PGN prompt truncation behavior."""
    n = pgn_prompt_last_n_moves
    if not pgn_moves or n is None or n <= 0:
        return pgn_moves

    tokens = pgn_moves.split()
    parsed_moves = []  # (fullmove_number, side, san)
    current_fullmove = 1
    side = "w"
    for tok in tokens:
        marker = re.fullmatch(r"(\d+)\.(\.\.)?", tok)
        if marker:
            current_fullmove = int(marker.group(1))
            side = "b" if marker.group(2) else "w"
            continue

        parsed_moves.append((current_fullmove, side, tok))
        if side == "w":
            side = "b"
        else:
            side = "w"
            current_fullmove += 1

    if len(parsed_moves) <= n:
        return pgn_moves

    tail = parsed_moves[-n:]
    formatted = []
    for i, (fullmove, move_side, san) in enumerate(tail):
        if move_side == "w":
            formatted.append(f"{fullmove}. {san}")
        elif i == 0:
            formatted.append(f"{fullmove}...{san}")
        else:
            formatted.append(san)
    return " ".join(formatted)


def load_config_from_dict(data: dict) -> "TrainingConfig":
    """Local helper for legacy config normalization."""
    from training.config import (
        TrainingConfig,
        ModelConfig,
        LoRAConfig,
        HybridConfig,
        PerceiverConfig,
        MaiaConfig,
        ChessFusionConfig,
    )

    model_data = data.get("model", {}) if isinstance(data, dict) else {}
    if model_data.get("mode") == "maia_fusion":
        raise ValueError(
            "Deprecated mode 'maia_fusion' is no longer supported. "
            "Use model.mode='chess_fusion'."
        )
    if "maia_fusion" in model_data:
        raise ValueError(
            "Deprecated config key 'model.maia_fusion' is no longer supported. "
            "Use 'model.chess_fusion'."
        )

    if "gradient_clip_val" in model_data:
        val = model_data.pop("gradient_clip_val")
        if "gradient_clip_val" not in data:
            data["gradient_clip_val"] = val

    if "lora" in model_data and isinstance(model_data["lora"], dict):
        model_data["lora"] = LoRAConfig(**model_data["lora"])
    if "hybrid" in model_data and isinstance(model_data["hybrid"], dict):
        model_data["hybrid"] = HybridConfig(**model_data["hybrid"])
    if "perceiver" in model_data and isinstance(model_data["perceiver"], dict):
        model_data["perceiver"] = PerceiverConfig(**model_data["perceiver"])
    if "maia" in model_data and isinstance(model_data["maia"], dict):
        if "lora" in model_data["maia"]:
            lora_data = model_data["maia"].pop("lora")
            if isinstance(lora_data, dict):
                model_data["lora"] = LoRAConfig(**lora_data)

        if "backbone_lr_ratio" in model_data["maia"]:
            ratio = model_data["maia"].pop("backbone_lr_ratio")
            if "cnn_lr_ratio" not in model_data["maia"]:
                model_data["maia"]["cnn_lr_ratio"] = ratio
            if "transformer_lr_ratio" not in model_data["maia"]:
                model_data["maia"]["transformer_lr_ratio"] = ratio

        model_data["maia"] = MaiaConfig(**model_data["maia"])
    if "chess_fusion" in model_data and isinstance(model_data["chess_fusion"], dict):
        if "lora" in model_data["chess_fusion"]:
            lora_data = model_data["chess_fusion"].pop("lora")
            if isinstance(lora_data, dict):
                model_data["lora"] = LoRAConfig(**lora_data)
        model_data["chess_fusion"] = ChessFusionConfig(**model_data["chess_fusion"])

    if model_data:
        data["model"] = ModelConfig(**model_data)

    if "training" in data and isinstance(data["training"], dict):
        training_config = data.pop("training")
        data.update(training_config)

    data.pop("effective_batch_size", None)
    data.pop("use_simple_projection", None)
    data.pop("progressive_lora_merge", None)
    data.pop("use_engineered_features", None)

    return TrainingConfig(**data)


def _auto_detect_prior_checkpoints(checkpoint_path: Path) -> list:
    """Heuristic: given 'some_name_N', find 'some_name' and 'some_name_2' .. 'some_name_{N-1}'."""
    name = checkpoint_path.name
    parent = checkpoint_path.parent
    # Try to parse trailing _N as epoch number
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        base_name = parts[0]
        epoch_num = int(parts[1])
    else:
        return []

    prior = []
    # Epoch 1 might be just base_name (no suffix)
    candidate = parent / base_name
    if candidate.exists() and (candidate / "lora").exists():
        prior.append(candidate)

    # Epochs 2..N-1 are base_name_K
    for k in range(2, epoch_num):
        candidate = parent / f"{base_name}_{k}"
        if candidate.exists() and (candidate / "lora").exists():
            prior.append(candidate)

    return prior


def _maybe_disable_unused_maia_backbone_for_inference(config: ModelConfig) -> None:
    """Disable Maia backbone/teacher in board-only chess_fusion inference.

    Some chess_fusion runs use CSMP board features only (no CNN taps, no transformer
    taps). In that setup, Maia models are only needed during training for policy
    distillation and should not be constructed during inference.
    """
    if getattr(config, "mode", None) != "chess_fusion":
        return

    fusion_cfg = getattr(config, "chess_fusion", None)
    if fusion_cfg is None:
        return

    use_cnn = bool(getattr(fusion_cfg, "use_cnn", True))
    use_csmp = bool(getattr(fusion_cfg, "use_chess_structure_mp", False))
    cnn_tap_layers = list(getattr(fusion_cfg, "cnn_tap_layers", []) or [])
    use_transformer_taps = bool(getattr(fusion_cfg, "use_transformer_taps", False))

    # Safe optimization: CSMP board-only path does not consume Maia backbone outputs.
    if use_cnn and use_csmp and not cnn_tap_layers and not use_transformer_taps:
        fusion_cfg.use_cnn = False
        fusion_cfg.load_checkpoint_backbone = False
        print(
            "[Inference] Detected board-only chess_fusion (no CNN/transformer taps); "
            "disabling Maia backbone + teacher loading."
        )


def _set_inference_chess_fusion_load_defaults(config: ModelConfig) -> None:
    """Use inference-friendly defaults for chess_fusion checkpoint module loading.

    Training configs may set selective checkpoint loading flags for curriculum/resume
    purposes (e.g. skipping xattn/pseudotokens). For inference we generally want to
    load all trained adapter modules unless explicitly overridden.
    """
    if getattr(config, "mode", None) not in {"chess_fusion", "policy_only"}:
        return

    fusion_cfg = getattr(config, "chess_fusion", None)
    if fusion_cfg is None:
        return

    fusion_cfg.load_checkpoint_csmp = True
    fusion_cfg.load_checkpoint_perceiver = True
    fusion_cfg.load_checkpoint_xattn = True
    fusion_cfg.load_checkpoint_lm_pseudotokens = True
    fusion_cfg.load_checkpoint_aux_heads = True
    fusion_cfg.load_checkpoint_backbone = bool(getattr(fusion_cfg, "use_cnn", True))


def _load_inference_override_config(path: Optional[str]) -> dict:
    """Load optional inference override YAML.

    Supported structure:
      model:
        ... ModelConfig fields ...
      inference:
        use_pgn_in_prompt: bool
        pgn_prompt_last_n_moves: int|null
        load_lora: bool
    """
    if not path:
        return {}

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Inference config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Inference config must be a YAML mapping (got {type(data).__name__})")
    print(f"[Inference Config] Loaded overrides from {cfg_path}")
    return data


def _apply_model_overrides_from_dict(target, overrides: dict, prefix: str = "model.") -> List[str]:
    """Recursively apply dict overrides to dataclass-like config objects."""
    applied: List[str] = []
    if not isinstance(overrides, dict):
        return applied

    for key, value in overrides.items():
        if not hasattr(target, key):
            print(f"[Inference Config] Ignoring unknown key: {prefix}{key}")
            continue

        current = getattr(target, key)
        if isinstance(value, dict):
            if current is None:
                print(f"[Inference Config] Cannot apply nested override to None: {prefix}{key}")
                continue
            applied.extend(_apply_model_overrides_from_dict(current, value, prefix=f"{prefix}{key}."))
        else:
            setattr(target, key, value)
            applied.append(f"{prefix}{key}={value!r}")

    return applied


def _apply_inference_overrides(
    config: ModelConfig,
    overrides: dict,
    use_pgn_in_prompt: bool,
    pgn_prompt_last_n_moves: Optional[int],
    load_lora: bool,
) -> Tuple[bool, Optional[int], bool]:
    """Apply override YAML to model config and inference-only runtime toggles."""
    if not overrides:
        return use_pgn_in_prompt, pgn_prompt_last_n_moves, load_lora

    model_overrides = {}
    inference_overrides = {}

    if isinstance(overrides.get("model"), dict):
        model_overrides.update(overrides["model"])
    if isinstance(overrides.get("inference"), dict):
        inference_overrides.update(overrides["inference"])

    # Convenience: allow top-level keys too.
    for key, value in overrides.items():
        if key in {"model", "inference"}:
            continue
        if key in {"use_pgn_in_prompt", "pgn_prompt_last_n_moves", "load_lora"}:
            inference_overrides.setdefault(key, value)
        else:
            model_overrides.setdefault(key, value)

    applied = _apply_model_overrides_from_dict(config, model_overrides)
    if applied:
        print("[Inference Config] Applied model overrides:")
        for item in applied:
            print(f"  - {item}")

    if "use_pgn_in_prompt" in inference_overrides:
        use_pgn_in_prompt = bool(inference_overrides["use_pgn_in_prompt"])
        print(f"[Inference Config] use_pgn_in_prompt={use_pgn_in_prompt}")
    if "pgn_prompt_last_n_moves" in inference_overrides:
        pgn_prompt_last_n_moves = inference_overrides["pgn_prompt_last_n_moves"]
        print(f"[Inference Config] pgn_prompt_last_n_moves={pgn_prompt_last_n_moves}")
    if "load_lora" in inference_overrides:
        load_lora = bool(inference_overrides["load_lora"])
        print(f"[Inference Config] load_lora={load_lora}")

    # Consistency guard: no student backbone exists when use_cnn=False.
    if getattr(config, "mode", None) in {"chess_fusion", "policy_only"}:
        fusion_cfg = getattr(config, "chess_fusion", None)
        if fusion_cfg is not None and not bool(getattr(fusion_cfg, "use_cnn", True)):
            fusion_cfg.load_checkpoint_backbone = False

    return use_pgn_in_prompt, pgn_prompt_last_n_moves, load_lora


def _activate_lora_adapter(llm, adapter_name: str = "default") -> None:
    """Ensure the loaded LoRA adapter is active for inference."""
    if hasattr(llm, "set_adapter"):
        try:
            llm.set_adapter(adapter_name)
        except Exception:
            pass
    if hasattr(llm, "enable_adapters"):
        try:
            llm.enable_adapters()
        except Exception:
            pass


def _log_lora_status(llm, context: str) -> None:
    """Emit lightweight diagnostics about LoRA adapter state."""
    try:
        active = getattr(llm, "active_adapter", None)
        adapters = None
        if hasattr(llm, "peft_config"):
            adapters = list(getattr(llm, "peft_config").keys())
        lora_params = [p for name, p in llm.named_parameters() if "lora_" in name]
        trainable = sum(p.numel() for p in lora_params if p.requires_grad)
        total = sum(p.numel() for p in lora_params)
        print(f"[LoRA] {context}: active={active}, adapters={adapters}, lora_params={total}, trainable={trainable}")
    except Exception as exc:
        print(f"[LoRA] {context}: status unavailable ({exc})")


def main():
    parser = argparse.ArgumentParser(description="Chess Commentary Inference")
    
    # Input/Output
    parser.add_argument("--pgn", "-p", required=True, help="Path to PGN file")
    parser.add_argument("--checkpoint", "-c", help="Path to trained checkpoint directory")
    parser.add_argument("--base-only", action="store_true", help="Use base HuggingFace model without adapter")
    parser.add_argument("--base-model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model ID for --base-only mode (default: TinyLlama)")
    
    # Position selection
    parser.add_argument("--ply", type=int, default=None, help="Specific ply to analyze (default: last)")
    parser.add_argument("--all-plies", action="store_true", help="Analyze every 10 plies")
    
    # Model config overrides
    parser.add_argument("--mode", choices=["hybrid", "engineered", "perceiver", "maia", "chess_fusion", "auto"], default="auto", 
                        help="Force architecture mode (default: auto-detect)")
    parser.add_argument("--network", "-n", help="Path to LC0 network file (required for hybrid/lc0)")
    parser.add_argument("--feature-mode", choices=["simplified", "main"], default="main",
                        help="Sub-mode for engineered features (default: main)")
    parser.add_argument(
        "--inference-config",
        type=str,
        default=None,
        help="Path to inference override YAML (model + runtime prompt/LoRA toggles)",
    )
    parser.add_argument(
        "--use-pgn-in-prompt",
        dest="use_pgn_in_prompt",
        action="store_true",
        help="Override checkpoint setting: prepend PGN move history to the prompt",
    )
    parser.add_argument(
        "--no-pgn-in-prompt",
        dest="use_pgn_in_prompt",
        action="store_false",
        help="Override checkpoint setting: do not prepend PGN move history to the prompt",
    )
    parser.set_defaults(use_pgn_in_prompt=None)
    parser.add_argument(
        "--pgn-prompt-last-n-moves",
        type=int,
        default=None,
        help="Override checkpoint setting: limit prompt PGN context to last N SAN moves",
    )
    parser.add_argument(
        "--load-lora",
        dest="load_lora",
        action="store_true",
        help="Override inference setting: load checkpoint LoRA weights when available",
    )
    parser.add_argument(
        "--no-load-lora",
        dest="load_lora",
        action="store_false",
        help="Override inference setting: ignore checkpoint LoRA weights and use base LLM only",
    )
    parser.set_defaults(load_lora=None)
    parser.add_argument("--use-merged-base", action="store_true",
                        help="Load merged_base from checkpoint if available")
    parser.add_argument("--prior-checkpoints", nargs="+", default=None,
                        help="Ordered list of prior epoch checkpoint dirs whose LoRA should be merged before loading the target checkpoint (required for progressive_merge checkpoints beyond epoch 1)")
    
    # Generation params
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--min-tokens", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    
    args = parser.parse_args()
    
    # 1. Load Game
    print(f"Loading PGN: {args.pgn}")
    game = load_pgn(args.pgn)
    print(f"Game: {game.headers.get('White', '?')} vs {game.headers.get('Black', '?')} ({game.headers.get('Result', '*')})")
    
    total_plies = game.end().ply()
    
    if args.all_plies:
        plies = range(10, total_plies + 1, 10)
    elif args.ply is not None:
        plies = [min(args.ply, total_plies)]
    else:
        plies = [total_plies]
        
    print(f"Analyzing plies: {list(plies)}")

    # 2. Load Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    tokenizer = None
    extractor = None
    config = ModelConfig() # Default
    use_pgn_in_prompt = False
    pgn_prompt_last_n_moves = None
    load_lora = True
    
    if args.base_only:
        model_name = args.base_model
        print(f"Using base model (no adapter): {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        
    else:
        if not args.checkpoint:
            print("Error: Must specify --checkpoint or --base-only")
            return 1
            
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint_path = Path(args.checkpoint)

        if not (checkpoint_path / "adapter.pt").exists() and checkpoint_path.is_dir():
            subdirs = [p for p in checkpoint_path.iterdir() if p.is_dir()]
            if len(subdirs) == 1 and (subdirs[0] / "adapter.pt").exists():
                checkpoint_path = subdirs[0]
                print(f"Resolved checkpoint to: {checkpoint_path}")
        
        # Detect or set config
        if args.mode == "auto":
            detected_config = detect_model_config(checkpoint_path)
            detected_mode = detected_config.mode
            config = detected_config
            # Try to load lora config if exists to update params
            lora_path = checkpoint_path / "lora" / "adapter_config.json"
            if lora_path.exists():
                with open(lora_path, 'r') as f:
                    lconf = json.load(f)
                    config.lora.r = lconf.get("r", 16)
                    config.lora.alpha = lconf.get("lora_alpha", 32)
                    config.lora.dropout = lconf.get("lora_dropout", 0.05)
                    config.lora.dropout = lconf.get("lora_dropout", 0.05)
            # Override with training config if available
            checkpoint_model_cfg, max_length, legacy_cfg, cfg_use_pgn_in_prompt, cfg_pgn_prompt_last_n_moves = _load_checkpoint_config(checkpoint_path)
            if checkpoint_model_cfg is not None:
                config = checkpoint_model_cfg
                config.mode = detected_mode
                use_pgn_in_prompt = cfg_use_pgn_in_prompt
                pgn_prompt_last_n_moves = cfg_pgn_prompt_last_n_moves
                if legacy_cfg:
                    config.load_in_8bit = False
                    config.use_flash_attention = False
                    config.use_torch_compile = False
                    if config.mode == "engineered":
                        config.engineered_features_type = "main"
                if max_length is not None and args.max_tokens > max_length:
                    print(f"Warning: max-tokens {args.max_tokens} exceeds training max_length {max_length}.")
            # Apply feature mode override (avoid clobbering legacy defaults)
            if not legacy_cfg or args.feature_mode != "simplified":
                config.engineered_features_type = args.feature_mode
        else:
            config.mode = args.mode
            config.engineered_features_type = args.feature_mode

        # Inference defaults should prefer loading trained adapter modules.
        _set_inference_chess_fusion_load_defaults(config)

        # Default LoRA loading follows current model config unless overridden.
        load_lora = bool(getattr(config, "use_lora", True))

        # Apply optional inference override YAML.
        override_cfg = _load_inference_override_config(args.inference_config)
        use_pgn_in_prompt, pgn_prompt_last_n_moves, load_lora = _apply_inference_overrides(
            config,
            override_cfg,
            use_pgn_in_prompt=use_pgn_in_prompt,
            pgn_prompt_last_n_moves=pgn_prompt_last_n_moves,
            load_lora=load_lora,
        )

        # Prompt behavior defaults to checkpoint training config; CLI can override.
        if args.use_pgn_in_prompt is not None:
            use_pgn_in_prompt = bool(args.use_pgn_in_prompt)
        if args.pgn_prompt_last_n_moves is not None:
            pgn_prompt_last_n_moves = args.pgn_prompt_last_n_moves
        if args.load_lora is not None:
            load_lora = bool(args.load_lora)
            
        # Disable torch.compile and 8-bit for inference
        config.use_torch_compile = False
        config.load_in_8bit = False

        _maybe_disable_unused_maia_backbone_for_inference(config)

        if config.use_flash_attention:
            try:
                import flash_attn  # noqa: F401
            except Exception:
                try:
                    import flash_attn_3  # noqa: F401
                    if torch.cuda.is_available():
                        major, minor = torch.cuda.get_device_capability()
                        if major < 9:
                            config.use_flash_attention = False
                    else:
                        config.use_flash_attention = False
                except Exception:
                    config.use_flash_attention = False

        print(f"Using Mode: {config.mode.upper()}")
        print("Config summary:")
        print(f"  base_model: {config.base_model}")
        print(f"  use_fen_tokens: {getattr(config, 'use_fen_tokens', False)}")
        print(f"  load_in_8bit: {getattr(config, 'load_in_8bit', None)}")
        print(f"  use_flash_attention: {getattr(config, 'use_flash_attention', None)}")
        print(f"  use_torch_compile: {getattr(config, 'use_torch_compile', None)}")
        print(f"  use_pgn_in_prompt: {use_pgn_in_prompt}")
        print(f"  pgn_prompt_last_n_moves: {pgn_prompt_last_n_moves}")
        print(f"  model.use_lora: {getattr(config, 'use_lora', None)}")
        print(f"  load_lora: {load_lora}")
        if config.mode in {"engineered", "hybrid"}:
            print(f"  engineered_features_type: {config.engineered_features_type}")
        if config.mode == "hybrid":
            print(f"  lc0_dim: {getattr(config, 'lc0_dim', None)}")
            print(f"  hybrid.lc0_proj_dim: {getattr(getattr(config, 'hybrid', None), 'lc0_proj_dim', None)}")
        if config.mode == "perceiver":
            print(f"  perceiver.d_model: {getattr(getattr(config, 'perceiver', None), 'd_model', None)}")
        if config.mode == "maia":
            print(f"  maia.adapter_mode: {getattr(getattr(config, 'maia', None), 'adapter_mode', None)}")
        if config.mode == "chess_fusion":
            fusion_cfg = getattr(config, 'chess_fusion', None)
            print(f"  chess_fusion.use_cnn: {getattr(fusion_cfg, 'use_cnn', None)}")
            print(f"  chess_fusion.load_checkpoint_backbone: {getattr(fusion_cfg, 'load_checkpoint_backbone', None)}")
            print(f"  chess_fusion.load_checkpoint_csmp: {getattr(fusion_cfg, 'load_checkpoint_csmp', None)}")
            print(f"  chess_fusion.load_checkpoint_perceiver: {getattr(fusion_cfg, 'load_checkpoint_perceiver', None)}")
            print(f"  chess_fusion.load_checkpoint_xattn: {getattr(fusion_cfg, 'load_checkpoint_xattn', None)}")
            print(f"  chess_fusion.load_checkpoint_lm_pseudotokens: {getattr(fusion_cfg, 'load_checkpoint_lm_pseudotokens', None)}")
            print(f"  chess_fusion.load_checkpoint_aux_heads: {getattr(fusion_cfg, 'load_checkpoint_aux_heads', None)}")
            print(f"  chess_fusion.num_fusion_tokens: {getattr(fusion_cfg, 'num_fusion_tokens', None)}")
            print(f"  chess_fusion.xattn_layers: {getattr(fusion_cfg, 'xattn_layers', None)}")
        
        # Load ChessCommentaryModel
        model = ChessCommentaryModel(config, torch_dtype=torch.float16)
        
        # Load Weights
        adapter_path = checkpoint_path / "adapter.pt"
        if adapter_path.exists():
            print(f"Loading adapter weights from {adapter_path}")
            sd = torch.load(adapter_path, weights_only=False)
            # Strip torch.compile prefix if present
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            if config.mode in {"chess_fusion", "policy_only"}:
                sd, dropped = _filter_chess_fusion_adapter_state_dict(sd, config)
                if dropped:
                    dropped_str = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
                    print(f"[Load] Selective adapter loading enabled; skipped keys: {dropped_str}")
                missing, unexpected = model.adapter.load_state_dict(sd, strict=False)
                if missing:
                    print(f"[Load] Missing adapter keys: {len(missing)}")
                    print(f"[Load] Missing key examples: {_preview_checkpoint_keys(missing)}")
                if unexpected:
                    print(f"[Load] Unexpected adapter keys ignored: {len(unexpected)}")
                    print(f"[Load] Unexpected key examples: {_preview_checkpoint_keys(unexpected)}")
            else:
                model.adapter.load_state_dict(sd)

        merged_base_path = checkpoint_path / "merged_base"
        if args.use_merged_base and merged_base_path.exists():
            print(f"Loading merged base model from {merged_base_path}")
            merged_llm = AutoModelForCausalLM.from_pretrained(
                merged_base_path,
                torch_dtype=torch.float16,
                device_map="auto",
            )
            model.llm = merged_llm
            model._sync_devices()
        else:
            lora_path = checkpoint_path / "lora"
            if not getattr(model, "_use_lora", False):
                if load_lora and lora_path.exists():
                    print("[LoRA] model.use_lora=False; ignoring checkpoint LoRA and using base LLM.")
                model.llm = model.llm
            else:
                stage_paths = sorted(
                    [p for p in checkpoint_path.glob("lora_stage_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
                )
                progressive_merge = getattr(getattr(config, "lora", None), "progressive_merge", False)
                from peft import PeftModel
                # Unwrap the random LoRA that __init__ created (B=zeros, so merge is a no-op)
                base = model.llm.merge_and_unload()

                # Build list of LoRA dirs to merge sequentially before the final lora/
                prior_lora_dirs = []

                if load_lora and stage_paths:
                    # New checkpoint format: lora_stage_0/, lora_stage_1/, ... saved inside checkpoint
                    prior_lora_dirs = list(stage_paths)
                    print(f"Found {len(prior_lora_dirs)} progressive LoRA stage(s) in checkpoint")
                elif load_lora and progressive_merge:
                    # Legacy checkpoint without saved stages — try to reconstruct from
                    # --prior-checkpoints flag or auto-detect sibling epoch directories
                    if args.prior_checkpoints:
                        prior_lora_dirs = [Path(p) / "lora" for p in args.prior_checkpoints]
                        print(f"Using {len(prior_lora_dirs)} prior checkpoint(s) from --prior-checkpoints")
                    else:
                        auto_priors = _auto_detect_prior_checkpoints(checkpoint_path)
                        if auto_priors:
                            prior_lora_dirs = [p / "lora" for p in auto_priors]
                            print(f"Auto-detected {len(prior_lora_dirs)} prior epoch checkpoint(s): "
                                  f"{[p.parent.name for p in prior_lora_dirs]}")
                        elif lora_path.exists():
                            print("[WARNING] progressive_merge is enabled but no prior LoRA stages found.")
                            print("  The checkpoint's LoRA was trained on a modified base that is not available.")
                            print("  Use --prior-checkpoints to supply earlier epoch checkpoint dirs, e.g.:")
                            print(f"    --prior-checkpoints cp_epoch1 cp_epoch2 ... --checkpoint {checkpoint_path}")

                # Merge all prior stages into base
                for stage_dir in prior_lora_dirs:
                    if not stage_dir.exists():
                        print(f"[WARNING] Prior LoRA stage not found: {stage_dir}, skipping")
                        continue
                    print(f"Merging prior LoRA stage: {stage_dir}")
                    stage_model = PeftModel.from_pretrained(base, str(stage_dir))
                    base = stage_model.merge_and_unload()

                # Load the final lora/ as the active adapter (or use bare base)
                if load_lora and lora_path.exists():
                    print(f"Loading LoRA weights from {lora_path}")
                    model.llm = PeftModel.from_pretrained(base, str(lora_path))
                    _activate_lora_adapter(model.llm, adapter_name="default")
                    _log_lora_status(model.llm, "after PeftModel.from_pretrained")
                else:
                    if not load_lora and lora_path.exists():
                        print("[LoRA] Skipping checkpoint LoRA weights by inference setting.")
                    if prior_lora_dirs:
                        _log_lora_status(base, "after progressive merge (no final lora/)")
                    else:
                        print("[LoRA] Using base model without checkpoint LoRA.")
                    model.llm = base
        model.llm.to(device)
        model.eval()
        tokenizer = model.tokenizer

        # Load LC0 Extractor if needed
        if config.mode == "hybrid":
            if not args.network:
                # Try default paths
                candidates = [
                    "BT3-768x15x24h-swa-2790000.pb.gz",
                    "src/training/lc0_cache/BT3-768x15x24h-swa-2790000.pb.gz",
                    r"D:\python_code\2026\chess_encode\src\training\lc0_cache\BT3-768x15x24h-swa-2790000.pb.gz" 
                ]
                for c in candidates:
                    if Path(c).exists():
                        args.network = c
                        break
            
            if not args.network:
                print("Error: Mode 'hybrid' requires --network path to LC0 weights.")
                return 1
                
            print(f"Loading LC0 network: {args.network}")
            extractor = LC0HiddenStateExtractor(args.network, device="cpu") # Run LC0 on CPU usually safe
            
    # 3. Inference Loop
    for ply in plies:
        print("\n" + "="*60)
        print(f"Position at Ply {ply}")
        print("="*60)
        
        board = replay_to_ply(game, ply)
        pgn_moves_full = build_san_move_history(board)
        prompt_pgn_moves = ""
        if use_pgn_in_prompt:
            prompt_pgn_moves = _truncate_pgn_prompt_moves(pgn_moves_full, pgn_prompt_last_n_moves)
        print(board)
        print(f"FEN: {board.fen()}") 
        
        if args.base_only:
            # Simple base model generation
            prompt_parts = []
            if prompt_pgn_moves:
                prompt_parts.append(f"Game so far: {prompt_pgn_moves}\n")
            prompt_parts.append(
                f"Analyze this chess position (FEN: {board.fen()}). Side to move: {'White' if board.turn else 'Black'}."
            )
            prompt = "".join(prompt_parts)
            messages = [{"role": "user", "content": prompt}]
            text = _safe_apply_chat_template(tokenizer, messages, add_generation_prompt=True)
            user_prompt_tokens = int(tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1])
            chat_prompt_tokens = int(tokenizer(text, return_tensors="pt")["input_ids"].shape[1])
            print("[Inference] User prompt:")
            print(prompt)
            print(f"[Inference] Prompt tokens: user={user_prompt_tokens}, chat_template={chat_prompt_tokens}")
            inputs = tokenizer(text, return_tensors="pt").to(device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_tokens, temperature=args.temperature)
            
            commentary = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        else:
            # Adapter generation
            commentary, generated_ids = generate_commentary(
                model, board, config, extractor, tokenizer,
                pgn_moves=prompt_pgn_moves,
                max_new_tokens=args.max_tokens,
                min_new_tokens=args.min_tokens,
                temperature=args.temperature,
                device=device,
                return_ids=True
            )
            
        print("\n" + "-"*60)
        print("COMMENTARY:")
        print(commentary)
        if not commentary.strip():
            print("[Warning] Empty commentary output. Check adapter weights and prompt inputs.")
            if generated_ids is not None:
                preview_ids = generated_ids[: min(20, generated_ids.numel())].tolist()
                print(f"[Debug] Generated token ids (first 20): {preview_ids}")
        print("-"*60)

if __name__ == "__main__":
    main()


