"""
Standalone structured decode inspector for chess_fusion checkpoints.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import threading
import traceback
from contextlib import nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional

import chess
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.train import (
    ChessCommentaryModel,
    _filter_chess_fusion_adapter_state_dict,
    _preview_checkpoint_keys,
)
from training.chess_fusion_model import FusionDecoderLayer


def _load_sibling_inference_module():
    module_path = Path(__file__).resolve().with_name("inference.py")
    spec = importlib.util.spec_from_file_location("decoding_inspector_inference", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load inference helpers from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


try:
    from .inference import (
        _activate_lora_adapter,
        _auto_detect_prior_checkpoints,
        _load_checkpoint_config,
        _log_lora_status,
        _maybe_disable_unused_maia_backbone_for_inference,
        _set_inference_chess_fusion_load_defaults,
        detect_model_config,
    )
except ImportError:
    _inference_module = _load_sibling_inference_module()
    _activate_lora_adapter = _inference_module._activate_lora_adapter
    _auto_detect_prior_checkpoints = _inference_module._auto_detect_prior_checkpoints
    _load_checkpoint_config = _inference_module._load_checkpoint_config
    _log_lora_status = _inference_module._log_lora_status
    _maybe_disable_unused_maia_backbone_for_inference = (
        _inference_module._maybe_disable_unused_maia_backbone_for_inference
    )
    _set_inference_chess_fusion_load_defaults = (
        _inference_module._set_inference_chess_fusion_load_defaults
    )
    detect_model_config = _inference_module.detect_model_config


DEFAULT_PROMPT = "Provide commentary on this chess position."
DEFAULT_START_FEN = chess.STARTING_FEN


def _resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    if not (checkpoint_path / "adapter.pt").exists() and checkpoint_path.is_dir():
        subdirs = [p for p in checkpoint_path.iterdir() if p.is_dir()]
        if len(subdirs) == 1 and (subdirs[0] / "adapter.pt").exists():
            checkpoint_path = subdirs[0]
    return checkpoint_path


def _ensure_fusion_layers_injected(model: ChessCommentaryModel) -> None:
    if getattr(model.config, "mode", None) != "chess_fusion":
        return

    layers = model.adapter._find_decoder_layers(model.llm)
    if layers is None:
        raise RuntimeError("Could not locate decoder layers on the loaded LLM.")
    if any(isinstance(layer, FusionDecoderLayer) for layer in layers):
        return
    model.adapter.inject_into_llm(model.llm)


def _load_config_for_inspector(checkpoint_path: Path):
    detected_config = detect_model_config(checkpoint_path)
    if detected_config.mode != "chess_fusion":
        raise ValueError(
            "Structured decode inspector only supports chess_fusion checkpoints "
            f"(detected mode={detected_config.mode!r})."
        )

    checkpoint_model_cfg, _max_length, _legacy_cfg, _use_pgn_in_prompt, _pgn_last_n = (
        _load_checkpoint_config(checkpoint_path)
    )
    if checkpoint_model_cfg is None:
        raise ValueError(
            "Checkpoint is missing config.yaml. The structured decode inspector "
            "requires the saved training config to reconstruct the chess_fusion model."
        )

    config = checkpoint_model_cfg
    config.mode = "chess_fusion"
    _set_inference_chess_fusion_load_defaults(config)
    config.use_torch_compile = False
    config.load_in_8bit = False
    _maybe_disable_unused_maia_backbone_for_inference(config)

    fusion_cfg = getattr(config, "chess_fusion", None)
    if fusion_cfg is None:
        raise ValueError("Checkpoint config does not include model.chess_fusion settings.")
    if getattr(fusion_cfg, "xattn_mode", None) != "structured_square_mixer":
        raise ValueError(
            "Structured decode inspector only supports checkpoints with "
            "model.chess_fusion.xattn_mode='structured_square_mixer' "
            f"(got {getattr(fusion_cfg, 'xattn_mode', None)!r})."
        )

    return config


def load_checkpointed_model(
    checkpoint_path: Path,
    *,
    use_merged_base: bool = False,
    load_lora: Optional[bool] = None,
    prior_checkpoints: Optional[list[str]] = None,
) -> tuple[Path, ChessCommentaryModel]:
    checkpoint_path = _resolve_checkpoint_path(checkpoint_path)
    config = _load_config_for_inspector(checkpoint_path)
    load_lora = bool(getattr(config, "use_lora", True)) if load_lora is None else bool(load_lora)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint from {checkpoint_path}")
    model = ChessCommentaryModel(config, torch_dtype=torch.float16)

    adapter_path = checkpoint_path / "adapter.pt"
    print(f"Loading adapter weights from {adapter_path}")
    state_dict = torch.load(adapter_path, weights_only=False)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict, dropped = _filter_chess_fusion_adapter_state_dict(state_dict, config)
    if dropped:
        dropped_str = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
        print(f"[Load] Selective adapter loading enabled; skipped keys: {dropped_str}")
    missing, unexpected = model.adapter.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[Load] Missing adapter keys: {len(missing)}")
        print(f"[Load] Missing key examples: {_preview_checkpoint_keys(missing)}")
    if unexpected:
        print(f"[Load] Unexpected adapter keys ignored: {len(unexpected)}")
        print(f"[Load] Unexpected key examples: {_preview_checkpoint_keys(unexpected)}")

    merged_base_path = checkpoint_path / "merged_base"
    if use_merged_base:
        if not merged_base_path.exists():
            raise FileNotFoundError(
                f"--use-merged-base was set, but {merged_base_path} does not exist."
            )
        print(f"Loading merged base model from {merged_base_path}")
        model.llm = AutoModelForCausalLM.from_pretrained(
            merged_base_path,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        _ensure_fusion_layers_injected(model)
        model._sync_devices()
    else:
        lora_path = checkpoint_path / "lora"
        if not getattr(model, "_use_lora", False):
            if load_lora and lora_path.exists():
                print("[LoRA] model.use_lora=False; ignoring checkpoint LoRA and using base LLM.")
        else:
            from peft import PeftModel

            stage_paths = sorted(
                [p for p in checkpoint_path.glob("lora_stage_*") if p.is_dir()],
                key=lambda p: int(p.name.split("_")[-1]) if p.name.split("_")[-1].isdigit() else -1,
            )
            progressive_merge = getattr(getattr(config, "lora", None), "progressive_merge", False)
            base = model.llm.merge_and_unload()

            prior_lora_dirs: list[Path] = []
            if load_lora and stage_paths:
                prior_lora_dirs = list(stage_paths)
                print(f"Found {len(prior_lora_dirs)} progressive LoRA stage(s) in checkpoint")
            elif load_lora and progressive_merge:
                if prior_checkpoints:
                    prior_lora_dirs = [Path(path) / "lora" for path in prior_checkpoints]
                    print(
                        f"Using {len(prior_lora_dirs)} prior checkpoint(s) from --prior-checkpoints"
                    )
                else:
                    auto_priors = _auto_detect_prior_checkpoints(checkpoint_path)
                    if auto_priors:
                        prior_lora_dirs = [p / "lora" for p in auto_priors]
                        print(
                            "Auto-detected prior epoch checkpoints: "
                            f"{[p.parent.name for p in prior_lora_dirs]}"
                        )
                    elif lora_path.exists():
                        print("[WARNING] progressive_merge is enabled but no prior LoRA stages found.")

            for stage_dir in prior_lora_dirs:
                if not stage_dir.exists():
                    print(f"[WARNING] Prior LoRA stage not found: {stage_dir}, skipping")
                    continue
                print(f"Merging prior LoRA stage: {stage_dir}")
                stage_model = PeftModel.from_pretrained(base, str(stage_dir))
                base = stage_model.merge_and_unload()

            if load_lora and lora_path.exists():
                print(f"Loading LoRA weights from {lora_path}")
                model.llm = PeftModel.from_pretrained(base, str(lora_path))
                _activate_lora_adapter(model.llm, adapter_name="default")
                _log_lora_status(model.llm, "after PeftModel.from_pretrained")
            else:
                if not load_lora and lora_path.exists():
                    print("[LoRA] Skipping checkpoint LoRA weights by request.")
                model.llm = base

            _ensure_fusion_layers_injected(model)
            model._sync_devices()

    model.llm.to(device)
    model.eval()
    model.adapter.set_last_token_trace_capture(True)
    model.adapter.clear_last_token_traces()
    return checkpoint_path, model


def _board_squares(board: chess.Board) -> list[str]:
    squares: list[str] = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        squares.append(piece.symbol() if piece is not None else "")
    return squares


class DecodeSession:
    def __init__(self, model: Any):
        self.model = model
        self.tokenizer = model.tokenizer
        self.available_layers = list(getattr(model.adapter, "xattn_layer_indices", []))
        self.reset()

    def reset(self) -> None:
        self.fen: Optional[str] = None
        self.prompt: str = DEFAULT_PROMPT
        self.board: Optional[chess.Board] = None
        self.generated_token_ids: list[int] = []
        self.past_key_values: Any = None
        self.next_logits: Optional[torch.Tensor] = None
        self.attention_mask: Optional[torch.Tensor] = None
        self.generation_context: Optional[dict[str, Any]] = None
        self.layer_traces: dict[str, Any] = {}
        self.eos_reached = False

    @property
    def has_active_session(self) -> bool:
        return self.fen is not None and self.next_logits is not None

    def _get_device(self) -> torch.device:
        try:
            return next(self.model.llm.get_input_embeddings().parameters()).device
        except Exception:
            return torch.device("cpu")

    def _token_to_string(self, token_id: int) -> str:
        if hasattr(self.tokenizer, "convert_ids_to_tokens"):
            token = self.tokenizer.convert_ids_to_tokens(int(token_id))
            if token is not None:
                return str(token)
        return self.tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

    def _llm_autocast_context(self):
        if not torch.cuda.is_available():
            return nullcontext()

        try:
            embed_param = next(self.model.llm.get_input_embeddings().parameters())
            device = embed_param.device
            amp_dtype = embed_param.dtype
        except Exception:
            device = self._get_device()
            amp_dtype = getattr(self.model, "torch_dtype", None)

        if device.type != "cuda" or amp_dtype not in (torch.float16, torch.bfloat16):
            return nullcontext()

        return torch.amp.autocast("cuda", dtype=amp_dtype)

    def _serialize_traces(self) -> dict[str, Any]:
        if not hasattr(self.model.adapter, "get_last_token_structured_traces"):
            return {}
        raw_traces = self.model.adapter.get_last_token_structured_traces(sample_index=0)
        traces: dict[str, Any] = {}
        for layer_idx, trace in raw_traces.items():
            traces[str(layer_idx)] = {
                "aggregate_64": trace["aggregate_square_weights"].tolist(),
                "csmp_64": trace["csmp_square_weights"].tolist(),
                "perceiver_64": trace["perceiver_square_weights"].tolist(),
                "policy_64": trace["policy_square_weights"].tolist(),
                "global_2": trace["global_weights"].tolist(),
                "raw_slot_192": trace["raw_slot_weights"].tolist(),
                "last_token_index": int(trace["last_token_index"].item()),
            }
        return traces

    def _run_llm_forward(
        self,
        *,
        attention_mask: torch.Tensor,
        text_attention_mask: torch.Tensor,
        **llm_kwargs: Any,
    ):
        self.model.set_generation_context(
            self.generation_context,
            text_attention_mask=text_attention_mask,
        )
        try:
            with torch.inference_mode():
                with self._llm_autocast_context():
                    return self.model.llm(
                        attention_mask=attention_mask,
                        use_cache=True,
                        return_dict=True,
                        **llm_kwargs,
                    )
        finally:
            self.model.clear_generation_context()

    def _update_from_outputs(self, outputs: Any) -> None:
        self.past_key_values = getattr(outputs, "past_key_values", None)
        self.next_logits = outputs.logits[:, -1, :].detach()
        self.layer_traces = self._serialize_traces()

    def _top_tokens(self, top_k: int = 5) -> list[dict[str, Any]]:
        if self.next_logits is None or self.eos_reached:
            return []
        probs = torch.softmax(self.next_logits[0].float(), dim=-1)
        k = min(int(top_k), int(probs.numel()))
        values, indices = probs.topk(k)
        rows: list[dict[str, Any]] = []
        for prob, token_id in zip(values.tolist(), indices.tolist()):
            rows.append(
                {
                    "token_id": int(token_id),
                    "token_text": self._token_to_string(int(token_id)),
                    "prob": float(prob),
                }
            )
        return rows

    def _generated_text(self) -> str:
        if not self.generated_token_ids:
            return ""
        return self.tokenizer.decode(
            self.generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def start(self, fen: str, prompt: Optional[str] = None) -> dict[str, Any]:
        board = chess.Board(fen)
        prompt = (prompt or DEFAULT_PROMPT).strip() or DEFAULT_PROMPT

        self.reset()
        self.board = board
        self.fen = board.fen()
        self.prompt = prompt

        generation_inputs = self.model.prepare_generation_inputs(
            lc0_hidden_states=None,
            side_to_move=board.turn,
            prompt=self.prompt,
            fen=self.fen,
            maia_policy=None,
        )
        self.generation_context = generation_inputs["generation_context"]
        self.attention_mask = generation_inputs["combined_mask"].detach().clone()
        self.model.adapter.clear_last_token_traces()
        outputs = self._run_llm_forward(
            inputs_embeds=generation_inputs["combined_embeds"],
            attention_mask=self.attention_mask,
            position_ids=generation_inputs["position_ids"],
            text_attention_mask=self.attention_mask,
        )
        self._update_from_outputs(outputs)
        return self.snapshot()

    def step(self, token_id: Optional[int] = None) -> dict[str, Any]:
        if not self.has_active_session:
            raise RuntimeError("No active decode session. Start one first.")
        if self.eos_reached:
            return self.snapshot()
        if self.attention_mask is None:
            raise RuntimeError("Decode session is missing attention state.")

        if token_id is None:
            next_token_id = int(torch.argmax(self.next_logits[0]).item())
        else:
            next_token_id = int(token_id)
            vocab_size = int(self.next_logits.size(-1))
            if next_token_id < 0 or next_token_id >= vocab_size:
                raise ValueError(f"token_id {next_token_id} is outside the vocabulary range 0..{vocab_size - 1}.")

        device = self._get_device()
        next_input_ids = torch.tensor([[next_token_id]], dtype=torch.long, device=device)
        next_mask_piece = torch.ones((1, 1), dtype=self.attention_mask.dtype, device=self.attention_mask.device)
        next_attention_mask = torch.cat([self.attention_mask, next_mask_piece], dim=1)
        position_ids = self.model._build_position_ids(
            next_attention_mask,
            prefix_len=self.model.num_prefix_tokens,
        )[:, -1:]

        self.model.adapter.clear_last_token_traces()
        outputs = self._run_llm_forward(
            input_ids=next_input_ids,
            attention_mask=next_attention_mask,
            position_ids=position_ids,
            past_key_values=self.past_key_values,
            text_attention_mask=next_mask_piece,
        )

        self.generated_token_ids.append(next_token_id)
        self.attention_mask = next_attention_mask
        self.eos_reached = (
            self.tokenizer.eos_token_id is not None
            and next_token_id == int(self.tokenizer.eos_token_id)
        )
        self._update_from_outputs(outputs)
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        return {
            "fen": self.fen,
            "prompt": self.prompt,
            "generated_text": self._generated_text(),
            "emitted_token_ids": list(self.generated_token_ids),
            "emitted_tokens": [self._token_to_string(token_id) for token_id in self.generated_token_ids],
            "top_tokens": self._top_tokens(top_k=5),
            "available_layers": list(self.available_layers),
            "layer_traces": self.layer_traces,
            "board_squares": _board_squares(self.board) if self.board is not None else [],
            "side_to_move": "w" if self.board is not None and self.board.turn else "b",
            "eos_reached": bool(self.eos_reached),
        }


class InspectorAppState:
    def __init__(self, checkpoint_path: Path, model: ChessCommentaryModel):
        self.checkpoint_path = str(checkpoint_path)
        self.model = model
        self.session = DecodeSession(model)
        self.lock = threading.Lock()

    def get_session_snapshot(self) -> dict[str, Any]:
        with self.lock:
            if not self.session.has_active_session:
                raise RuntimeError("No active session.")
            return self.session.snapshot()

    def start_session(self, fen: str, prompt: Optional[str]) -> dict[str, Any]:
        with self.lock:
            return self.session.start(fen=fen, prompt=prompt)

    def step_session(self, token_id: Optional[int]) -> dict[str, Any]:
        with self.lock:
            return self.session.step(token_id=token_id)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Structured Decode Inspector</title>
  <style>
    :root {
      --bg: #f4efe5;
      --surface: #fbf7ef;
      --surface-2: #efe4d0;
      --border: #a98f68;
      --text: #26190f;
      --muted: #6d5a42;
      --accent: #8c3b2f;
      --accent-2: #2f5d50;
      --light-square: #f6e7ce;
      --dark-square: #b78a58;
      --cell-text: #1d130b;
      --danger: #9f2c2c;
      --radius: 14px;
      --shadow: 0 12px 30px rgba(38, 25, 15, 0.12);
      --font: "Segoe UI", "Trebuchet MS", system-ui, sans-serif;
      --mono: "Consolas", "Courier New", monospace;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: var(--font);
      color: var(--text);
      background:
        radial-gradient(circle at top left, rgba(140, 59, 47, 0.08), transparent 28%),
        radial-gradient(circle at top right, rgba(47, 93, 80, 0.08), transparent 24%),
        linear-gradient(180deg, #efe5d3 0%, #f8f3ea 32%, #f3ebde 100%);
      min-height: 100vh;
    }

    .page {
      max-width: 1500px;
      margin: 0 auto;
      padding: 24px;
    }

    .hero {
      display: grid;
      gap: 18px;
      padding: 22px 24px;
      border: 1px solid var(--border);
      border-radius: calc(var(--radius) + 4px);
      background:
        linear-gradient(135deg, rgba(251, 247, 239, 0.97), rgba(233, 219, 191, 0.97)),
        linear-gradient(135deg, rgba(140, 59, 47, 0.08), transparent 55%);
      box-shadow: var(--shadow);
    }

    .hero h1 {
      margin: 0;
      font-size: clamp(1.6rem, 2.4vw, 2.2rem);
      letter-spacing: 0.02em;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 900px;
    }

    .controls {
      display: grid;
      grid-template-columns: 1.4fr 1fr auto auto;
      gap: 14px;
      align-items: end;
      margin-top: 16px;
    }

    .control {
      display: grid;
      gap: 8px;
    }

    label {
      font-size: 0.78rem;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }

    input, textarea, select, button {
      font: inherit;
    }

    input, textarea, select {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: rgba(255, 252, 246, 0.95);
      color: var(--text);
      padding: 12px 14px;
      box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
    }

    textarea {
      min-height: 78px;
      resize: vertical;
    }

    button {
      border: none;
      border-radius: 999px;
      padding: 12px 18px;
      font-weight: 700;
      cursor: pointer;
      transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.12s ease;
    }

    button:hover { transform: translateY(-1px); box-shadow: 0 8px 16px rgba(38, 25, 15, 0.14); }
    button:disabled { opacity: 0.55; cursor: not-allowed; transform: none; box-shadow: none; }

    .primary {
      background: linear-gradient(135deg, #a6493a, #8c3b2f);
      color: #fff7f0;
    }

    .secondary {
      background: linear-gradient(135deg, #496e61, #2f5d50);
      color: #f2fbf8;
    }

    .status-bar {
      margin-top: 12px;
      min-height: 24px;
      color: var(--muted);
      font-size: 0.95rem;
    }

    .status-bar.error {
      color: var(--danger);
      font-weight: 700;
    }

    .layout {
      display: grid;
      grid-template-columns: 360px 1fr;
      gap: 20px;
      margin-top: 20px;
    }

    .panel {
      border: 1px solid var(--border);
      border-radius: var(--radius);
      background: rgba(251, 247, 239, 0.94);
      box-shadow: var(--shadow);
      overflow: hidden;
    }

    .panel-head {
      padding: 14px 18px;
      border-bottom: 1px solid rgba(169, 143, 104, 0.45);
      background: linear-gradient(135deg, rgba(239, 228, 208, 0.88), rgba(233, 219, 191, 0.88));
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }

    .panel-head h2 {
      margin: 0;
      font-size: 0.92rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }

    .panel-body {
      padding: 16px 18px 18px;
    }

    .token-list {
      display: grid;
      gap: 10px;
    }

    .token-button {
      width: 100%;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 12px;
      align-items: center;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(255, 252, 246, 0.96);
      border: 1px solid rgba(169, 143, 104, 0.65);
      color: var(--text);
      text-align: left;
    }

    .token-id {
      color: var(--muted);
      font-size: 0.78rem;
      font-family: var(--mono);
    }

    .token-prob {
      font-family: var(--mono);
      font-weight: 700;
    }

    .generated {
      min-height: 160px;
      max-height: 360px;
      overflow: auto;
      padding: 14px;
      border-radius: 12px;
      background: rgba(255, 252, 246, 0.96);
      border: 1px solid rgba(169, 143, 104, 0.55);
      font-family: var(--mono);
      line-height: 1.55;
      white-space: pre-wrap;
    }

    .board-grid {
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      border: 1px solid rgba(169, 143, 104, 0.8);
      border-radius: 10px;
      overflow: hidden;
      background: var(--surface-2);
    }

    .square {
      aspect-ratio: 1 / 1;
      position: relative;
      padding: 4px;
      display: flex;
      align-items: flex-end;
      justify-content: flex-end;
      overflow: hidden;
    }

    .square::before {
      content: attr(data-square);
      position: absolute;
      top: 4px;
      left: 4px;
      font-size: 0.62rem;
      color: rgba(29, 19, 11, 0.56);
      font-family: var(--mono);
    }

    .piece {
      position: absolute;
      top: 20px;
      left: 6px;
      font-size: 1rem;
      font-weight: 800;
      color: var(--cell-text);
    }

    .piece.black {
      color: rgba(255, 250, 243, 0.92);
      text-shadow: 0 1px 1px rgba(29, 19, 11, 0.45);
    }

    .weight {
      position: relative;
      z-index: 1;
      font-size: 0.72rem;
      font-family: var(--mono);
      font-weight: 700;
      color: var(--cell-text);
      background: rgba(255, 251, 244, 0.78);
      padding: 1px 4px;
      border-radius: 999px;
    }

    .boards {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 18px;
      margin-top: 16px;
    }

    .board-card {
      display: grid;
      gap: 10px;
    }

    .board-meta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 0.86rem;
      font-family: var(--mono);
    }

    .trace-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 10px 18px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.88rem;
      font-family: var(--mono);
    }

    .trace-meta strong {
      color: var(--text);
    }

    .empty {
      padding: 28px 14px;
      color: var(--muted);
      text-align: center;
      border: 1px dashed rgba(169, 143, 104, 0.7);
      border-radius: 12px;
      background: rgba(255, 252, 246, 0.78);
    }

    @media (max-width: 1120px) {
      .controls { grid-template-columns: 1fr; }
      .layout { grid-template-columns: 1fr; }
    }

    @media (max-width: 760px) {
      .boards { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <h1>Structured Decode Inspector</h1>
        <p>Step through commentary generation one token at a time, inspect the live top-5 next-token distribution, and compare aggregate, CSMP, Perceiver, and Policy square weights for the currently selected x-attn layer.</p>
      </div>
      <div class="controls">
        <div class="control">
          <label for="fen-input">FEN</label>
          <input id="fen-input" value="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1">
        </div>
        <div class="control">
          <label for="prompt-input">Prompt</label>
          <textarea id="prompt-input">Provide commentary on this chess position.</textarea>
        </div>
        <button class="primary" id="restart-btn">Restart</button>
        <button class="secondary" id="step-btn">Step (Greedy)</button>
      </div>
      <div class="status-bar" id="status-bar"></div>
    </section>

    <section class="layout">
      <div style="display:grid; gap:20px;">
        <div class="panel">
          <div class="panel-head"><h2>Next Tokens</h2></div>
          <div class="panel-body"><div class="token-list" id="token-list"></div></div>
        </div>

        <div class="panel">
          <div class="panel-head"><h2>Generated Text</h2></div>
          <div class="panel-body"><div class="generated" id="generated-text"></div></div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-head">
          <h2>Layer Trace</h2>
          <select id="layer-select"></select>
        </div>
        <div class="panel-body">
          <div class="trace-meta" id="trace-meta"></div>
          <div class="boards" id="boards"></div>
        </div>
      </div>
    </section>
  </div>

  <script>
    const DEFAULT_PROMPT = "Provide commentary on this chess position.";
    const boardLabels = [
      ["Aggregate", "aggregate_64"],
      ["CSMP", "csmp_64"],
      ["Perceiver", "perceiver_64"],
      ["Policy", "policy_64"],
    ];

    let session = null;
    let selectedLayer = null;

    function setStatus(message, isError = false) {
      const el = document.getElementById("status-bar");
      el.textContent = message || "";
      el.className = "status-bar" + (isError ? " error" : "");
    }

    function formatTokenText(tokenText) {
      if (tokenText == null) return "<null>";
      let text = String(tokenText).replace(/\\n/g, "\\\\n").replace(/\\t/g, "[tab]");
      if (text.trim().length === 0) {
        text = text.replace(/ /g, "[sp]");
      }
      return text || "<empty>";
    }

    function squareName(index) {
      const file = "abcdefgh"[index % 8];
      const rank = Math.floor(index / 8) + 1;
      return `${file}${rank}`;
    }

    function clamp(value, minValue, maxValue) {
      return Math.min(maxValue, Math.max(minValue, value));
    }

    function heatColor(isDark, intensity) {
      const base = isDark ? [183, 138, 88] : [246, 231, 206];
      const alpha = 0.12 + 0.76 * clamp(intensity, 0, 1);
      return `linear-gradient(0deg, rgba(166, 73, 58, ${alpha}), rgba(166, 73, 58, ${alpha})), rgb(${base[0]}, ${base[1]}, ${base[2]})`;
    }

    function renderBoardCard(title, key, trace, boardSquares) {
      const weights = Array.isArray(trace?.[key]) ? trace[key] : new Array(64).fill(0);
      const maxWeight = Math.max(...weights, 0);
      const totalWeight = weights.reduce((sum, value) => sum + value, 0);
      let cells = "";

      for (let rank = 7; rank >= 0; rank -= 1) {
        for (let file = 0; file < 8; file += 1) {
          const index = rank * 8 + file;
          const square = squareName(index);
          const piece = boardSquares[index] || "";
          const value = Number(weights[index] || 0);
          const dark = (rank + file) % 2 === 1;
          const intensity = maxWeight > 0 ? value / maxWeight : 0;
          const pieceClass = piece && piece === piece.toLowerCase() ? "piece black" : "piece";
          const titleText = `${square} | weight=${value.toFixed(6)}`;
          cells += `
            <div class="square" data-square="${square}" title="${titleText}" style="background:${heatColor(dark, intensity)}">
              ${piece ? `<div class="${pieceClass}">${piece}</div>` : ""}
              <div class="weight">${value.toFixed(3)}</div>
            </div>
          `;
        }
      }

      return `
        <div class="board-card">
          <div class="board-meta">
            <strong>${title}</strong>
            <span>${(totalWeight * 100).toFixed(1)}%</span>
          </div>
          <div class="board-grid">${cells}</div>
        </div>
      `;
    }

    function syncLayerOptions(availableLayers) {
      const select = document.getElementById("layer-select");
      const layerStrings = availableLayers.map((value) => String(value));
      if (!layerStrings.length) {
        select.innerHTML = "";
        selectedLayer = null;
        return;
      }

      if (!selectedLayer || !layerStrings.includes(String(selectedLayer))) {
        selectedLayer = layerStrings[0];
      }

      select.innerHTML = layerStrings.map((layer) => {
        const selected = String(layer) === String(selectedLayer) ? "selected" : "";
        return `<option value="${layer}" ${selected}>Layer ${layer}</option>`;
      }).join("");
    }

    function renderTracePanel() {
      const boardsEl = document.getElementById("boards");
      const metaEl = document.getElementById("trace-meta");
      if (!session || !selectedLayer) {
        boardsEl.innerHTML = `<div class="empty">Start a session to inspect structured square weights.</div>`;
        metaEl.innerHTML = "";
        return;
      }

      const trace = session.layer_traces?.[String(selectedLayer)];
      if (!trace) {
        boardsEl.innerHTML = `<div class="empty">No trace is available for layer ${selectedLayer} yet.</div>`;
        metaEl.innerHTML = "";
        return;
      }

      const sourceMass = [
        (trace.csmp_64 || []).reduce((sum, value) => sum + value, 0),
        (trace.perceiver_64 || []).reduce((sum, value) => sum + value, 0),
        (trace.policy_64 || []).reduce((sum, value) => sum + value, 0),
      ];
      const global = trace.global_2 || [0, 0];

      metaEl.innerHTML = `
        <span><strong>CSMP</strong> ${(sourceMass[0] * 100).toFixed(1)}%</span>
        <span><strong>Perceiver</strong> ${(sourceMass[1] * 100).toFixed(1)}%</span>
        <span><strong>Policy</strong> ${(sourceMass[2] * 100).toFixed(1)}%</span>
        <span><strong>Global P</strong> ${(Number(global[0] || 0) * 100).toFixed(1)}%</span>
        <span><strong>Side</strong> ${(Number(global[1] || 0) * 100).toFixed(1)}%</span>
      `;

      const boardSquares = session.board_squares || new Array(64).fill("");
      boardsEl.innerHTML = boardLabels.map(([title, key]) => renderBoardCard(title, key, trace, boardSquares)).join("");
    }

    function renderTopTokens() {
      const tokenList = document.getElementById("token-list");
      const rows = session?.top_tokens || [];
      if (!rows.length) {
        tokenList.innerHTML = `<div class="empty">${session?.eos_reached ? "EOS reached. Restart to decode again." : "No token distribution available yet."}</div>`;
        return;
      }

      tokenList.innerHTML = rows.map((row, index) => `
        <button class="token-button" data-token-id="${row.token_id}">
          <div>
            <div><strong>${index + 1}.</strong> ${formatTokenText(row.token_text)}</div>
            <div class="token-id">token_id=${row.token_id}</div>
          </div>
          <div class="token-prob">${(row.prob * 100).toFixed(2)}%</div>
        </button>
      `).join("");

      tokenList.querySelectorAll(".token-button").forEach((button) => {
        button.addEventListener("click", async () => {
          const tokenId = Number(button.dataset.tokenId);
          await stepSession(tokenId);
        });
      });
    }

    function renderGeneratedText() {
      const generated = document.getElementById("generated-text");
      if (!session) {
        generated.textContent = "";
        return;
      }
      const pieces = [];
      pieces.push(`FEN: ${session.fen || ""}`);
      pieces.push(`Prompt: ${session.prompt || DEFAULT_PROMPT}`);
      pieces.push("");
      pieces.push(session.generated_text || "");
      generated.textContent = pieces.join("\\n");
    }

    function renderSession() {
      renderTopTokens();
      renderGeneratedText();
      syncLayerOptions(session?.available_layers || []);
      renderTracePanel();
      document.getElementById("step-btn").disabled = !session || !!session.eos_reached;
    }

    async function fetchJson(path, options = {}) {
      const response = await fetch(path, options);
      let payload = {};
      try {
        payload = await response.json();
      } catch (_error) {
        payload = {};
      }
      if (!response.ok) {
        throw new Error(payload.error || `Request failed with status ${response.status}`);
      }
      return payload;
    }

    async function restartSession() {
      const fen = document.getElementById("fen-input").value.trim();
      const prompt = document.getElementById("prompt-input").value.trim() || DEFAULT_PROMPT;
      setStatus("Priming prompt and capturing the first next-token distribution...");
      try {
        session = await fetchJson("/api/session", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ fen, prompt }),
        });
        setStatus("Session ready.");
        renderSession();
      } catch (error) {
        setStatus(error.message || String(error), true);
      }
    }

    async function stepSession(tokenId = null) {
      setStatus(tokenId == null ? "Advancing with greedy argmax..." : `Advancing with forced token ${tokenId}...`);
      try {
        const body = tokenId == null ? {} : { token_id: tokenId };
        session = await fetchJson("/api/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        setStatus(session.eos_reached ? "EOS reached." : "Step complete.");
        renderSession();
      } catch (error) {
        setStatus(error.message || String(error), true);
      }
    }

    document.getElementById("restart-btn").addEventListener("click", restartSession);
    document.getElementById("step-btn").addEventListener("click", async () => {
      await stepSession(null);
    });
    document.getElementById("layer-select").addEventListener("change", (event) => {
      selectedLayer = event.target.value;
      renderTracePanel();
    });

    restartSession();
  </script>
</body>
</html>
"""


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    content_length = int(handler.headers.get("Content-Length", "0"))
    if content_length <= 0:
        return {}
    raw_body = handler.rfile.read(content_length)
    if not raw_body:
        return {}
    return json.loads(raw_body.decode("utf-8"))


def _write_json(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _make_handler(app_state: InspectorAppState):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML_PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/api/session":
                try:
                    snapshot = app_state.get_session_snapshot()
                    _write_json(self, 200, snapshot)
                except Exception as exc:
                    print(
                        f"[Inspector] GET {self.path} failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
                    _write_json(self, 404, {"error": f"{type(exc).__name__}: {exc}"})
                return

            _write_json(self, 404, {"error": f"Unknown endpoint: {self.path}"})

        def do_POST(self) -> None:
            try:
                body = _read_json_body(self)
            except json.JSONDecodeError as exc:
                _write_json(self, 400, {"error": f"Invalid JSON body: {exc}"})
                return

            try:
                if self.path == "/api/session":
                    fen = str(body.get("fen") or DEFAULT_START_FEN).strip()
                    prompt = body.get("prompt")
                    snapshot = app_state.start_session(fen=fen, prompt=prompt)
                    _write_json(self, 200, snapshot)
                    return

                if self.path == "/api/step":
                    token_id = body.get("token_id", None)
                    snapshot = app_state.step_session(token_id=token_id)
                    _write_json(self, 200, snapshot)
                    return

                _write_json(self, 404, {"error": f"Unknown endpoint: {self.path}"})
            except Exception as exc:
                print(
                    f"[Inspector] POST {self.path} failed: {type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                traceback.print_exc()
                _write_json(self, 400, {"error": f"{type(exc).__name__}: {exc}"})

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description="Standalone structured decode inspector")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved chess_fusion checkpoint")
    parser.add_argument("--port", type=int, default=8765, help="Local port for the inspector web app")
    parser.add_argument(
        "--load-lora",
        dest="load_lora",
        action="store_true",
        help="Load checkpoint LoRA weights when available",
    )
    parser.add_argument(
        "--no-load-lora",
        dest="load_lora",
        action="store_false",
        help="Ignore checkpoint LoRA weights and use the merged base model state only",
    )
    parser.set_defaults(load_lora=None)
    parser.add_argument(
        "--use-merged-base",
        action="store_true",
        help="Load merged_base from the checkpoint directory instead of base+LoRA replay",
    )
    parser.add_argument(
        "--prior-checkpoints",
        nargs="+",
        default=None,
        help="Ordered list of prior progressive-merge checkpoint directories",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    try:
        resolved_checkpoint_path, model = load_checkpointed_model(
            checkpoint_path,
            use_merged_base=bool(args.use_merged_base),
            load_lora=args.load_lora,
            prior_checkpoints=args.prior_checkpoints,
        )
    except Exception as exc:
        print(f"Failed to load checkpoint: {exc}")
        traceback.print_exc()
        return 1

    app_state = InspectorAppState(resolved_checkpoint_path, model)
    server = ThreadingHTTPServer(("127.0.0.1", int(args.port)), _make_handler(app_state))
    print(f"Structured decode inspector ready at http://127.0.0.1:{args.port}")
    print(f"Checkpoint: {resolved_checkpoint_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
