"""
Chess Commentary Training Script with LoRA

Trains a HuggingFace causal LM with LoRA fine-tuning to generate chess commentary
using chess position context from various adapters.

Supported base models:
- TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default, LLaMA architecture)
- Waterhorse/chessgpt-chat-v1 (GPT-NeoX architecture, 2.8B)
- Any HuggingFace AutoModelForCausalLM-compatible model

Architecture:
- ChessPositionAdapter: chess features -> prefix embeddings
- HuggingFace LLM with LoRA (auto-detected target modules per architecture)
- Quantization support (4-bit/8-bit) for memory efficiency
"""

from __future__ import annotations

import os
import sys
import re
from pathlib import Path
from typing import Any, Dict, Optional
from collections import Counter
from contextlib import contextmanager, nullcontext
import torch
import chess
import shutil
import datetime
import yaml
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm

# Optional wandb import for logging
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training
)

from training.chess_adapter import (
    ChessPositionAdapter,
    ENGINEERED_FEATURE_DIM,
    EngineeredPositionAdapter,
    HybridPositionAdapter,
    extract_engineered_features,
)
try:
    from training.perceiver_adapter import PerceiverChessAdapter, extract_perceiver_features
except ImportError:
    PerceiverChessAdapter = None
    extract_perceiver_features = None
from training.maia_model import (
    MaiaLLMAdapter,
    extract_maia_features,
    get_maia_mapping,
    unmirror_policy_move,
)
from training.chess_fusion_model import ChessFusionAdapter
from training.config import TrainingConfig, ModelConfig, load_config
from training.live_control import TrainingController
from training.sample_contract import load_training_sample
from training.chess_token_weights import build_chess_token_loss_weights


# Default configuration
DEFAULT_BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_PROJECTION_DIM = 128
DEFAULT_LC0_DIM = 768
DEFAULT_NUM_LAYERS = 4


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            torch.cuda.nvtx.range_pop()
        return
    yield


def _filter_chess_fusion_adapter_state_dict(state_dict: dict, model_config: ModelConfig):
    """Filter chess_fusion adapter checkpoint keys based on config flags.

    NOTE: Maia backbone loading is controlled by
    `model.chess_fusion.load_checkpoint_backbone`.
    """
    if model_config.mode not in ("chess_fusion", "policy_only"):
        return state_dict, {}

    fusion_cfg = getattr(model_config, "chess_fusion", None)
    if fusion_cfg is None:
        return state_dict, {}

    pseudotokens_enabled = bool(getattr(fusion_cfg, "enable_lm_pseudotokens", True))

    load_flags = {
        "backbone": getattr(fusion_cfg, "load_checkpoint_backbone", True),
        # Objective teacher must stay pretrained/frozen; never resume from adapter checkpoints.
        "teacher_backbone": False,
        "csmp": getattr(fusion_cfg, "load_checkpoint_csmp", True),
        "perceiver": getattr(fusion_cfg, "load_checkpoint_perceiver", True),
        "xattn": getattr(fusion_cfg, "load_checkpoint_xattn", True),
        "prepend_latents": getattr(
            fusion_cfg,
            "load_checkpoint_prepend_latents",
            getattr(fusion_cfg, "load_checkpoint_xattn", True),
        ),
        "lm_pseudotokens": pseudotokens_enabled and getattr(fusion_cfg, "load_checkpoint_lm_pseudotokens", True),
        "aux_heads": getattr(fusion_cfg, "load_checkpoint_aux_heads", True),
    }

    module_prefixes = {
        "backbone": ("backbone.",),
        "teacher_backbone": ("teacher_backbone.",),
        "csmp": ("multi_scale.chess_mp.",),
        "perceiver": (
            "perceiver.",
            "multi_scale.side_token.",
            "multi_scale.rank_embedding.",
            "multi_scale.file_embedding.",
            "multi_scale.proj_cnn.",
            "multi_scale.proj_cnn_concat.",
            "multi_scale.proj_mid.",
            "multi_scale.proj_final.",
        ),
        "xattn": ("gated_xattns.", "shared_readout.", "shared_recurrent_query_gru."),
        "prepend_latents": ("prepend_latent_readout.",),
        "lm_pseudotokens": ("lm_pseudotoken_layers.",),
        "aux_heads": ("bsr_head.", "spp_head."),
    }

    filtered = {}
    dropped_counts = {k: 0 for k in module_prefixes}
    raw_xattn_mode = getattr(fusion_cfg, "xattn_mode", "cross_attn")

    for key, value in state_dict.items():
        if key.startswith("gated_xattns."):
            parts = key.split(".")
            if len(parts) >= 3:
                leaf_name = parts[2]
                if leaf_name in {
                    "q_norm",
                    "k_proj",
                    "v_proj",
                    "recurrent_query_proj",
                    "recurrent_query_gru",
                    "recurrent_query_norm",
                    "structured_router_stem",
                    "structured_square_weight_proj",
                    "structured_global_weight_proj",
                }:
                    dropped_counts["xattn"] += 1
                    continue

        matched_module = None
        for module_name, prefixes in module_prefixes.items():
            if any(key.startswith(prefix) for prefix in prefixes):
                matched_module = module_name
                break

        if matched_module is None:
            filtered[key] = value
            continue

        if load_flags[matched_module]:
            filtered[key] = value
        else:
            dropped_counts[matched_module] += 1

    dropped_counts = {k: v for k, v in dropped_counts.items() if v > 0}
    return filtered, dropped_counts


def _prepare_adapter_state_dict_for_save(adapter: nn.Module) -> tuple[dict, int]:
    """Strip non-checkpointable keys from adapter state dict before saving."""
    state = adapter.state_dict()
    filtered = {}
    dropped = 0
    for key, value in state.items():
        if key.startswith("teacher_backbone."):
            dropped += 1
            continue
        filtered[key] = value
    return filtered, dropped


def _cp_to_eval_bucket(cp: float, num_buckets: int) -> int:
    """Map centipawn eval to a class index for the eval head."""
    n = int(max(2, num_buckets))

    # Default 5-way bins: losing / slight loss / equal / slight win / winning
    if n == 5:
        if cp <= -300:
            return 0
        if cp <= -100:
            return 1
        if cp < 100:
            return 2
        if cp < 300:
            return 3
        return 4

    # Generic fallback: symmetric bins over [-800, 800], clamped to endpoints.
    lo = -800.0
    hi = 800.0
    width = (hi - lo) / n
    idx = int((float(cp) - lo) / width)
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx


def _encode_move_evals_to_policy_vocab(
    move_evals: dict,
    side_to_move: bool,
):
    """Encode absolute UCI move_evals into Maia policy vocab indices."""
    mapping = get_maia_mapping()
    indices = []
    cp_values = []
    for uci, cp in move_evals.items():
        try:
            uci_abs = str(uci)
            # Policy vocab is perspective-relative for Black-to-move positions.
            uci_rel = unmirror_policy_move(uci_abs, side_to_move)
            idx = mapping.encode(uci_rel)
            if idx >= 0:
                indices.append(idx)
                cp_values.append(float(cp))
        except Exception:
            continue
    return indices, cp_values


def _encode_named_moves_to_policy_vocab(
    move_list,
    move_evals: dict,
    side_to_move: bool,
):
    """Encode an explicit UCI move list (e.g., Stockfish best moves) into policy vocab."""
    if not move_list or not isinstance(move_evals, dict):
        return [], []
    mapping = get_maia_mapping()
    seen = set()
    indices = []
    cp_values = []
    for uci in move_list:
        try:
            uci_abs = str(uci)
            if uci_abs in seen:
                continue
            seen.add(uci_abs)
            if uci_abs not in move_evals:
                continue
            uci_rel = unmirror_policy_move(uci_abs, side_to_move)
            idx = mapping.encode(uci_rel)
            if idx >= 0:
                indices.append(idx)
                cp_values.append(float(move_evals[uci_abs]))
        except Exception:
            continue
    return indices, cp_values


def _preview_checkpoint_keys(keys, max_items: int = 12) -> str:
    """Return a short, human-readable preview of checkpoint key names."""
    if not keys:
        return ""
    shown = list(keys[:max_items])
    preview = ", ".join(shown)
    remaining = len(keys) - len(shown)
    if remaining > 0:
        preview = f"{preview}, ... (+{remaining} more)"
    return preview


def _get_decoder_layers(model: nn.Module):
    """Locate the decoder layer list for different HuggingFace model architectures.
    
    Supports LLaMA/Mistral/Qwen (.model.layers), GPT-NeoX (.gpt_neox.layers),
    GPT-2 (.transformer.h), Phi (.model.layers), and PEFT-wrapped variants.
    
    Returns:
        nn.ModuleList of decoder layers, or None if not found.
    """
    # Unwrap PEFT to get the underlying HF model
    base = model
    if hasattr(model, 'base_model'):
        base = model.base_model
        if hasattr(base, 'model'):
            base = base.model

    # Try common decoder layer paths
    # LLaMA / Mistral / Qwen / Phi: model.model.layers
    if hasattr(base, 'model') and hasattr(base.model, 'layers'):
        return base.model.layers
    # GPT-NeoX (ChessGPT): model.gpt_neox.layers
    if hasattr(base, 'gpt_neox') and hasattr(base.gpt_neox, 'layers'):
        return base.gpt_neox.layers
    # GPT-2: model.transformer.h
    if hasattr(base, 'transformer') and hasattr(base.transformer, 'h'):
        return base.transformer.h
    # Fallback: direct .layers attribute
    if hasattr(base, 'layers'):
        return base.layers
    return None


# Map of model_type -> LoRA target module names for common architectures
_LORA_TARGET_MODULES_MAP = {
    # LLaMA / Mistral / Qwen / TinyLlama
    "llama": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen2": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    # GPT-NeoX (Waterhorse/chessgpt-chat-v1)
    "gpt_neox": ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    # GPT-2
    "gpt2": ["c_attn", "c_proj", "c_fc"],
    # Phi / Phi-2
    "phi": ["q_proj", "k_proj", "v_proj", "dense", "fc1", "fc2"],
    "phi3": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    # Gemma
    "gemma": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "gemma2": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def _auto_lora_target_modules(model: nn.Module) -> list[str]:
    """Auto-detect LoRA target modules based on the HuggingFace model architecture.
    
    Inspects model.config.model_type to select appropriate linear layer names.
    Falls back to PEFT's default 'all-linear' if the architecture is unknown.
    """
    config = getattr(model, 'config', None)
    model_type = getattr(config, 'model_type', None)
    
    if model_type and model_type in _LORA_TARGET_MODULES_MAP:
        targets = _LORA_TARGET_MODULES_MAP[model_type]
        print(f"Auto-detected LoRA targets for {model_type}: {targets}")
        return targets
    
    # Fallback: let PEFT find all linear layers
    print(f"Unknown model_type '{model_type}' — using PEFT 'all-linear' target module detection")
    return "all-linear"


def _safe_apply_chat_template(tokenizer, messages: list[dict], **kwargs) -> str:
    """Apply chat template with fallback for models that don't have one.
    
    Some models (e.g. GPT-NeoX based ChessGPT) lack a chat_template.
    Falls back to a simple concatenation of message contents.
    """
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)
    except (AttributeError, Exception):
        # Simple fallback: join message contents
        parts = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role:
                parts.append(f"{role}: {content}")
            else:
                parts.append(content)
        result = "\n".join(parts)
        if kwargs.get("add_generation_prompt"):
            result += "\nassistant: "
        return result


def build_commentary_prompt(
    fen: Optional[str] = None,
    pgn_moves: str = "",
    last_move: Optional[str] = None,
    use_pgn_in_prompt: bool = False,
    use_last_move_in_prompt: bool = False,
    prepend_fen_in_prompt: bool = False,
) -> str:
    """Build a user prompt for commentary generation/training."""
    parts: list[str] = []
    if prepend_fen_in_prompt and fen:
        parts.append(f"FEN: {fen}\n")
    if use_pgn_in_prompt and pgn_moves:
        parts.append(f"Game so far: {pgn_moves}\n")
    if use_last_move_in_prompt and last_move:
        parts.append(f"After the move {last_move}, provide commentary on this chess position.")
    else:
        parts.append("Provide commentary on this chess position.")
    return "".join(parts)


class ChessCommentaryModel(nn.Module):
    """
    Combines ChessPositionAdapter with a HuggingFace causal LM + LoRA.
    
    Architecture:
    1. ChessPositionAdapter projects chess features to prefix embeddings
    2. Prefix embeddings are prepended to text token embeddings
    3. Combined sequence is processed by the base LLM with LoRA
    """
    
    
    def __init__(self, config: ModelConfig, torch_dtype: torch.dtype = torch.float16, device_map="auto"):
        super().__init__()
        
        self.config = config
        self.base_model_name = config.base_model
        self.torch_dtype = torch_dtype
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Quantization config
        if config.load_in_4bit:
            compute_dtype = torch.bfloat16 if config.bnb_4bit_compute_dtype == "bf16" else torch.float16
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=False,
            )
            print(f"Using 4-bit NF4 quantization (compute_dtype={config.bnb_4bit_compute_dtype})")
        elif config.load_in_8bit:
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            bnb_config = None
        
        # Check Flash Attention 2 availability
        attn_impl = None
        if config.use_flash_attention:
            try:
                import flash_attn
                if torch.cuda.is_available():
                    major, minor = torch.cuda.get_device_capability()
                    if major >= 8:
                        attn_impl = "flash_attention_2"
                        print(f"Flash Attention enabled (flash_attention_2)")
                    else:
                        print(
                            "Flash Attention 2 detected but your GPU compute capability "
                            f"is {major}.{minor}; falling back to default attention"
                        )
                else:
                    print("Flash Attention 2 detected but CUDA is not available; falling back to default attention")
            except Exception as e:
                try:
                    import flash_attn_3
                    if torch.cuda.is_available():
                        major, minor = torch.cuda.get_device_capability()
                        if major >= 9:
                            attn_impl = "flash_attention_3"
                            print(f"Flash Attention enabled (flash_attention_3)")
                        else:
                            print(
                                "Flash Attention 3 detected but your GPU compute capability "
                                f"is {major}.{minor}; falling back to default attention"
                            )
                    else:
                        print("Flash Attention 3 detected but CUDA is not available; falling back to default attention")
                except Exception:
                    print(f"Flash Attention not available: {e}")
                    print("Using default SDPA attention")
        
        # Load base model
        model_kwargs = {
            "quantization_config": bnb_config,
            "device_map": device_map,
            "torch_dtype": self.torch_dtype,
        }
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
        
        try:
            self.llm = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                **model_kwargs,
            )
        except ValueError as e:
            msg = str(e).lower()
            # Some architectures (e.g. GPT-NeoX) don't support flash attention.
            # Some transformers versions don't recognize flash_attention_3 yet.
            if attn_impl and (
                "does not support" in msg
                or "attn_implementation" in msg
                or "attention implementation" in msg
            ):
                print(f"Attention backend '{attn_impl}' not supported, falling back to default")
                model_kwargs.pop("attn_implementation", None)
                self.llm = AutoModelForCausalLM.from_pretrained(
                    self.base_model_name,
                    **model_kwargs,
                )
            else:
                raise
        
        self._model_type = getattr(self.llm.config, 'model_type', 'unknown')
        print(f"Loaded {self.base_model_name} (model_type={self._model_type}, "
              f"hidden_size={self.llm.config.hidden_size}, "
              f"num_layers={getattr(self.llm.config, 'num_hidden_layers', '?')})")
        
        runtime_attn_impl = getattr(self.llm.config, "_attn_implementation", "unknown")
        print(f"LLM attention backend: {runtime_attn_impl}")
        if config.use_flash_attention and attn_impl and runtime_attn_impl != attn_impl:
            print(f"Warning: use_flash_attention=True but backend is not {attn_impl}")
        
        # Prepare for k-bit training (enables gradients for quantized models)
        if config.load_in_8bit or config.load_in_4bit:
            self.llm = prepare_model_for_kbit_training(self.llm)
        
        self._use_lora = getattr(config, 'use_lora', True)
        if self._use_lora:
            # Resolve LoRA target modules: "auto" triggers architecture-aware detection
            target_modules = config.lora.target_modules
            if target_modules == "auto" or target_modules == ["auto"]:
                target_modules = _auto_lora_target_modules(self.llm)
            print(f"LoRA target_modules: {target_modules}")
            
            # Apply LoRA
            lora_config = LoraConfig(
                r=config.lora.r,
                lora_alpha=config.lora.alpha,
                target_modules=target_modules,
                lora_dropout=config.lora.dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            self.llm = get_peft_model(self.llm, lora_config)
            print(f"LoRA applied (r={config.lora.r}, alpha={config.lora.alpha})")
        else:
            # No LoRA — full fine-tuning mode.
            # Freeze all LLM params initially; start_frozen config controls whether
            # they get unfrozen before training begins.
            for p in self.llm.parameters():
                p.requires_grad = False
            print("LoRA disabled — full fine-tuning mode (LLM params frozen until training loop applies start_frozen config)")

        # Ensure input embeddings require grads - CRITICAL for Adapter training
        # This ensures gradients backpropagate through the entire frozen LLM to reach the adapter
        # regardless of gradient checkpointing settings.
        if self._use_lora:
            self.llm.enable_input_require_grads()
        else:
            # Without PEFT, manually enable input require grads
            def _enable_grads_hook(module, input, output):
                output.requires_grad_(True)
            self.llm.get_input_embeddings().register_forward_hook(_enable_grads_hook)
        
        # Get LLM hidden dimension and attention head count
        llm_dim = self.llm.config.hidden_size
        llm_num_heads = getattr(self.llm.config, 'num_attention_heads', 16)
        
        # Chess position adapter
        if config.mode == "hybrid":
            # Use hybrid adapter (engineered features + LC0 embeddings)
            self.adapter = HybridPositionAdapter(
                lc0_dim=config.lc0_dim,
                lc0_proj_dim=config.hybrid.lc0_proj_dim,
                llm_dim=llm_dim,
                num_layers=4, # Hardcoded or add to config if really needed, but simplification was requested
            )
            print(f"Using HybridPositionAdapter (Engineered + LC0, lc0_proj_dim={config.hybrid.lc0_proj_dim})")
        elif config.mode == "engineered":
            self.adapter = EngineeredPositionAdapter(
                llm_dim=llm_dim,
            )
            print(f"Using EngineeredPositionAdapter (FEN-based, {ENGINEERED_FEATURE_DIM}-dim features)")
        elif config.mode == "perceiver":
            if PerceiverChessAdapter is None:
                raise ImportError("Perceiver adapter requested but training.perceiver_adapter module not found.")
            self.adapter = PerceiverChessAdapter(config, d_model=config.perceiver.d_model)
            print("Using PerceiverChessAdapter")
        elif config.mode == "maia":
            self.adapter = MaiaLLMAdapter(config)
            print(f"Using MaiaLLMAdapter (mode={config.maia.adapter_mode})")
        elif config.mode == "chess_fusion":
            self.adapter = ChessFusionAdapter(config, llm_dim=llm_dim, llm_num_heads=llm_num_heads)
            # Inject gated cross-attention into LLM layers AFTER LoRA is applied
            self.adapter.inject_into_llm(self.llm)
            print(f"Using ChessFusionAdapter (fusion tokens={config.chess_fusion.num_fusion_tokens})")
        else:
            raise ValueError(f"Unknown mode: {config.mode}")
        
        # Move adapter to same device as LLM embeddings
        self._sync_devices()
        
        # 1 side token + 64 square tokens = 65 prefix tokens
        self.num_prefix_tokens = self.adapter.get_num_prefix_tokens()

        # FEN tokens feature
        self.use_fen_tokens = getattr(config, "use_fen_tokens", False)
        # Runtime LM objective/generation gate (used in chess_fusion mode).
        self.lm_enabled = bool(getattr(config, "enable_lm", True)) if config.mode == "chess_fusion" else True
        
        # Apply torch.compile for faster forward/backward passes (PyTorch 2.0+)
        if config.use_torch_compile and hasattr(torch, 'compile'):
            if config.mode == 'chess_fusion':
                # Adapter stays eager (CNN forward hooks break torch.compile graph caching).
                # Instead, compile individual LLM decoder layers to fuse kernels and
                # reduce GPU idle time from many small kernel launches.
                self._compile_llm_layers()
            else:
                try:
                    self.adapter = torch.compile(self.adapter, mode="default")
                    print("torch.compile enabled on adapter (default mode)")
                except Exception as e:
                    print(f"torch.compile failed, using eager mode: {e}")
    
    def _compile_llm_layers(self):
        """Compile individual LLM decoder layers for chess_fusion mode.
        
        Strategy:
        - Regular decoder layers: compile the whole layer
        - FusionDecoderLayers: compile original_layer and gated_xattn separately
          (wrapper stays eager so dynamic _perceiver_latents state works)
        - Adapter stays eager (CNN forward hooks break torch.compile)
        """
        from training.chess_fusion_model import FusionDecoderLayer
        
        layers = _get_decoder_layers(self.llm)
        if layers is None:
            print("torch.compile: cannot locate decoder layers, skipping")
            return
        
        compiled_regular = 0
        compiled_fusion = 0
        for i, layer in enumerate(layers):
            try:
                if isinstance(layer, FusionDecoderLayer):
                    # Compile sub-components; wrapper stays eager for dynamic state
                    layer.original_layer = torch.compile(layer.original_layer)
                    if layer.gated_xattn is not None:
                        layer.gated_xattn = torch.compile(layer.gated_xattn)
                    if layer.pseudotoken_attn is not None:
                        layer.pseudotoken_attn = torch.compile(layer.pseudotoken_attn)
                    compiled_fusion += 1
                else:
                    layers[i] = torch.compile(layer)
                    compiled_regular += 1
            except Exception as e:
                print(f"torch.compile failed on layer {i}: {e}")
                break
        
        print(f"torch.compile enabled on LLM layers: "
              f"{compiled_regular} regular + {compiled_fusion} fusion "
              f"(mode=default/inductor)")

    def _sync_devices(self):
        """Ensure adapter is on the same device as LLM embeddings."""
        embed_device = next(self.llm.get_input_embeddings().parameters()).device
        self.adapter = self.adapter.to(embed_device)

    def set_lm_enabled(self, enabled: bool):
        """Enable/disable LM objective and generation at runtime (chess_fusion only)."""
        if self.config.mode == "chess_fusion":
            self.lm_enabled = bool(enabled)

    def is_lm_enabled(self) -> bool:
        return bool(self.lm_enabled)

    def _build_position_ids(self, attention_mask: torch.Tensor, prefix_len: int = 0) -> torch.Tensor:
        """Build LLM position ids, optionally making the prepended latent prefix position-neutral."""
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        if (
            self.config.mode != "chess_fusion"
            or prefix_len <= 0
            or bool(getattr(self.config.chess_fusion, "lm_prepend_latents_use_positional_encoding", True))
        ):
            return position_ids

        prefix_len = min(int(prefix_len), position_ids.size(1))
        if prefix_len <= 0:
            return position_ids

        # Keep the latent prefix position-neutral and let later tokens start at 0
        # so disabling prefix positional encoding does not shift the text sequence.
        position_ids[:, :prefix_len] = 0
        if prefix_len < position_ids.size(1):
            suffix_mask = attention_mask[:, prefix_len:]
            suffix_position_ids = suffix_mask.long().cumsum(-1) - 1
            suffix_position_ids.masked_fill_(suffix_mask == 0, 1)
            position_ids[:, prefix_len:] = suffix_position_ids
        return position_ids
    
    def forward(
        self,
        lc0_hidden_states: Optional[dict[str, torch.Tensor]],
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        side_to_move: Optional[torch.Tensor] = None,
        fen: Optional[list[str]] = None,
        engineered_features: Optional[torch.Tensor] = None,
        perceiver_features: Optional[tuple] = None,
        maia_features: Optional[torch.Tensor] = None,
        maia_policy: Optional[torch.Tensor] = None,
        maia_policy_mask: Optional[torch.Tensor] = None,
        move_eval_indices: Optional[torch.Tensor] = None,
        move_eval_targets: Optional[torch.Tensor] = None,
        move_eval_mask: Optional[torch.Tensor] = None,
        move_ce_indices: Optional[torch.Tensor] = None,
        move_ce_targets: Optional[torch.Tensor] = None,
        move_ce_mask: Optional[torch.Tensor] = None,
        eval_targets: Optional[torch.Tensor] = None,
        loss_weights: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass combining chess embeddings with text.

        Args:
            lc0_hidden_states: Dict with layer tensors, each (B, 64, 768).
                               Ignored when use_engineered_features=True.
            input_ids: Text token IDs (B, seq_len)
            attention_mask: Attention mask (B, seq_len)
            labels: Target labels for loss (B, seq_len)
            eval_targets: (B,) eval bucket class indices for Stockfish position eval
            side_to_move: Boolean tensor (B,) - True=White, False=Black
            fen: List of FEN strings (for LC0 adapter piece encoding)
            engineered_features: Pre-computed features (B, 64, ENGINEERED_FEATURE_DIM)
            perceiver_features: Tuple of (sq_features, glob_features)
            maia_features: Tensor (B, 18, 8, 8)
            maia_policy: Precomputed (B, 1880) teacher policy logits
            maia_policy_mask: (B,) bool mask for valid maia_policy rows
            move_eval_indices: (B, M) vocab indices for per-move eval supervision
            move_eval_targets: (B, M) centipawn eval targets (float)
            move_eval_mask: (B, M) bool mask for valid entries
            move_ce_indices: (B, K) CE-candidate move indices (prefer Stockfish top-k)
            move_ce_targets: (B, K) CE-candidate centipawn values
            move_ce_mask: (B, K) bool mask for valid CE candidates

        Returns:
            Model outputs with loss if labels provided
        """
        batch_size = input_ids.shape[0]
        
        # Get adapter device (where we need all tensors to be)
        adapter_device = next(self.adapter.parameters()).device
        
        # 1. Get position embeddings from adapter (1 side token + 64 square embeddings)
        if self.config.mode == "hybrid":
            # Hybrid adapter takes both LC0 hidden states and engineered features
            if engineered_features is None:
                raise ValueError("engineered_features must be provided when mode='hybrid'")
            lc0_states = {k: v.to(adapter_device) for k, v in lc0_hidden_states.items()}
            position_embeds = self.adapter(lc0_states, engineered_features, side_to_move=side_to_move)  # (B, 65, hidden_size)
        elif self.config.mode == "engineered":
            # Engineered adapter takes pre-computed feature tensors
            if engineered_features is None:
                raise ValueError("engineered_features must be provided when mode='engineered'")
            position_embeds = self.adapter(engineered_features, side_to_move=side_to_move)  # (B, 65, hidden_size)
        elif self.config.mode == "perceiver":
            if perceiver_features is None:
                 raise ValueError("perceiver_features must be provided when mode='perceiver'")
            # perceiver_features is tuple (sq, gl)
            # Move to adapter device
            sq, gl = perceiver_features
            perceiver_features = (sq.to(adapter_device), gl.to(adapter_device))
            position_embeds = self.adapter(perceiver_features, side_to_move=side_to_move)
        elif self.config.mode == "maia":
            if maia_features is None:
                 raise ValueError("maia_features must be provided when mode='maia'")
            maia_features = maia_features.to(adapter_device)
            # Maia adapter handles ELO defaults internally
            if getattr(self.config.maia, "use_main_engineered_concat", False):
                if engineered_features is None:
                    raise ValueError("engineered_features must be provided when Maia perceiver uses main engineered concat")
                engineered_features = engineered_features.to(adapter_device)
                position_embeds = self.adapter(
                    maia_features,
                    side_to_move=side_to_move,
                    engineered_features=engineered_features,
                )
            else:
                position_embeds = self.adapter(maia_features, side_to_move=side_to_move)
        elif self.config.mode == "chess_fusion":
            engineered_only_ablation = bool(
                getattr(self.config.chess_fusion, "engineered_only_xattn_ablation", False)
            )
            if engineered_only_ablation:
                maia_features = None
            else:
                if maia_features is None:
                    raise ValueError("maia_features must be provided when mode='chess_fusion'")
                maia_features = maia_features.to(adapter_device)
            fusion_kwargs = {'side_to_move': side_to_move}
            needs_main_engineered_features = (
                getattr(self.config.chess_fusion, "use_engineered_concat", False)
                or getattr(self.config.chess_fusion, "xattn_structured_use_engineered_source", False)
                or engineered_only_ablation
            )
            if needs_main_engineered_features:
                if engineered_features is None:
                    raise ValueError(
                        "engineered_features required for chess_fusion when "
                        "use_engineered_concat, xattn_structured_use_engineered_source, "
                        "or engineered_only_xattn_ablation is enabled"
                    )
                fusion_kwargs['engineered_features'] = engineered_features.to(adapter_device)
            if maia_policy is not None:
                fusion_kwargs['precomputed_policy'] = maia_policy
            _profile = getattr(self.adapter, '_profile', False)
            if _profile:
                import time as _time
                torch.cuda.synchronize()
                _ta0 = _time.perf_counter()
            adapter_out = self.adapter(maia_features, **fusion_kwargs)
            if _profile:
                torch.cuda.synchronize()
                _ta1 = _time.perf_counter()
            perceiver_latents = adapter_out['perceiver_latents']  # (B, N_lat, perceiver_dim)
            # Store auxiliary outputs for loss computation later
            self._fusion_policy_logits = adapter_out['policy_logits']
            self._fusion_eval_logits = adapter_out['eval_logits']
            self._fusion_eval_targets = eval_targets
            self._fusion_policy_targets = adapter_out['policy_targets']
            self._fusion_policy_mask = maia_policy_mask
            self._fusion_entropy_metrics = adapter_out.get('entropy_metrics', {})
            self._fusion_bsr_logits = adapter_out.get('bsr_logits', None)
            self._fusion_bsr_targets = adapter_out.get('bsr_targets', None)
            self._fusion_spp_preds = adapter_out.get('spp_preds', None)
            self._fusion_spp_targets = adapter_out.get('spp_targets', None)
            self._fusion_move_eval_logits = adapter_out.get('move_eval_logits', None)
            self._fusion_move_eval_mse_logits = adapter_out.get('move_eval_mse_logits', None)
            self._fusion_move_mate_logits = adapter_out.get('move_mate_logits', None)
            # Store batch-provided move eval supervision data
            self._fusion_move_eval_indices = move_eval_indices
            self._fusion_move_eval_targets = move_eval_targets
            self._fusion_move_eval_mask = move_eval_mask
            self._fusion_move_ce_indices = move_ce_indices
            self._fusion_move_ce_targets = move_ce_targets
            self._fusion_move_ce_mask = move_ce_mask
            # Set Perceiver latents on injected LLM layers (each layer reads them via its own readout)
            ctx = adapter_out.get('context', None)
            if ctx is not None:
                ctx = ctx.to(dtype=self.torch_dtype)
            csmp_square_tokens = adapter_out.get('csmp_square_tokens', None)
            if csmp_square_tokens is not None:
                csmp_square_tokens = csmp_square_tokens.to(dtype=self.torch_dtype)
            engineered_square_features = adapter_out.get('engineered_square_features', None)
            if engineered_square_features is not None:
                engineered_square_features = engineered_square_features.to(dtype=self.torch_dtype)
            prepend_embeddings = adapter_out.get('prepend_embeddings', None)
            if prepend_embeddings is not None:
                if prepend_embeddings.size(1) != self.num_prefix_tokens:
                    raise ValueError(
                        "Mismatch between adapter prefix token count and model expectation: "
                        f"{prepend_embeddings.size(1)} vs {self.num_prefix_tokens}"
                    )
                position_embeds = prepend_embeddings
            else:
                # No text-space prepend latents configured; chess signal is injected via x-attn only.
                position_embeds = perceiver_latents.new_zeros(batch_size, 0, self.llm.config.hidden_size)
        else:
             raise ValueError(f"Unknown mode: {self.config.mode}")

        if self.config.mode == "chess_fusion" and not self.lm_enabled:
            # Adapter-only training while keeping LLM loaded for later re-enable.
            self.adapter.clear_chess_context()
            aux_losses = self.adapter.compute_auxiliary_losses(
                policy_logits=self._fusion_policy_logits,
                eval_logits=self._fusion_eval_logits,
                policy_targets=self._fusion_policy_targets,
                policy_mask=self._fusion_policy_mask,
                eval_targets=self._fusion_eval_targets,
                bsr_logits=self._fusion_bsr_logits,
                bsr_targets=self._fusion_bsr_targets,
                spp_preds=self._fusion_spp_preds,
                spp_targets=self._fusion_spp_targets,
                move_eval_logits=self._fusion_move_eval_logits,
                move_eval_mse_logits=self._fusion_move_eval_mse_logits,
                move_mate_logits=self._fusion_move_mate_logits,
                move_eval_indices=self._fusion_move_eval_indices,
                move_eval_targets=self._fusion_move_eval_targets,
                move_eval_mask=self._fusion_move_eval_mask,
                move_ce_indices=self._fusion_move_ce_indices,
                move_ce_targets=self._fusion_move_ce_targets,
                move_ce_mask=self._fusion_move_ce_mask,
            )
            self._last_aux_losses = {k: v.detach() for k, v in aux_losses.items()}
            self._fusion_policy_logits = None
            self._fusion_eval_logits = None
            self._fusion_eval_targets = None
            self._fusion_policy_targets = None
            self._fusion_policy_mask = None
            self._fusion_bsr_logits = None
            self._fusion_bsr_targets = None
            self._fusion_spp_preds = None
            self._fusion_spp_targets = None
            self._fusion_move_eval_logits = None
            self._fusion_move_eval_mse_logits = None
            self._fusion_move_mate_logits = None
            self._fusion_move_eval_indices = None
            self._fusion_move_eval_targets = None
            self._fusion_move_eval_mask = None
            self._fusion_move_ce_indices = None
            self._fusion_move_ce_targets = None
            self._fusion_move_ce_mask = None

            class _Output:
                pass

            output = _Output()
            output.loss = aux_losses['total_aux_loss']
            return output
        
        # Move input tensors to same device as LLM
        device = next(self.llm.parameters()).device
        input_ids = input_ids.to(device)
        
        # 2. Get token embeddings from LLM
        token_embeds = self.llm.get_input_embeddings()(input_ids)  # (B, seq, hidden)
        
        # 3. Cast adapter output to match LLM embedding dtype (float16)
        position_embeds = position_embeds.to(dtype=token_embeds.dtype)
        
        # 4. Optionally add FEN tokens right after position embeddings
        if self.use_fen_tokens and fen is not None:
            # Tokenize FEN strings for batch
            fen_encodings = self.tokenizer(
                fen,
                padding=True,
                truncation=True,
                max_length=64,  # FEN is ~70 chars max, leaves some buffer
                return_tensors="pt",
            )
            fen_ids = fen_encodings["input_ids"].to(device)
            fen_mask = fen_encodings["attention_mask"].to(device)
            fen_embeds = self.llm.get_input_embeddings()(fen_ids)  # (B, fen_len, hidden)
            self._fen_len = fen_ids.shape[1]  # Store for mask/label extension
            self._fen_mask = fen_mask
        else:
            fen_embeds = None
            self._fen_len = 0
            self._fen_mask = None
        
        # 5. Prepend position embeddings (side token + squares) as prefix
        if fen_embeds is not None:
            combined_embeds = torch.cat([position_embeds, fen_embeds, token_embeds], dim=1)
        else:
            combined_embeds = torch.cat([position_embeds, token_embeds], dim=1)
        
        # 6. Extend attention mask for prefix tokens (and optional FEN tokens)
        attention_mask = attention_mask.to(device)
        prefix_mask = torch.ones(
            (batch_size, self.num_prefix_tokens),
            dtype=attention_mask.dtype,
            device=device
        )
        if self._fen_len > 0 and self._fen_mask is not None:
            combined_mask = torch.cat([prefix_mask, self._fen_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if self.config.mode == "chess_fusion":
            policy_latents = adapter_out.get('policy_latents', None)
            if policy_latents is not None:
                policy_latents = policy_latents.to(dtype=self.torch_dtype)
            self.adapter.set_chess_context(
                perceiver_latents.to(dtype=self.torch_dtype),
                ctx,
                csmp_square_tokens=csmp_square_tokens,
                text_attention_mask=combined_mask,
                policy_latents=policy_latents,
                engineered_square_features=engineered_square_features,
            )
        
        # 7. Extend labels if provided (use -100 for prefix and FEN - no loss)
        if labels is not None:
            labels = labels.to(device)
            prefix_len = self.num_prefix_tokens + self._fen_len
            prefix_labels = torch.full(
                (batch_size, prefix_len),
                -100,
                dtype=labels.dtype,
                device=device
            )
            combined_labels = torch.cat([prefix_labels, labels], dim=1)
        else:
            combined_labels = None

        # Extend loss_weights with prefix (value irrelevant — prefix has label -100)
        _has_loss_weights = loss_weights is not None
        combined_loss_weights = None
        if _has_loss_weights and combined_labels is not None:
            loss_weights = loss_weights.to(device)
            prefix_weights = torch.ones(
                (batch_size, prefix_len),
                dtype=torch.float32,
                device=device,
            )
            combined_loss_weights = torch.cat([prefix_weights, loss_weights], dim=1)

        # In chess_fusion, allow runtime LM disable while keeping LLM loaded.
        # This gates LM objective (adapter aux losses can still train).
        if self.config.mode == "chess_fusion" and not self.lm_enabled:
            combined_labels = None
            combined_loss_weights = None

        # 6. Forward through LLM
        # Explicitly generate position_ids to ensure Flash Attention handles inputs_embeds correctly
        # Combined mask is (B, Total_Len) with 1s for valid, 0s for pad
        # Standard position_ids: cumsum of mask, minus 1, with pads set to 1
        position_ids = self._build_position_ids(
            combined_mask,
            prefix_len=self.num_prefix_tokens,
        )

        if combined_embeds.dtype != self.torch_dtype:
            combined_embeds = combined_embeds.to(dtype=self.torch_dtype)

        _profile = getattr(self, '_profile_forward', False)
        if _profile:
            import time as _time
            from training.chess_fusion_model import FusionDecoderLayer
            FusionDecoderLayer._profile_enabled = True
            FusionDecoderLayer.reset_profile()
            torch.cuda.synchronize()
            _tl0 = _time.perf_counter()

        _use_amp = (
            combined_embeds.is_cuda
            and self.torch_dtype in (torch.float16, torch.bfloat16)
        )
        # When chess token weights are active, skip HF's internal loss and compute it ourselves.
        _labels_for_hf = None if combined_loss_weights is not None else combined_labels
        _amp_ctx = torch.amp.autocast('cuda', dtype=self.torch_dtype) if _use_amp else nullcontext()
        with _amp_ctx:
            outputs = self.llm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_mask,
                position_ids=position_ids,
                labels=_labels_for_hf,
                use_cache=False,
            )

        # Compute weighted LM loss when chess token weights are active
        if combined_loss_weights is not None and combined_labels is not None:
            import torch.nn.functional as F
            # Shift logits and labels (standard causal LM pattern)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = combined_labels[..., 1:].contiguous()
            shift_weights = combined_loss_weights[..., 1:].contiguous()

            # Flatten
            vocab_size = shift_logits.size(-1)
            flat_logits = shift_logits.view(-1, vocab_size)
            flat_labels = shift_labels.view(-1)
            flat_weights = shift_weights.view(-1)

            # Per-token CE (ignore_index=-100 zeroes out prompt/padding positions)
            per_token_loss = F.cross_entropy(
                flat_logits.float(), flat_labels, reduction='none', ignore_index=-100,
            )

            # Apply weights (only non-ignored positions contribute)
            valid_mask = (flat_labels != -100)
            weighted_loss = (per_token_loss * flat_weights)
            # Normalise by sum of weights over valid positions (not count) to keep
            # the loss scale comparable to unweighted CE.
            denom = flat_weights[valid_mask].sum().clamp(min=1.0)
            outputs.loss = weighted_loss.sum() / denom

            # Store metrics for logging: mean weight over valid tokens and
            # unweighted LM loss for comparison.
            n_valid = valid_mask.sum().clamp(min=1)
            self._chess_token_weight_metrics = {
                'mean_weight': flat_weights[valid_mask].mean().detach(),
                'unweighted_lm_loss': (per_token_loss.sum() / n_valid).detach(),
                'weighted_frac': ((flat_weights[valid_mask] > 1.0).float().mean()).detach(),
            }
        
        if _profile:
            torch.cuda.synchronize()
            _tl1 = _time.perf_counter()
            _total = (_tl1 - _tl0) * 1000
            _orig_list  = FusionDecoderLayer._profile_original_ms
            _xattn_list = FusionDecoderLayer._profile_xattn_ms
            _n = FusionDecoderLayer._profile_count
            _orig  = sum(_orig_list)  if isinstance(_orig_list,  list) else _orig_list
            _xattn = sum(_xattn_list) if isinstance(_xattn_list, list) else _xattn_list
            _llm_only = _total - _xattn
            print(f"  [PROFILE] LLM forward = {_total:.1f}ms  "
                  f"(LLM self-attn ~{_llm_only:.1f}ms  |  "
                  f"readouts = {_xattn:.1f}ms across {_n} fusion layers, "
                  f"avg {(_xattn/_n):.1f}ms/layer)" if _n > 0 else
                  f"  [PROFILE] LLM forward = {_total:.1f}ms (no fusion layers active)")
            FusionDecoderLayer._profile_enabled = False
        
        # Clear chess context from fusion layers after forward pass
        if self.config.mode == "chess_fusion":
            self.adapter.clear_chess_context()
            # Compute and add auxiliary losses
            aux_losses = self.adapter.compute_auxiliary_losses(
                policy_logits=self._fusion_policy_logits,
                eval_logits=self._fusion_eval_logits,
                policy_targets=self._fusion_policy_targets,
                policy_mask=self._fusion_policy_mask,
                eval_targets=self._fusion_eval_targets,
                bsr_logits=self._fusion_bsr_logits,
                bsr_targets=self._fusion_bsr_targets,
                spp_preds=self._fusion_spp_preds,
                spp_targets=self._fusion_spp_targets,
                move_eval_logits=self._fusion_move_eval_logits,
                move_eval_mse_logits=self._fusion_move_eval_mse_logits,
                move_mate_logits=self._fusion_move_mate_logits,
                move_eval_indices=self._fusion_move_eval_indices,
                move_eval_targets=self._fusion_move_eval_targets,
                move_eval_mask=self._fusion_move_eval_mask,
                move_ce_indices=self._fusion_move_ce_indices,
                move_ce_targets=self._fusion_move_ce_targets,
                move_ce_mask=self._fusion_move_ce_mask,
            )
            if outputs.loss is not None:
                outputs.loss = outputs.loss + aux_losses['total_aux_loss']
            else:
                outputs.loss = aux_losses['total_aux_loss']
            # Store scalar values for logging, then release graph tensors
            self._last_aux_losses = {
                k: v.detach() for k, v in aux_losses.items()
            }
            # Release intermediate fusion tensors (no longer needed after aux loss)
            self._fusion_policy_logits = None
            self._fusion_eval_logits = None
            self._fusion_eval_targets = None
            self._fusion_policy_targets = None
            self._fusion_policy_mask = None
            self._fusion_bsr_logits = None
            self._fusion_bsr_targets = None
            self._fusion_spp_preds = None
            self._fusion_spp_targets = None
            self._fusion_move_eval_logits = None
            self._fusion_move_eval_mse_logits = None
            self._fusion_move_mate_logits = None
            self._fusion_move_eval_indices = None
            self._fusion_move_eval_targets = None
            self._fusion_move_eval_mask = None
            self._fusion_move_ce_indices = None
            self._fusion_move_ce_targets = None
            self._fusion_move_ce_mask = None
        
        return outputs

    def _build_generation_prompt_inputs(self, prompt: str) -> tuple[str, dict[str, torch.Tensor]]:
        messages = [{"role": "user", "content": prompt}]
        prompt_text = _safe_apply_chat_template(
            self.tokenizer, messages, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt_text,
            return_tensors="pt",
            padding=True,
        )
        return prompt_text, inputs

    def _prepare_generation_position_state(
        self,
        lc0_hidden_states: Optional[dict[str, torch.Tensor]],
        side_to_move: bool = True,
        fen: Optional[str] = None,
        maia_policy: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        device = next(self.llm.parameters()).device
        batch_size = 1
        generation_context: Optional[Dict[str, Optional[torch.Tensor]]] = None

        if self.config.mode == "hybrid":
            if fen is None:
                raise ValueError("FEN is required for hybrid features mode")
            feat_mode = getattr(self.config, "engineered_features_type", "simplified")
            features = extract_engineered_features(fen, mode=feat_mode).unsqueeze(0).to(device)
            lc0_states = {
                k: v.unsqueeze(0).to(device) if v.dim() == 2 else v.to(device)
                for k, v in lc0_hidden_states.items()
            }
            side_tensor = torch.tensor([side_to_move], dtype=torch.bool, device=device)
            position_embeds = self.adapter(lc0_states, features, side_to_move=side_tensor)
        elif self.config.mode == "engineered":
            if fen is None:
                raise ValueError("FEN is required for engineered features mode")
            feat_mode = getattr(self.config, "engineered_features_type", "simplified")
            features = extract_engineered_features(fen, mode=feat_mode).unsqueeze(0).to(device)
            side_tensor = torch.tensor([side_to_move], dtype=torch.bool, device=device)
            position_embeds = self.adapter(features, side_to_move=side_tensor)
        elif self.config.mode == "maia":
            if fen is None:
                raise ValueError("FEN is required for maia features mode")
            features = extract_maia_features(fen).unsqueeze(0).to(device)
            side_tensor = torch.tensor([side_to_move], dtype=torch.bool, device=device)
            if getattr(self.config.maia, "use_main_engineered_concat", False):
                engineered = extract_engineered_features(fen, mode="main").unsqueeze(0).to(device)
                position_embeds = self.adapter(
                    features,
                    side_to_move=side_tensor,
                    engineered_features=engineered,
                )
            else:
                position_embeds = self.adapter(features, side_to_move=side_tensor)
        elif self.config.mode == "chess_fusion":
            if fen is None:
                raise ValueError("FEN is required for chess_fusion mode")
            engineered_only_ablation = bool(
                getattr(self.config.chess_fusion, "engineered_only_xattn_ablation", False)
            )
            features = None
            if not engineered_only_ablation:
                features = extract_maia_features(fen).unsqueeze(0).to(device)
            side_tensor = torch.tensor([side_to_move], dtype=torch.bool, device=device)
            fusion_kwargs = {"side_to_move": side_tensor}
            if maia_policy is not None:
                fusion_kwargs["precomputed_policy"] = maia_policy.to(device)
            if (
                getattr(self.config.chess_fusion, "use_engineered_concat", False)
                or getattr(self.config.chess_fusion, "xattn_structured_use_engineered_source", False)
                or engineered_only_ablation
            ):
                engineered = extract_engineered_features(fen, mode="main").unsqueeze(0).to(device)
                fusion_kwargs["engineered_features"] = engineered
            adapter_out = self.adapter(features, **fusion_kwargs)
            self._last_adapter_out = adapter_out
            perceiver_latents = adapter_out["perceiver_latents"]
            policy_latents = adapter_out.get("policy_latents", None)
            ctx = adapter_out.get("context", None)
            if ctx is not None:
                ctx = ctx.to(dtype=self.torch_dtype)
            csmp_square_tokens = adapter_out.get("csmp_square_tokens", None)
            if csmp_square_tokens is not None:
                csmp_square_tokens = csmp_square_tokens.to(dtype=self.torch_dtype)
            if policy_latents is not None:
                policy_latents = policy_latents.to(dtype=self.torch_dtype)
            engineered_square_features = adapter_out.get("engineered_square_features", None)
            if engineered_square_features is not None:
                engineered_square_features = engineered_square_features.to(dtype=self.torch_dtype)
            generation_context = {
                "perceiver_latents": perceiver_latents.to(dtype=self.torch_dtype),
                "context": ctx,
                "csmp_square_tokens": csmp_square_tokens,
                "policy_latents": policy_latents,
                "engineered_square_features": engineered_square_features,
            }
            prepend_embeddings = adapter_out.get("prepend_embeddings", None)
            if prepend_embeddings is not None:
                if prepend_embeddings.size(1) != self.num_prefix_tokens:
                    raise ValueError(
                        "Mismatch between adapter prefix token count and model expectation: "
                        f"{prepend_embeddings.size(1)} vs {self.num_prefix_tokens}"
                    )
                position_embeds = prepend_embeddings
            else:
                position_embeds = perceiver_latents.new_zeros(
                    batch_size, 0, self.llm.config.hidden_size
                )
        else:
            raise ValueError(f"Unknown mode: {self.config.mode}")

        return {
            "position_embeds": position_embeds,
            "generation_context": generation_context,
        }

    def _build_generation_model_inputs(
        self,
        position_embeds: torch.Tensor,
        prompt_inputs: dict[str, torch.Tensor],
        fen: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        device = next(self.llm.parameters()).device
        input_ids = prompt_inputs["input_ids"].to(device)
        attention_mask = prompt_inputs["attention_mask"].to(device)
        batch_size = int(input_ids.size(0))

        token_embeds = self.llm.get_input_embeddings()(input_ids)
        position_embeds = position_embeds.to(
            device=token_embeds.device,
            dtype=token_embeds.dtype,
        )

        if self.use_fen_tokens and fen is not None:
            fen_encodings = self.tokenizer(
                [fen],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            fen_ids = fen_encodings["input_ids"].to(device)
            fen_mask = fen_encodings["attention_mask"].to(device)
            fen_embeds = self.llm.get_input_embeddings()(fen_ids)
            fen_len = int(fen_ids.shape[1])
        else:
            fen_embeds = None
            fen_mask = None
            fen_len = 0

        if fen_embeds is not None:
            combined_embeds = torch.cat([position_embeds, fen_embeds, token_embeds], dim=1)
        else:
            combined_embeds = torch.cat([position_embeds, token_embeds], dim=1)

        prefix_mask = torch.ones(
            (batch_size, self.num_prefix_tokens),
            dtype=attention_mask.dtype,
            device=device,
        )
        if fen_len > 0 and fen_mask is not None:
            combined_mask = torch.cat([prefix_mask, fen_mask, attention_mask], dim=1)
        else:
            combined_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        position_ids = self._build_position_ids(
            combined_mask,
            prefix_len=self.num_prefix_tokens,
        )

        return {
            "prompt_input_ids": input_ids,
            "prompt_attention_mask": attention_mask,
            "combined_embeds": combined_embeds,
            "combined_mask": combined_mask,
            "position_ids": position_ids,
        }

    def prepare_generation_inputs(
        self,
        lc0_hidden_states: Optional[dict[str, torch.Tensor]],
        side_to_move: bool = True,
        prompt: str = "Provide commentary.",
        fen: Optional[str] = None,
        maia_policy: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        prompt_text, prompt_inputs = self._build_generation_prompt_inputs(prompt)
        position_state = self._prepare_generation_position_state(
            lc0_hidden_states=lc0_hidden_states,
            side_to_move=side_to_move,
            fen=fen,
            maia_policy=maia_policy,
        )
        model_inputs = self._build_generation_model_inputs(
            position_embeds=position_state["position_embeds"],
            prompt_inputs=prompt_inputs,
            fen=fen,
        )
        return {
            "prompt_text": prompt_text,
            "generation_context": position_state["generation_context"],
            **model_inputs,
        }

    def set_generation_context(
        self,
        generation_context: Optional[Dict[str, Optional[torch.Tensor]]],
        text_attention_mask: Optional[torch.Tensor] = None,
    ) -> None:
        if self.config.mode != "chess_fusion" or generation_context is None:
            return

        self.adapter.set_chess_context(
            generation_context["perceiver_latents"],
            generation_context.get("context"),
            csmp_square_tokens=generation_context.get("csmp_square_tokens"),
            text_attention_mask=text_attention_mask,
            policy_latents=generation_context.get("policy_latents"),
            engineered_square_features=generation_context.get("engineered_square_features"),
        )

    def clear_generation_context(self) -> None:
        if self.config.mode == "chess_fusion":
            self.adapter.clear_chess_context()

    def generate(
        self,
        lc0_hidden_states: dict[str, torch.Tensor],
        side_to_move: bool = True,
        prompt: str = "Provide commentary.",
        max_new_tokens: int = 256,
        min_new_tokens: int = 0,
        temperature: float = 0.7,
        top_p: float = 0.9,
        fen: Optional[str] = None,
        maia_policy: Optional[torch.Tensor] = None,
        return_ids: bool = False,
    ) -> str:
        """
        Generate commentary for a chess position.
        
        Args:
            lc0_hidden_states: LC0 hidden states for the position
            side_to_move: True = White to move, False = Black to move
            prompt: Text prompt (minimal by design)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        
        Returns:
            Generated commentary string
        """
        self.eval()

        with torch.inference_mode():
            generation_inputs = self.prepare_generation_inputs(
                lc0_hidden_states=lc0_hidden_states,
                side_to_move=side_to_move,
                prompt=prompt,
                fen=fen,
                maia_policy=maia_policy,
            )
            if self.config.mode == "chess_fusion" and not self.lm_enabled:
                self.clear_generation_context()
                return "[LM disabled] Adapter-only mode active; commentary generation skipped."

            self.set_generation_context(
                generation_inputs["generation_context"],
                text_attention_mask=generation_inputs["combined_mask"],
            )
            try:
                outputs = self.llm.generate(
                    inputs_embeds=generation_inputs["combined_embeds"],
                    attention_mask=generation_inputs["combined_mask"],
                    position_ids=generation_inputs["position_ids"],
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            finally:
                self.clear_generation_context()

        generated_ids = outputs[0]
        commentary = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        if return_ids:
            return commentary, generated_ids
        return commentary
    
    def save_pretrained(self, output_dir: str, save_merged_base: bool = False):
        """Save adapter and LoRA weights (no base model)."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        merged_base_path = output_path / "merged_base"
        if merged_base_path.exists():
            shutil.rmtree(merged_base_path, ignore_errors=True)
        
        # Save LLM weights (LoRA or full)
        if self._use_lora:
            self.llm.save_pretrained(output_path / "lora")
        else:
            # No LoRA — save full LLM state dict
            torch.save(self.llm.state_dict(), output_path / "llm_full.pt")
        
        # Save adapter weights
        adapter_state, dropped = _prepare_adapter_state_dict_for_save(self.adapter)
        torch.save(adapter_state, output_path / "adapter.pt")
        if dropped:
            print(f"[Save] Excluded {dropped} teacher_backbone checkpoint keys from adapter.pt")
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_path / "tokenizer")

        if save_merged_base:
            print("Skipping merged base save; adapter-only checkpoints are enabled.")
        
        print(f"Saved model to {output_dir}")
    
    @classmethod
    def load_pretrained(
        cls, 
        checkpoint_dir: str, 
        config: Optional[ModelConfig] = None,
    ):
        """Load saved adapter and LoRA weights."""
        checkpoint_path = Path(checkpoint_dir)
        
        # If config not provided, try to load from checkpoint (TODO: save config with checkpoint)
        # For now, we assume config is passed or we default to something safe (but hybrid/engineered mode matters)
        if config is None:
            # Try to infer mode from adapter weights? Or just default to hybrid
            # Better to require config if we can't infer
            # For this refactor, let's assume config is passed in train() loop
             raise ValueError("config object required for load_pretrained")

        # Create model (initializes with random LoRA weights)
        model = cls(config)
        
        # Load adapter weights
        adapter_path = checkpoint_path / "adapter.pt"
        if adapter_path.exists():
            print(f"Loading adapter weights from {adapter_path}")
            state_dict = torch.load(adapter_path, map_location="cpu", weights_only=False)
            state_dict, dropped = _filter_chess_fusion_adapter_state_dict(state_dict, model.config)
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
        
        # Load LLM weights (LoRA or full)
        if bool(getattr(model.config, "load_lm_checkpoint", True)):
            if model._use_lora:
                lora_path = checkpoint_path / "lora"
                if lora_path.exists():
                    print(f"Loading LoRA weights from {lora_path}")
                    try:
                        model.llm.load_adapter(str(lora_path), adapter_name="default")
                    except Exception as e:
                        print(f"Error loading LoRA weights: {e}")
                        print("Attempting alternative loading method...")
                        from peft import PeftModel
                        base = model.llm.base_model.model
                        model.llm = PeftModel.from_pretrained(base, str(lora_path))
            else:
                llm_path = checkpoint_path / "llm_full.pt"
                if llm_path.exists():
                    print(f"Loading full LLM weights from {llm_path}")
                    model.llm.load_state_dict(torch.load(llm_path, weights_only=False))
        else:
            print("Skipping LM weight loading (config.model.load_lm_checkpoint=false)")
        
        return model
    
    def resume_checkpoint_weights(self, checkpoint_dir: str):
        """Load adapter + LoRA weights from a prior checkpoint (model weights only, no training state).
        
        Use this when starting training on a new dataset from a previous checkpoint.
        Unlike load_pretrained(), this operates on an already-constructed model.
        Handles policy-only checkpoints (adapter.pt only, no LLM weights) gracefully.
        """
        checkpoint_root = Path(checkpoint_dir)
        if not checkpoint_root.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Smart directory resolution: if the dir doesn't have weights, check subdirs
        checkpoint_path = checkpoint_root
        if not (checkpoint_path / "adapter.pt").exists() and not (checkpoint_path / "lora").exists() and not (checkpoint_path / "llm_full.pt").exists():
            print(f"[Resume] Weights not found in {checkpoint_root}. Searching subdirectories...")
            checkpoint_path = self._find_valid_checkpoint_subdir(checkpoint_root)
        
        # Load adapter weights (with corruption fallback)
        adapter_loaded = self._try_load_adapter(checkpoint_path, checkpoint_root)
        if not adapter_loaded:
            print(f"[Resume] WARNING: Could not load adapter weights from any checkpoint in {checkpoint_root}")
        
        # Load LLM weights (LoRA or full) — may not exist if resuming from policy-only checkpoint
        is_policy_only_ckpt = (checkpoint_path / "policy_only_meta.pt").exists()
        if is_policy_only_ckpt:
            print(f"[Resume] Detected policy-only checkpoint — skipping LLM weight loading (using fresh LLM weights)")
        elif not bool(getattr(self.config, "load_lm_checkpoint", True)):
            print(f"[Resume] Skipping LLM weight loading (model.load_lm_checkpoint=false)")
        elif self._use_lora:
            lora_path = checkpoint_path / "lora"
            if lora_path.exists():
                print(f"[Resume] Loading LoRA weights from {lora_path}")
                try:
                    self.llm.load_adapter(str(lora_path), adapter_name="default")
                except Exception as e:
                    print(f"[Resume] Error loading LoRA: {e}")
            else:
                print(f"[Resume] No lora/ directory found in {checkpoint_path} (starting with fresh LoRA weights)")
        else:
            llm_path = checkpoint_path / "llm_full.pt"
            if llm_path.exists():
                print(f"[Resume] Loading full LLM weights from {llm_path}")
                self.llm.load_state_dict(torch.load(llm_path, map_location="cpu", weights_only=False))
            else:
                print(f"[Resume] No llm_full.pt found in {checkpoint_path} (starting with fresh LLM weights)")
        
        print(f"[Resume] Checkpoint weights loaded from {checkpoint_path}")
    
    def _find_valid_checkpoint_subdir(self, checkpoint_root: Path) -> Path:
        """Find the latest valid checkpoint subdirectory."""
        subdirs = sorted(
            [d for d in checkpoint_root.iterdir() if d.is_dir() and (d.name.startswith("epoch-") or d.name.startswith("checkpoint-"))],
            key=lambda x: (x.name.split("-")[0], int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0),
            reverse=True
        )
        if subdirs:
            print(f"[Resume] Automatically selected latest subdirectory: {subdirs[0]}")
            return subdirs[0]
        print(f"[Resume] No epoch/checkpoint subdirectories found in {checkpoint_root}")
        return checkpoint_root
    
    def _try_load_adapter(self, checkpoint_path: Path, checkpoint_root: Path) -> bool:
        """Try to load adapter.pt, falling back to earlier checkpoints if corrupted."""
        adapter_path = checkpoint_path / "adapter.pt"
        if adapter_path.exists():
            try:
                print(f"[Resume] Loading adapter weights from {adapter_path}")
                state_dict = torch.load(adapter_path, map_location="cpu", weights_only=False)
                state_dict, dropped = _filter_chess_fusion_adapter_state_dict(state_dict, self.config)
                if dropped:
                    dropped_str = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
                    print(f"[Resume] Selective adapter loading enabled; skipped keys: {dropped_str}")
                missing, unexpected = self.adapter.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[Resume] Missing keys (expected for cross-mode resume): {len(missing)} keys")
                    print(f"[Resume] Missing key examples: {_preview_checkpoint_keys(missing)}")
                if unexpected:
                    print(f"[Resume] Unexpected keys (ignored): {len(unexpected)} keys")
                    print(f"[Resume] Unexpected key examples: {_preview_checkpoint_keys(unexpected)}")
                return True
            except (RuntimeError, Exception) as e:
                print(f"[Resume] ERROR: Failed to load {adapter_path}: {e}")
                print(f"[Resume] File may be corrupted. Trying earlier checkpoints...")
                return self._try_fallback_checkpoints(checkpoint_root, checkpoint_path)
        else:
            print(f"[Resume] WARNING: No adapter.pt found in {checkpoint_path}")
            return False
    
    def _try_fallback_checkpoints(self, checkpoint_root: Path, failed_path: Path) -> bool:
        """Try loading adapter.pt from earlier checkpoint subdirectories."""
        subdirs = sorted(
            [d for d in checkpoint_root.iterdir() if d.is_dir() and (d.name.startswith("epoch-") or d.name.startswith("checkpoint-"))],
            key=lambda x: (x.name.split("-")[0], int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0),
            reverse=True
        )
        for subdir in subdirs:
            if subdir == failed_path:
                continue
            adapter_path = subdir / "adapter.pt"
            if adapter_path.exists():
                try:
                    print(f"[Resume] Trying fallback: {adapter_path}")
                    state_dict = torch.load(adapter_path, map_location="cpu", weights_only=False)
                    state_dict, dropped = _filter_chess_fusion_adapter_state_dict(state_dict, self.config)
                    if dropped:
                        dropped_str = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
                        print(f"[Resume] Selective adapter loading enabled; skipped keys: {dropped_str}")
                    missing, unexpected = self.adapter.load_state_dict(state_dict, strict=False)
                    if missing:
                        print(f"[Resume] Missing keys (expected for cross-mode resume): {len(missing)} keys")
                        print(f"[Resume] Missing key examples: {_preview_checkpoint_keys(missing)}")
                    if unexpected:
                        print(f"[Resume] Unexpected keys (ignored): {len(unexpected)} keys")
                        print(f"[Resume] Unexpected key examples: {_preview_checkpoint_keys(unexpected)}")
                    print(f"[Resume] Successfully loaded adapter from fallback: {subdir.name}")
                    return True
                except (RuntimeError, Exception) as e:
                    print(f"[Resume] Fallback {subdir.name} also failed: {e}")
                    continue
        return False
    
    def print_trainable_parameters(self):
        """Print trainable parameter counts."""
        # Adapter parameters
        adapter_params = sum(p.numel() for p in self.adapter.parameters())
        
        # LoRA parameters  
        lora_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
        
        # Total LLM parameters
        total_llm = sum(p.numel() for p in self.llm.parameters())
        
        label = "LoRA" if self._use_lora else "LLM"
        print(f"\nTrainable Parameters:")
        print(f"  Adapter: {adapter_params:,}")
        print(f"  {label}: {lora_params:,}")
        print(f"  Total trainable: {adapter_params + lora_params:,}")
        print(f"  Total LLM: {total_llm:,}")
        print(f"  Trainable %: {100 * (adapter_params + lora_params) / total_llm:.2f}%")
    
    def freeze_lora(self):
        """Freeze LoRA parameters (train only adapter).
        When use_lora=False, freezes all LLM parameters."""
        if self._use_lora:
            for name, param in self.llm.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = False
            print("LoRA parameters frozen - training adapter only")
        else:
            # Full fine-tuning: freeze all LLM params
            adapter_param_ids = {id(p) for p in self.adapter.parameters()}
            for p in self.llm.parameters():
                if id(p) not in adapter_param_ids:
                    p.requires_grad = False
            print("LLM parameters frozen (no-LoRA mode) - training adapter only")
    
    def unfreeze_lora(self):
        """Unfreeze LoRA parameters (train both adapter and LoRA).
        When use_lora=False, unfreezes all LLM parameters."""
        if self._use_lora:
            for name, param in self.llm.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
            print("LoRA parameters unfrozen - training adapter + LoRA")
        else:
            # Full fine-tuning: unfreeze all LLM params
            for p in self.llm.parameters():
                p.requires_grad = True
            print("LLM parameters unfrozen (no-LoRA mode) - full fine-tuning active")
    
    def is_lora_frozen(self) -> bool:
        """Check if LoRA parameters are frozen.
        When use_lora=False, checks if LLM params are frozen."""
        if self._use_lora:
            for name, param in self.llm.named_parameters():
                if 'lora_' in name:
                    return not param.requires_grad
            return True  # No LoRA params found
        else:
            # Check if any non-adapter LLM param is trainable
            adapter_param_ids = {id(p) for p in self.adapter.parameters()}
            for p in self.llm.parameters():
                if id(p) not in adapter_param_ids and p.requires_grad:
                    return False
            return True
    
    def merge_and_reinit_lora(self):
        """
        Merge current LoRA weights into base model and reinitialize fresh LoRA.
        
        This implements "progressive LoRA" where each epoch's LoRA learns on top
        of previously merged weights, allowing cumulative learning.
        """
        if not self._use_lora:
            print("merge_and_reinit_lora: skipped (LoRA disabled)")
            return
        import torch.nn.init as init
        
        # Merge LoRA into base weights
        self.llm.merge_adapter()
        print("  Merged LoRA weights into base model")
        
        # Reinitialize LoRA weights (lora_A: Kaiming, lora_B: zeros)
        lora_a_count = 0
        lora_b_count = 0
        for name, param in self.llm.named_parameters():
            if 'lora_A' in name:
                # Kaiming uniform initialization (default PEFT behavior)
                init.kaiming_uniform_(param, a=5**0.5)
                lora_a_count += 1
            elif 'lora_B' in name:
                # Zero initialization (so LoRA starts as identity)
                init.zeros_(param)
                lora_b_count += 1
        
        # Unmerge so we can train new LoRA (keeps merged weights, LoRA now identity)
        self.llm.unmerge_adapter()
        
        print(f"  Reinitialized {lora_a_count} lora_A and {lora_b_count} lora_B matrices")


class PolicyOnlyModel(nn.Module):
    """
    Trains CSMP + Perceiver + auxiliary heads (policy, BSR, SPP)
    without loading the LLM. For efficient pretraining of the core
    chess understanding architecture.
    
    Checkpoints are saved as adapter.pt, identical to ChessCommentaryModel,
    so they can be loaded directly via resume_from_checkpoint when starting
    full chess_fusion training with the LLM.
    """
    
    def __init__(self, config: ModelConfig, torch_dtype: torch.dtype = torch.float16, **kwargs):
        super().__init__()
        self.config = config
        self.torch_dtype = torch_dtype
        self._use_lora = False  # No LLM, no LoRA
        
        # Build the adapter (same as chess_fusion mode)
        # Use default llm_dim=2048 since policy-only mode has no LLM
        self.adapter = ChessFusionAdapter(config, llm_dim=2048)
        
        # Move to GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = self.adapter.to(device)
        
        # NOTE: torch.compile is NOT applied to the adapter here.
        # CNN forward hooks create new tensor objects each pass, breaking
        # torch.compile's object-identity guards and causing repeated
        # recompilation until cache_size_limit is exhausted (same issue
        # ChessCommentaryModel avoids for chess_fusion mode).
        # There are no LLM layers to compile in policy-only mode, so
        # eager execution is appropriate.
        if config.use_torch_compile:
            print("torch.compile: skipped for policy-only mode "
                  "(CNN hooks break graph caching; no LLM layers to compile)")
        
        self._print_param_summary()
    
    def _print_param_summary(self):
        """Print parameter count summary."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n[PolicyOnlyModel] Parameters: {total:,} total, {trainable:,} trainable")
        print(f"  (No LLM loaded — pure adapter training)")
    
    def get_num_prefix_tokens(self):
        return 0  # No LLM, no prefix tokens

    def set_lm_enabled(self, enabled: bool):
        # No-op: policy_only never has an LM path.
        return

    def is_lm_enabled(self) -> bool:
        return False
    
    def forward(self, maia_features, side_to_move=None, **kwargs):
        """
        Forward pass — runs adapter and computes auxiliary losses.
        
        Args:
            maia_features: (B, 18, 8, 8) board tensors
            side_to_move: (B,) bool tensor
            
        Returns:
            namedtuple-like with .loss attribute for training loop compatibility
        """
        device = next(self.adapter.parameters()).device
        maia_features = maia_features.to(device)
        
        fusion_kwargs = {'side_to_move': side_to_move}
        if kwargs.get("maia_policy", None) is not None:
            fusion_kwargs["precomputed_policy"] = kwargs.get("maia_policy")
        adapter_out = self.adapter(maia_features, **fusion_kwargs)
        self._last_adapter_out = adapter_out  # Store for live inference inspection
        
        # Compute auxiliary losses (this IS the training objective)
        aux_losses = self.adapter.compute_auxiliary_losses(
            policy_logits=adapter_out['policy_logits'],
            eval_logits=adapter_out['eval_logits'],
            policy_targets=adapter_out['policy_targets'],
            policy_mask=kwargs.get('maia_policy_mask', None),
            eval_targets=kwargs.get('eval_targets', None),
            bsr_logits=adapter_out.get('bsr_logits', None),
            bsr_targets=adapter_out.get('bsr_targets', None),
            spp_preds=adapter_out.get('spp_preds', None),
            spp_targets=adapter_out.get('spp_targets', None),
            move_eval_logits=adapter_out.get('move_eval_logits', None),
            move_eval_mse_logits=adapter_out.get('move_eval_mse_logits', None),
            move_mate_logits=adapter_out.get('move_mate_logits', None),
            move_eval_indices=kwargs.get('move_eval_indices', None),
            move_eval_targets=kwargs.get('move_eval_targets', None),
            move_eval_mask=kwargs.get('move_eval_mask', None),
            move_ce_indices=kwargs.get('move_ce_indices', None),
            move_ce_targets=kwargs.get('move_ce_targets', None),
            move_ce_mask=kwargs.get('move_ce_mask', None),
        )
        
        # Store for logging
        self._last_aux_losses = {k: v.detach() for k, v in aux_losses.items()}
        self._fusion_entropy_metrics = adapter_out.get('entropy_metrics', {})
        
        # Return an object with .loss for compatibility with the training loop
        class _Output:
            pass
        output = _Output()
        output.loss = aux_losses['total_aux_loss']
        return output
    
    def generate(self, fen=None, side_to_move=True, **kwargs):
        """
        Run inference for live controller — returns adapter predictions 
        (no text generation since there's no LLM).
        
        Returns:
            str: Summary of policy/BSR/SPP predictions
        """
        self.eval()
        if fen is None:
            return "[PolicyOnly] No FEN provided"

        # FEN is authoritative for side-to-move; fall back to argument if parse fails.
        inferred_side_to_move = bool(side_to_move)
        legal_moves = None
        try:
            board = chess.Board(fen)
            inferred_side_to_move = bool(board.turn)
            legal_moves = {m.uci() for m in board.legal_moves}
        except Exception:
            pass
        
        device = next(self.adapter.parameters()).device
        features = extract_maia_features(fen).unsqueeze(0).to(device)
        side_tensor = torch.tensor([inferred_side_to_move], dtype=torch.bool, device=device)
        
        with torch.no_grad():
            adapter_out = self.adapter(features, side_to_move=side_tensor)
        
        self._last_adapter_out = adapter_out  # Store for live inference inspection
        
        # Build text summary of predictions
        lines = [f"[PolicyOnly Inference]"]
        try:
            from training.maia_model import get_maia_mapping, unmirror_policy_move
            mapping = get_maia_mapping()
            pol_logits = adapter_out['policy_logits'][0]
            if legal_moves is None:
                lines.append("Policy: unavailable (could not parse FEN for legality filtering)")
                return "\n".join(lines)

            def _build_legal_vocab_mask(stm_val: bool) -> torch.Tensor:
                vocab_size = int(pol_logits.numel())
                mask = torch.zeros(vocab_size, dtype=torch.bool, device=pol_logits.device)
                for idx in range(vocab_size):
                    rel_uci = mapping.decode(idx)
                    if rel_uci is None:
                        continue
                    abs_uci = unmirror_policy_move(rel_uci, stm_val)
                    if abs_uci in legal_moves:
                        mask[idx] = True
                return mask

            decode_stm = inferred_side_to_move
            legal_vocab_mask = _build_legal_vocab_mask(decode_stm)
            legal_vocab_hits = int(legal_vocab_mask.sum().item())
            if legal_vocab_hits == 0 and len(legal_moves) > 0:
                flipped_stm = not decode_stm
                flipped_mask = _build_legal_vocab_mask(flipped_stm)
                flipped_hits = int(flipped_mask.sum().item())
                if flipped_hits > legal_vocab_hits:
                    decode_stm = flipped_stm
                    legal_vocab_mask = flipped_mask
                    legal_vocab_hits = flipped_hits

            if legal_vocab_hits == 0:
                if len(legal_moves) == 0:
                    lines.append("Policy: (terminal position — no legal moves)")
                    return "\n".join(lines)
                lines.append("Policy: unavailable (no legal move found in policy distribution)")
                return "\n".join(lines)

            masked_logits = pol_logits.masked_fill(~legal_vocab_mask, float('-inf'))
            pol_probs = torch.softmax(masked_logits, dim=-1)
            top_vals, top_idxs = pol_probs.topk(min(5, legal_vocab_hits))
            moves = []
            for v, i in zip(top_vals.tolist(), top_idxs.tolist()):
                move = unmirror_policy_move(mapping.decode(i) or '?', decode_stm)
                moves.append(f"{move}: {v*100:.1f}%")
            lines.append(f"Policy: {', '.join(moves)}")
        except Exception as e:
            lines.append(f"Policy: error ({e})")
        
        return "\n".join(lines)
    
    def save_pretrained(self, output_dir: str, **kwargs):
        """Save adapter weights — compatible with ChessCommentaryModel checkpoint format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save adapter weights (same key as ChessCommentaryModel)
        adapter_state, dropped = _prepare_adapter_state_dict_for_save(self.adapter)
        torch.save(adapter_state, output_path / "adapter.pt")
        if dropped:
            print(f"[Save] Excluded {dropped} teacher_backbone checkpoint keys from adapter.pt")
        
        # Save a marker so we know this is a policy-only checkpoint
        marker = {"mode": "policy_only", "adapter_class": "ChessFusionAdapter"}
        torch.save(marker, output_path / "policy_only_meta.pt")
        
        print(f"Saved policy-only model to {output_dir}")
    
    def resume_checkpoint_weights(self, checkpoint_dir: str):
        """Load adapter weights from a prior checkpoint (policy_only or chess_fusion)."""
        checkpoint_root = Path(checkpoint_dir)
        if not checkpoint_root.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        # Smart directory resolution (same as ChessCommentaryModel)
        checkpoint_path = checkpoint_root
        subdirs = []
        if not (checkpoint_path / "adapter.pt").exists():
            print(f"[Resume] adapter.pt not found in {checkpoint_root}. Searching subdirectories...")
            subdirs = sorted(
                [d for d in checkpoint_root.iterdir() if d.is_dir() and (d.name.startswith("epoch-") or d.name.startswith("checkpoint-"))],
                key=lambda x: (x.name.split("-")[0], int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0),
                reverse=True
            )
            if subdirs:
                checkpoint_path = subdirs[0]
                print(f"[Resume] Automatically selected latest subdirectory: {checkpoint_path}")
        
        adapter_path = checkpoint_path / "adapter.pt"
        if adapter_path.exists():
            try:
                print(f"[Resume] Loading adapter weights from {adapter_path}")
                state_dict = torch.load(adapter_path, map_location="cpu", weights_only=False)
                state_dict, dropped = _filter_chess_fusion_adapter_state_dict(state_dict, self.config)
                if dropped:
                    dropped_str = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
                    print(f"[Resume] Selective adapter loading enabled; skipped keys: {dropped_str}")
                missing, unexpected = self.adapter.load_state_dict(state_dict, strict=False)
                if missing:
                    print(f"[Resume] Missing keys: {len(missing)} keys")
                    print(f"[Resume] Missing key examples: {_preview_checkpoint_keys(missing)}")
                if unexpected:
                    print(f"[Resume] Unexpected keys (ignored): {len(unexpected)} keys")
                    print(f"[Resume] Unexpected key examples: {_preview_checkpoint_keys(unexpected)}")
            except (RuntimeError, Exception) as e:
                print(f"[Resume] ERROR: Failed to load {adapter_path}: {e}")
                print(f"[Resume] File may be corrupted. Trying earlier checkpoints...")
                # Try earlier checkpoint subdirectories
                loaded = False
                for subdir in subdirs:
                    if subdir == checkpoint_path:
                        continue
                    fallback = subdir / "adapter.pt"
                    if fallback.exists():
                        try:
                            print(f"[Resume] Trying fallback: {fallback}")
                            state_dict = torch.load(fallback, map_location="cpu", weights_only=False)
                            state_dict, dropped = _filter_chess_fusion_adapter_state_dict(state_dict, self.config)
                            if dropped:
                                dropped_str = ", ".join(f"{k}={v}" for k, v in sorted(dropped.items()))
                                print(f"[Resume] Selective adapter loading enabled; skipped keys: {dropped_str}")
                            self.adapter.load_state_dict(state_dict, strict=False)
                            print(f"[Resume] Successfully loaded from fallback: {subdir.name}")
                            loaded = True
                            break
                        except Exception as e2:
                            print(f"[Resume] Fallback {subdir.name} also failed: {e2}")
                if not loaded:
                    print(f"[Resume] WARNING: Could not load adapter weights from any checkpoint")
        else:
            print(f"[Resume] WARNING: No adapter.pt found in {checkpoint_path}")
    
    def print_trainable_parameters(self):
        """Print trainable parameter counts."""
        self._print_param_summary()
    
    def freeze_lora(self):
        pass  # No LoRA
    
    def unfreeze_lora(self):
        pass  # No LoRA
    
    def is_lora_frozen(self):
        return True  # No LoRA params exist
    
    def merge_and_reinit_lora(self):
        pass  # No LoRA


class PolicyOnlyDataset(Dataset):
    """
    Dataset for policy-only training. Loads only FEN + board features.
    No text tokenization, no LLM-related processing.
    """
    
    def __init__(
        self,
        samples_dir: str,
        preload: bool = False,
        source_tag: str = "primary",
        num_eval_buckets: int = 5,
    ):
        self.samples_dir = Path(samples_dir)
        self.source_tag = source_tag
        self.num_eval_buckets = int(max(2, num_eval_buckets))
        self.sample_files = sorted(self.samples_dir.glob("*.pt"))
        print(f"Found {len(self.sample_files)} training samples (policy-only mode, source={source_tag})")
        self._print_supervision_probe()
        
        self._cache = None
        if preload:
            import gc
            from concurrent.futures import ThreadPoolExecutor, as_completed
            n_samples = len(self.sample_files)
            n_threads = min(os.cpu_count() or 4, 16, n_samples)
            print(f"  [Preload] Caching {n_samples} samples with {n_threads} threads...")
            
            cache = [None] * n_samples
            done = 0
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                futures = {pool.submit(self._load_and_prepare, i): i for i in range(n_samples)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    cache[idx] = fut.result()
                    done += 1
                    if done % 10000 == 0:
                        print(f"    ... {done}/{n_samples}")
            self._cache = cache
            gc.collect()
            print(f"  [Preload] Done. Per-step __getitem__ is now a list index.")

    def _print_supervision_probe(self, max_scan: int = 512):
        """Print supervision coverage over a small prefix of .pt samples."""
        n_total = len(self.sample_files)
        if n_total == 0:
            return
        n_scan = min(max_scan, n_total)
        has_policy = 0
        has_move_evals = 0
        has_best_moves = 0
        scanned = 0
        for fp in self.sample_files[:n_scan]:
            try:
                s = load_training_sample(fp)
            except Exception:
                continue
            scanned += 1
            if "maia_policy" in s:
                has_policy += 1
            if isinstance(s.get("move_evals", None), dict) and len(s.get("move_evals", {})) > 0:
                has_move_evals += 1
            if isinstance(s.get("stockfish_best_moves", None), list) and len(s.get("stockfish_best_moves", [])) > 0:
                has_best_moves += 1
        if scanned == 0:
            print("  [Data Probe] Could not load any .pt files for supervision probe.")
            return
        print(
            "  [Data Probe] first "
            f"{scanned}/{n_total}: maia_policy={has_policy/scanned:.1%}, "
            f"move_evals={has_move_evals/scanned:.1%}, "
            f"stockfish_best_moves={has_best_moves/scanned:.1%}"
        )
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._load_and_prepare(idx)
    
    def _load_and_prepare(self, idx):
        """Load a single sample with board features and optional aux supervision."""
        sample = load_training_sample(self.sample_files[idx])
        fen = sample.get("fen", "")
        
        # Extract side-to-move from FEN
        side_to_move = True  # Default to White
        if fen:
            fen_parts = fen.split()
            if len(fen_parts) >= 2:
                side_to_move = (fen_parts[1] == 'w')
        
        # Extract Maia board features
        maia_features = extract_maia_features(fen) if fen else torch.zeros(18, 8, 8)
        
        result = {
            "maia_features": maia_features,
            "side_to_move": side_to_move,
            "fen": fen,
            "source_tag": self.source_tag,
        }

        maia_policy = sample.get("maia_policy", None)
        if maia_policy is not None:
            if not isinstance(maia_policy, torch.Tensor):
                maia_policy = torch.tensor(maia_policy, dtype=torch.float32)
            result["maia_policy"] = maia_policy.float()

        if "stockfish_eval_cp" in sample:
            try:
                cp_val = float(sample["stockfish_eval_cp"])
                result["eval_targets"] = torch.tensor(
                    _cp_to_eval_bucket(cp_val, self.num_eval_buckets),
                    dtype=torch.long,
                )
            except Exception:
                pass

        move_evals = sample.get("move_evals", None)
        if move_evals and isinstance(move_evals, dict):
            try:
                indices, cp_values = _encode_move_evals_to_policy_vocab(
                    move_evals=move_evals,
                    side_to_move=side_to_move,
                )
                if indices:
                    result["move_eval_indices"] = torch.tensor(indices, dtype=torch.long)
                    result["move_eval_targets"] = torch.tensor(cp_values, dtype=torch.float32)

                sf_best_moves = sample.get("stockfish_best_moves", None)
                if sf_best_moves and isinstance(sf_best_moves, list):
                    ce_indices, ce_cp = _encode_named_moves_to_policy_vocab(
                        move_list=sf_best_moves,
                        move_evals=move_evals,
                        side_to_move=side_to_move,
                    )
                    if ce_indices:
                        result["move_ce_indices"] = torch.tensor(ce_indices, dtype=torch.long)
                        result["move_ce_targets"] = torch.tensor(ce_cp, dtype=torch.float32)
            except Exception:
                pass

        return result


def policy_only_collate_fn(batch):
    """Collate function for policy-only training — no text padding needed."""
    result = {
        "maia_features": torch.stack([item["maia_features"] for item in batch]),
        "side_to_move": torch.tensor([item["side_to_move"] for item in batch], dtype=torch.bool),
        "fen": [item["fen"] for item in batch],
    }
    if "source_tag" in batch[0]:
        result["source_tag"] = [item["source_tag"] for item in batch]

    # Precomputed Maia policy (optional)
    if any("maia_policy" in item for item in batch):
        template = next((item["maia_policy"] for item in batch if "maia_policy" in item), None)
        if template is not None:
            result["maia_policy"] = torch.stack([
                item["maia_policy"] if "maia_policy" in item else torch.zeros_like(template)
                for item in batch
            ])
            result["maia_policy_mask"] = torch.tensor(
                [("maia_policy" in item) for item in batch],
                dtype=torch.bool,
            )

    # Eval bucket targets (optional); -100 means ignore in cross_entropy.
    if any("eval_targets" in item for item in batch):
        result["eval_targets"] = torch.tensor(
            [int(item["eval_targets"].item()) if "eval_targets" in item else -100 for item in batch],
            dtype=torch.long,
        )

    # Per-move evaluation targets (variable length per sample -> pad to max)
    if any("move_eval_indices" in item for item in batch):
        max_moves = max(
            item["move_eval_indices"].shape[0]
            for item in batch if "move_eval_indices" in item
        ) if any("move_eval_indices" in item for item in batch) else 0
        if max_moves > 0:
            pad_indices = []
            pad_targets = []
            pad_mask = []
            for item in batch:
                if "move_eval_indices" in item:
                    n = item["move_eval_indices"].shape[0]
                    pad_len = max_moves - n
                    pad_indices.append(torch.cat([item["move_eval_indices"], torch.zeros(pad_len, dtype=torch.long)]))
                    pad_targets.append(torch.cat([item["move_eval_targets"], torch.zeros(pad_len, dtype=torch.float32)]))
                    mask = torch.zeros(max_moves, dtype=torch.bool)
                    mask[:n] = True
                    pad_mask.append(mask)
                else:
                    pad_indices.append(torch.zeros(max_moves, dtype=torch.long))
                    pad_targets.append(torch.zeros(max_moves, dtype=torch.float32))
                    pad_mask.append(torch.zeros(max_moves, dtype=torch.bool))
            result["move_eval_indices"] = torch.stack(pad_indices)
            result["move_eval_targets"] = torch.stack(pad_targets)
            result["move_eval_mask"] = torch.stack(pad_mask)

    # CE candidate moves (prefer Stockfish top-k when available)
    if any("move_ce_indices" in item for item in batch):
        max_ce = max(
            item["move_ce_indices"].shape[0]
            for item in batch if "move_ce_indices" in item
        ) if any("move_ce_indices" in item for item in batch) else 0
        if max_ce > 0:
            ce_indices = []
            ce_targets = []
            ce_mask = []
            for item in batch:
                if "move_ce_indices" in item:
                    n = item["move_ce_indices"].shape[0]
                    pad_len = max_ce - n
                    ce_indices.append(torch.cat([item["move_ce_indices"], torch.zeros(pad_len, dtype=torch.long)]))
                    ce_targets.append(torch.cat([item["move_ce_targets"], torch.zeros(pad_len, dtype=torch.float32)]))
                    m = torch.zeros(max_ce, dtype=torch.bool)
                    m[:n] = True
                    ce_mask.append(m)
                else:
                    ce_indices.append(torch.zeros(max_ce, dtype=torch.long))
                    ce_targets.append(torch.zeros(max_ce, dtype=torch.float32))
                    ce_mask.append(torch.zeros(max_ce, dtype=torch.bool))
            result["move_ce_indices"] = torch.stack(ce_indices)
            result["move_ce_targets"] = torch.stack(ce_targets)
            result["move_ce_mask"] = torch.stack(ce_mask)
    return result


class ChessCommentaryTrainingDataset(Dataset):
    """
    Training dataset that tokenizes commentary on the fly (or pre-caches everything
    in memory when ``preload=True`` for zero per-step I/O overhead).
    """
    
    def __init__(
        self,
        samples_dir: str,
        tokenizer,
        max_length: int = 512,
        prompt: str = "Provide commentary on this chess position.",
        use_engineered_features: bool = False,
        use_hybrid_features: bool = False,
        use_perceiver_features: bool = False,
        use_maia_features: bool = False,
        feature_mode: str = "simplified",
        use_perceiver_main_engineered_concat: bool = False,
        use_maia_main_engineered_concat: bool = False,
        use_chess_fusion_main_engineered_source: bool = False,
        preload: bool = False,
        use_last_move_in_prompt: bool = False,
        use_pgn_in_prompt: bool = False,
        prepend_fen_in_prompt: bool = False,
        pgn_prompt_last_n_moves: Optional[int] = None,
        reserved_prefix_tokens: int = 0,
        source_tag: str = "primary",
        num_eval_buckets: int = 5,
        training_config: Optional[TrainingConfig] = None,
    ):
        self.samples_dir = Path(samples_dir)
        self.tokenizer = tokenizer
        self.source_tag = source_tag
        self.num_eval_buckets = int(max(2, num_eval_buckets))
        self.training_config = training_config
        self.max_length = max_length
        self.prompt = prompt
        self.use_engineered_features = use_engineered_features
        self.use_hybrid_features = use_hybrid_features
        self.use_perceiver_features = use_perceiver_features
        self.use_maia_features = use_maia_features
        self.feature_mode = feature_mode
        self.use_perceiver_main_engineered_concat = use_perceiver_main_engineered_concat
        self.use_maia_main_engineered_concat = use_maia_main_engineered_concat
        self.use_chess_fusion_main_engineered_source = use_chess_fusion_main_engineered_source
        self.use_last_move_in_prompt = use_last_move_in_prompt
        self.use_pgn_in_prompt = use_pgn_in_prompt
        self.prepend_fen_in_prompt = prepend_fen_in_prompt
        self.pgn_prompt_last_n_moves = pgn_prompt_last_n_moves
        self.reserved_prefix_tokens = int(max(0, reserved_prefix_tokens))
        
        # Find all sample files
        self.sample_files = sorted(self.samples_dir.glob("*.pt"))
        print(f"Found {len(self.sample_files)} training samples")
        self._print_supervision_probe()
        if use_engineered_features:
            print("  [Engineered features] Pre-computing features during data loading")
        if use_chess_fusion_main_engineered_source:
            print("  [Structured engineered source] Pre-computing main engineered square features")
        if use_last_move_in_prompt:
            print("  [Last move] Including last move in prompt when available")
        if use_pgn_in_prompt:
            print("  [PGN context] Prepending PGN move list to prompt when available")
        if prepend_fen_in_prompt:
            print("  [FEN context] Prepending FEN string to prompt when available")
        if pgn_prompt_last_n_moves:
            print(f"  [PGN context] Limiting prompt PGN to last {pgn_prompt_last_n_moves} SAN moves")
        if self.reserved_prefix_tokens > 0:
            print(f"  [Context budget] Reserving {self.reserved_prefix_tokens} tokens for prepended embeddings")
        if training_config and getattr(training_config, 'chess_token_weight_enabled', False):
            print(f"  [Chess token weights] Enabled — squares={training_config.chess_token_weight_squares:.1f}, "
                  f"pieces={training_config.chess_token_weight_pieces:.1f}, "
                  f"moves={training_config.chess_token_weight_moves:.1f}, "
                  f"sides={training_config.chess_token_weight_sides:.1f}, "
                  f"files={training_config.chess_token_weight_files:.1f}")

        # Cache base prompt text & token length (used when last_move is not available)
        messages = [{"role": "user", "content": self.prompt}]
        self._prompt_text = _safe_apply_chat_template(
            self.tokenizer, messages, add_generation_prompt=True
        )
        self._prompt_token_len = len(
            self.tokenizer(self._prompt_text, return_tensors="pt")["input_ids"][0]
        )

        # Pre-load entire dataset into memory (eliminates disk I/O + tokenization during training)
        self._cache = None
        if preload:
            import gc
            from concurrent.futures import ThreadPoolExecutor, as_completed
            n_samples = len(self.sample_files)
            n_threads = min(os.cpu_count() or 4, 16, n_samples)
            print(f"  [Preload] Caching {n_samples} samples with {n_threads} threads...")
            self._truncation_warned = 0

            cache = [None] * n_samples
            done = 0
            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                futures = {pool.submit(self._load_and_prepare, i): i for i in range(n_samples)}
                for fut in as_completed(futures):
                    idx = futures[fut]
                    cache[idx] = fut.result()
                    done += 1
                    if done % 10000 == 0:
                        print(f"    ... {done}/{n_samples}")
            self._cache = cache
            gc.collect()
            est_mb = n_samples * 17 / 1024  # ~17 KB/sample estimate
            print(f"  [Preload] Done. ~{est_mb:.0f} MB in RAM. Per-step __getitem__ is now a list index.")

    def _print_supervision_probe(self, max_scan: int = 512):
        """Print supervision coverage over a small prefix of .pt samples."""
        n_total = len(self.sample_files)
        if n_total == 0:
            return
        n_scan = min(max_scan, n_total)
        has_policy = 0
        has_move_evals = 0
        has_best_moves = 0
        scanned = 0
        for fp in self.sample_files[:n_scan]:
            try:
                s = load_training_sample(fp)
            except Exception:
                continue
            scanned += 1
            if "maia_policy" in s:
                has_policy += 1
            if isinstance(s.get("move_evals", None), dict) and len(s.get("move_evals", {})) > 0:
                has_move_evals += 1
            if isinstance(s.get("stockfish_best_moves", None), list) and len(s.get("stockfish_best_moves", [])) > 0:
                has_best_moves += 1
        if scanned == 0:
            print("  [Data Probe] Could not load any .pt files for supervision probe.")
            return
        print(
            "  [Data Probe] first "
            f"{scanned}/{n_total}: maia_policy={has_policy/scanned:.1%}, "
            f"move_evals={has_move_evals/scanned:.1%}, "
            f"stockfish_best_moves={has_best_moves/scanned:.1%}"
        )

    def _truncate_pgn_prompt_moves(self, pgn_moves: str) -> str:
        """Keep the last N plies for prompt context while preserving move numbering."""
        n = self.pgn_prompt_last_n_moves
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

    def _split_commentary_sentences(self, commentary: str) -> list[str]:
        text = commentary.strip()
        if not text:
            return []
        parts = re.split(r"(?<=[.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _fit_commentary_to_budget(
        self,
        prompt_text: str,
        commentary: str,
        effective_text_budget: int,
        idx: int,
    ) -> str:
        prompt_token_len = len(self.tokenizer.encode(prompt_text))
        if prompt_token_len > effective_text_budget:
            raise ValueError(
                f"Sample {idx}: prompt requires {prompt_token_len} tokens but effective "
                f"text budget is {effective_text_budget} (max_length={self.max_length}, "
                f"reserved_prefix_tokens={self.reserved_prefix_tokens})."
            )

        full_text = prompt_text + commentary
        if len(self.tokenizer.encode(full_text)) <= effective_text_budget:
            return commentary

        sentences = self._split_commentary_sentences(commentary)
        trimmed_commentary = ""
        if sentences:
            for start in range(1, len(sentences) + 1):
                candidate = " ".join(sentences[start:])
                if len(self.tokenizer.encode(prompt_text + candidate)) <= effective_text_budget:
                    trimmed_commentary = candidate
                    break

        if len(self.tokenizer.encode(prompt_text + trimmed_commentary)) > effective_text_budget:
            raise ValueError(
                f"Sample {idx}: prompt-only text exceeds effective budget after commentary trimming "
                f"(budget={effective_text_budget})."
            )
        return trimmed_commentary
    
    def __len__(self):
        return len(self.sample_files)
    
    def __getitem__(self, idx):
        if self._cache is not None:
            return self._cache[idx]
        return self._load_and_prepare(idx)

    def _load_and_prepare(self, idx):
        """Load a single sample from disk, extract features, and tokenize."""
        # Load preprocessed sample
        sample = load_training_sample(self.sample_files[idx])
        
        # Convert hidden states to float32 and validate shapes (only if needed)
        lc0_hidden_states = {}
        if self.use_hybrid_features:
            raw_states = sample.get("lc0_hidden_states", {})
            if not raw_states and self.use_hybrid_features:
                 # Warn or error? Error is safer for hybrid mode
                 raise ValueError(f"Sample {self.sample_files[idx]} missing LC0 states for hybrid mode")
                 
            for k, v in raw_states.items():
                tensor = v.float()
                # Expected shape: (64, 768) - 64 squares, 768 embedding dim
                if tensor.dim() != 2 or tensor.shape[0] != 64 or tensor.shape[1] != 768:
                    raise ValueError(
                        f"Malformed lc0_hidden_states in {self.sample_files[idx]}: "
                        f"layer {k} has shape {tensor.shape}, expected (64, 768)"
                    )
                lc0_hidden_states[k] = tensor
        elif not self.use_engineered_features:
            # Legacy/Pure LC0 mode also needs states
             raw_states = sample.get("lc0_hidden_states", {})
             for k, v in raw_states.items():
                lc0_hidden_states[k] = v.float()
        
        # Extract side-to-move from FEN
        # FEN format: "pieces turn castling en_passant halfmove fullmove"
        # e.g., "rnbqkb1r/... w KQkq - 0 1" -> 'w' = White to move
        fen = sample.get("fen", "")
        side_to_move = True  # Default to White
        if fen:
            fen_parts = fen.split()
            if len(fen_parts) >= 2:
                side_to_move = (fen_parts[1] == 'w')
        
        # Pre-compute engineered features if enabled (moves CPU work out of forward pass).
        # Some consumers need the configured feature mode, while chess-fusion's
        # structured engineered source specifically needs the full "main" features.
        engineered_features = None
        main_engineered_features = None
        if fen:
            if self.use_engineered_features or self.use_hybrid_features:
                engineered_features = extract_engineered_features(fen, mode=self.feature_mode)  # (64, ENGINEERED_FEATURE_DIM)
                if self.feature_mode == "main":
                    main_engineered_features = engineered_features

            needs_main_engineered_features = (
                self.use_perceiver_main_engineered_concat
                or self.use_maia_main_engineered_concat
                or self.use_chess_fusion_main_engineered_source
            )
            if needs_main_engineered_features and main_engineered_features is None:
                main_engineered_features = extract_engineered_features(fen, mode="main")
            
        # Online extraction for Perceiver
        perceiver_sq_feats = None
        perceiver_glob_feats = None
        if self.use_perceiver_features and fen:
            if extract_perceiver_features is None:
                raise ImportError("Perceiver features requested but training.perceiver_adapter module not found.")
            sq, glob = extract_perceiver_features(fen)
            if self.use_perceiver_main_engineered_concat:
                sq = torch.cat([sq, main_engineered_features], dim=-1)
            perceiver_sq_feats = sq
            perceiver_glob_feats = glob
            
        # Online extraction for Maia
        maia_features = None
        if self.use_maia_features and fen:
            maia_features = extract_maia_features(fen)

        if main_engineered_features is not None:
            engineered_features = main_engineered_features
        
        # Build prompt text (may vary per-sample when PGN/last_move is included)
        last_move = sample.get("last_move", None)
        pgn_moves = sample.get("pgn_moves", "") if self.use_pgn_in_prompt else ""
        pgn_moves = self._truncate_pgn_prompt_moves(pgn_moves)
        needs_dynamic_prompt = (
            (self.use_last_move_in_prompt and last_move)
            or pgn_moves
            or (self.prepend_fen_in_prompt and fen)
        )

        if needs_dynamic_prompt:
            prompt_content = build_commentary_prompt(
                fen=fen,
                pgn_moves=pgn_moves,
                last_move=last_move,
                use_pgn_in_prompt=self.use_pgn_in_prompt,
                use_last_move_in_prompt=self.use_last_move_in_prompt,
                prepend_fen_in_prompt=self.prepend_fen_in_prompt,
            )
            messages = [{"role": "user", "content": prompt_content}]
            prompt_text = _safe_apply_chat_template(
                self.tokenizer, messages, add_generation_prompt=True
            )
            prompt_token_len = len(
                self.tokenizer(prompt_text, return_tensors="pt")["input_ids"][0]
            )
        else:
            prompt_text = self._prompt_text
            prompt_token_len = self._prompt_token_len
        
        effective_text_budget = self.max_length - self.reserved_prefix_tokens
        if effective_text_budget <= 0:
            raise ValueError(
                f"Invalid effective text budget: max_length={self.max_length}, "
                f"reserved_prefix_tokens={self.reserved_prefix_tokens}"
            )

        original_commentary = sample["commentary"]
        fitted_commentary = self._fit_commentary_to_budget(
            prompt_text=prompt_text,
            commentary=original_commentary,
            effective_text_budget=effective_text_budget,
            idx=idx,
        )
        if fitted_commentary != original_commentary:
            if not hasattr(self, '_truncation_warned'):
                self._truncation_warned = 0
            self._truncation_warned += 1
            if self._truncation_warned <= 5:
                print(
                    f"  [TRUNCATION] Sample {idx}: trimmed earliest commentary sentence(s) "
                    f"to preserve prompt/PGN context within {effective_text_budget} tokens"
                )
            elif self._truncation_warned == 6:
                print("  [TRUNCATION] Suppressing further warnings...")

        full_text = prompt_text + fitted_commentary
        encoding = self.tokenizer(
            full_text,
            truncation=False,
            padding=False,
            return_tensors="pt",
        )
        full_token_len = int(encoding["input_ids"].shape[1])
        if full_token_len > effective_text_budget:
            raise ValueError(
                f"Sample {idx}: prepared text has {full_token_len} tokens, exceeding "
                f"effective budget {effective_text_budget}"
            )
        
        # Create labels - mask prompt tokens with -100
        labels = encoding["input_ids"].clone()
        labels[:, :prompt_token_len] = -100  # Don't compute loss on prompt
        
        # Build per-token chess loss weights (1.0 for ordinary tokens)
        input_ids_1d = encoding["input_ids"].squeeze(0)
        if self.training_config and getattr(self.training_config, 'chess_token_weight_enabled', False):
            loss_weights = build_chess_token_loss_weights(
                input_ids_1d, self.tokenizer, self.training_config,
            )
        else:
            loss_weights = None

        result = {
            "lc0_hidden_states": lc0_hidden_states,
            "input_ids": input_ids_1d,
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
            "side_to_move": side_to_move,
            "fen": fen,
            "source_tag": self.source_tag,
        }
        if loss_weights is not None:
            result["loss_weights"] = loss_weights
        if self.use_pgn_in_prompt:
            result["pgn_moves"] = pgn_moves
        
        if engineered_features is not None:
            result["engineered_features"] = engineered_features
            
        if perceiver_sq_feats is not None:
            result["perceiver_sq_features"] = perceiver_sq_feats
            result["perceiver_glob_features"] = perceiver_glob_feats
            
        if maia_features is not None:
            result["maia_features"] = maia_features

        # Precomputed Maia policy (1880-dim logits) for teacher-free distillation
        maia_policy = sample.get("maia_policy", None)
        if maia_policy is not None:
            if not isinstance(maia_policy, torch.Tensor):
                maia_policy = torch.tensor(maia_policy, dtype=torch.float32)
            result["maia_policy"] = maia_policy.float()

        if "stockfish_eval_cp" in sample:
            try:
                cp_val = float(sample["stockfish_eval_cp"])
                result["eval_targets"] = torch.tensor(
                    _cp_to_eval_bucket(cp_val, self.num_eval_buckets),
                    dtype=torch.long,
                )
            except Exception:
                pass

        # Per-move evaluations: {uci_string: eval_cp, ...} → padded tensors
        move_evals = sample.get("move_evals", None)
        if move_evals and isinstance(move_evals, dict):
            try:
                indices, cp_values = _encode_move_evals_to_policy_vocab(
                    move_evals=move_evals,
                    side_to_move=side_to_move,
                )
                if indices:
                    result["move_eval_indices"] = torch.tensor(indices, dtype=torch.long)
                    result["move_eval_targets"] = torch.tensor(cp_values, dtype=torch.float32)

                sf_best_moves = sample.get("stockfish_best_moves", None)
                if sf_best_moves and isinstance(sf_best_moves, list):
                    ce_indices, ce_cp = _encode_named_moves_to_policy_vocab(
                        move_list=sf_best_moves,
                        move_evals=move_evals,
                        side_to_move=side_to_move,
                    )
                    if ce_indices:
                        result["move_ce_indices"] = torch.tensor(ce_indices, dtype=torch.long)
                        result["move_ce_targets"] = torch.tensor(ce_cp, dtype=torch.float32)
            except Exception:
                pass  # Skip if maia2 not available or encoding fails

        return result


def _bucket_pad_length(max_len: int, bucket_size: int = 64) -> int:
    """Round up to the nearest multiple of bucket_size for torch.compile compatibility."""
    return ((max_len + bucket_size - 1) // bucket_size) * bucket_size


def collate_fn(batch):
    """Custom collate function with dynamic padding bucketed to nearest multiple of 64."""
    # Stack LC0 hidden states
    lc0_keys = batch[0]["lc0_hidden_states"].keys()
    lc0_hidden_states = {
        key: torch.stack([item["lc0_hidden_states"][key] for item in batch])
        for key in lc0_keys
    }
    
    # Dynamic padding: find longest sequence in batch, bucket to nearest 64
    max_seq_len = max(item["input_ids"].shape[0] for item in batch)
    padded_len = _bucket_pad_length(max_seq_len)
    
    # Pad input_ids, attention_mask, and labels to bucketed length
    pad_token_id = batch[0]["input_ids"].new_zeros(1).item()  # 0 as default pad value
    padded_input_ids = []
    padded_attention_mask = []
    padded_labels = []
    for item in batch:
        seq_len = item["input_ids"].shape[0]
        pad_len = padded_len - seq_len
        if pad_len > 0:
            padded_input_ids.append(torch.cat([item["input_ids"], item["input_ids"].new_zeros(pad_len)]))
            padded_attention_mask.append(torch.cat([item["attention_mask"], item["attention_mask"].new_zeros(pad_len)]))
            padded_labels.append(torch.cat([item["labels"], item["labels"].new_full((pad_len,), -100)]))
        else:
            padded_input_ids.append(item["input_ids"])
            padded_attention_mask.append(item["attention_mask"])
            padded_labels.append(item["labels"])
    
    # Pad loss_weights if present (same padding as labels, default weight 1.0)
    padded_loss_weights = None
    if "loss_weights" in batch[0]:
        padded_loss_weights = []
        for item in batch:
            seq_len = item["loss_weights"].shape[0]
            pad_len = padded_len - seq_len
            if pad_len > 0:
                padded_loss_weights.append(torch.cat([item["loss_weights"], torch.ones(pad_len, dtype=torch.float32)]))
            else:
                padded_loss_weights.append(item["loss_weights"])

    result = {
        "lc0_hidden_states": lc0_hidden_states,
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(padded_attention_mask),
        "labels": torch.stack(padded_labels),
        "side_to_move": torch.tensor([item["side_to_move"] for item in batch], dtype=torch.bool),
        "fen": [item["fen"] for item in batch],  # List of FEN strings
    }
    if padded_loss_weights is not None:
        result["loss_weights"] = torch.stack(padded_loss_weights)
    if "pgn_moves" in batch[0]:
        result["pgn_moves"] = [item.get("pgn_moves", "") for item in batch]
    if "source_tag" in batch[0]:
        result["source_tag"] = [item["source_tag"] for item in batch]
    
    # Stack engineered features if present
    if "engineered_features" in batch[0]:
        result["engineered_features"] = torch.stack([item["engineered_features"] for item in batch])

    if "perceiver_sq_features" in batch[0]:
        sq = torch.stack([item["perceiver_sq_features"] for item in batch])
        glob = torch.stack([item["perceiver_glob_features"] for item in batch])
        result["perceiver_features"] = (sq, glob)
        
    if "maia_features" in batch[0]:
        result["maia_features"] = torch.stack([item["maia_features"] for item in batch])

    # Precomputed Maia policy (optional, for teacher-free distillation)
    if any("maia_policy" in item for item in batch):
        template = next((item["maia_policy"] for item in batch if "maia_policy" in item), None)
        if template is not None:
            result["maia_policy"] = torch.stack([
                item["maia_policy"] if "maia_policy" in item else torch.zeros_like(template)
                for item in batch
            ])
            result["maia_policy_mask"] = torch.tensor(
                [("maia_policy" in item) for item in batch],
                dtype=torch.bool,
            )

    # Eval bucket targets (optional); -100 means ignore in cross_entropy.
    if any("eval_targets" in item for item in batch):
        result["eval_targets"] = torch.tensor(
            [int(item["eval_targets"].item()) if "eval_targets" in item else -100 for item in batch],
            dtype=torch.long,
        )

    # Per-move evaluation targets (variable length per sample → pad to max)
    if any("move_eval_indices" in item for item in batch):
        max_moves = max(
            item["move_eval_indices"].shape[0]
            for item in batch if "move_eval_indices" in item
        ) if any("move_eval_indices" in item for item in batch) else 0
        if max_moves > 0:
            pad_indices = []
            pad_targets = []
            pad_mask = []
            for item in batch:
                if "move_eval_indices" in item:
                    n = item["move_eval_indices"].shape[0]
                    pad_len = max_moves - n
                    pad_indices.append(torch.cat([item["move_eval_indices"], torch.zeros(pad_len, dtype=torch.long)]))
                    pad_targets.append(torch.cat([item["move_eval_targets"], torch.zeros(pad_len, dtype=torch.float32)]))
                    mask = torch.zeros(max_moves, dtype=torch.bool)
                    mask[:n] = True
                    pad_mask.append(mask)
                else:
                    pad_indices.append(torch.zeros(max_moves, dtype=torch.long))
                    pad_targets.append(torch.zeros(max_moves, dtype=torch.float32))
                    pad_mask.append(torch.zeros(max_moves, dtype=torch.bool))
            result["move_eval_indices"] = torch.stack(pad_indices)
            result["move_eval_targets"] = torch.stack(pad_targets)
            result["move_eval_mask"] = torch.stack(pad_mask)

    # CE candidate moves (prefer Stockfish top-k when available)
    if any("move_ce_indices" in item for item in batch):
        max_ce = max(
            item["move_ce_indices"].shape[0]
            for item in batch if "move_ce_indices" in item
        ) if any("move_ce_indices" in item for item in batch) else 0
        if max_ce > 0:
            ce_indices = []
            ce_targets = []
            ce_mask = []
            for item in batch:
                if "move_ce_indices" in item:
                    n = item["move_ce_indices"].shape[0]
                    pad_len = max_ce - n
                    ce_indices.append(torch.cat([item["move_ce_indices"], torch.zeros(pad_len, dtype=torch.long)]))
                    ce_targets.append(torch.cat([item["move_ce_targets"], torch.zeros(pad_len, dtype=torch.float32)]))
                    m = torch.zeros(max_ce, dtype=torch.bool)
                    m[:n] = True
                    ce_mask.append(m)
                else:
                    ce_indices.append(torch.zeros(max_ce, dtype=torch.long))
                    ce_targets.append(torch.zeros(max_ce, dtype=torch.float32))
                    ce_mask.append(torch.zeros(max_ce, dtype=torch.bool))
            result["move_ce_indices"] = torch.stack(ce_indices)
            result["move_ce_targets"] = torch.stack(ce_targets)
            result["move_ce_mask"] = torch.stack(ce_mask)

    return result


def save_training_state(
    checkpoint_dir: Path,
    optimizer,
    scheduler,
    plateau_scheduler,
    epoch: int,
    step_in_epoch: int,
    is_epoch_end: bool,
    global_step: int,
    best_val_loss: float,
    lora_unfrozen: bool,
    wandb_run_id: Optional[str] = None,
):
    """Save training state for resume capability."""
    state = {
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "plateau_scheduler_state_dict": plateau_scheduler.state_dict(),
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "is_epoch_end": is_epoch_end,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "lora_unfrozen": lora_unfrozen,
        "wandb_run_id": wandb_run_id,
    }
    torch.save(state, checkpoint_dir / "training_state.pt")


def load_training_state(checkpoint_dir: Path):
    """Load training state for resume."""
    state_path = checkpoint_dir / "training_state.pt"
    if state_path.exists():
        return torch.load(state_path, weights_only=False)
    return None


def resolve_training_state_checkpoint_dir(checkpoint_dir: str) -> Path:
    """Resolve the checkpoint directory that contains training_state.pt."""
    checkpoint_root = Path(checkpoint_dir)
    if not checkpoint_root.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    if (checkpoint_root / "training_state.pt").exists():
        return checkpoint_root

    subdirs = sorted(
        [
            d
            for d in checkpoint_root.iterdir()
            if d.is_dir() and (d.name.startswith("epoch-") or d.name.startswith("checkpoint-"))
        ],
        key=lambda x: (x.name.split("-")[0], int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0),
        reverse=True,
    )
    for subdir in subdirs:
        if (subdir / "training_state.pt").exists():
            print(f"[Resume] Using training state from: {subdir}")
            return subdir

    print(f"[Resume] WARNING: No training_state.pt found in {checkpoint_root} or its epoch/checkpoint subdirectories")
    return checkpoint_root


def train(config: TrainingConfig):
    """
    Main training function using configuration object.
    
    Args:
        config: TrainingConfig object containing all settings
    """
    print("=" * 60)
    print("Chess Commentary Training with LoRA")
    print("=" * 60)
    
    output_path = Path(config.output_dir)
    output_path = Path(config.output_dir)
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_path.mkdir(parents=True, exist_ok=True)
    
    # DDP Setup
    is_ddp = int(os.environ.get("RANK", -1)) != -1
    global_rank = 0
    local_rank = 0
    world_size = 1
    
    if is_ddp:
        if not dist.is_initialized():
            ddp_timeout_s = int(os.environ.get("DDP_TIMEOUT_S", "600"))
            ddp_backend = os.environ.get("DDP_BACKEND", "nccl")
            dist.init_process_group(
                backend=ddp_backend,
                timeout=datetime.timedelta(seconds=ddp_timeout_s),
            )
        global_rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        print(f"Initialized DDP: Rank {global_rank}/{world_size}")

    pending_training_state = None
    pending_training_state_dir = None
    pending_wandb_run_id = None
    if config.resume_from_checkpoint and config.resume_training_state:
        pending_training_state_dir = resolve_training_state_checkpoint_dir(config.resume_from_checkpoint)
        pending_training_state = load_training_state(pending_training_state_dir)
        if pending_training_state is not None:
            pending_wandb_run_id = pending_training_state.get("wandb_run_id")

    # Initialize wandb
    if config.use_wandb and not WANDB_AVAILABLE and global_rank == 0:
        print("Warning: wandb not installed; disabling wandb logging.")
        config.use_wandb = False

    if config.use_wandb and global_rank == 0:
        # Convert config to dict for logging
        from dataclasses import asdict
        wandb_config = asdict(config)
        wandb_run_id = config.wandb_run_id or pending_wandb_run_id
        wandb_init_kwargs = {
            "project": config.wandb_project,
            "name": config.wandb_run_name,
            "config": wandb_config,
        }
        if config.wandb_resume and wandb_run_id:
            wandb_init_kwargs["id"] = wandb_run_id
            wandb_init_kwargs["resume"] = "allow"
        elif config.wandb_resume and not wandb_run_id:
            print("[Wandb] Resume requested but no run id found; starting a new run")
        
        try:
            wandb.init(**wandb_init_kwargs)
            if getattr(wandb, "run", None) is not None:
                config.wandb_run_id = wandb.run.id
            print(f"\nWandb initialized: {wandb.run.name}")
        except Exception as e:
            print(f"Warning: wandb.init failed ({e}); disabling wandb logging.")
            config.use_wandb = False
    
    # Initialize model
    print(f"\n[DEBUG] Config use_torch_compile: {config.model.use_torch_compile}")
    if config.model.use_torch_compile and hasattr(torch, "_dynamo"):
        try:
            torch._dynamo.config.capture_scalar_outputs = True
            print("[Compile] torch._dynamo.config.capture_scalar_outputs=True")
        except Exception as e:
            print(f"[Compile] Could not set capture_scalar_outputs: {e}")
        try:
            torch._dynamo.config.suppress_errors = False
            print("[Compile] torch._dynamo.config.suppress_errors=False")
        except Exception as e:
            print(f"[Compile] Could not set suppress_errors: {e}")
        try:
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            print("[Compile] torch._dynamo.config.allow_unspec_int_on_nn_module=True")
        except Exception as e:
            print(f"[Compile] Could not set allow_unspec_int_on_nn_module: {e}")
    
    print(f"\nLoading model: {config.model.base_model}")
    
    # Determine model dtype based on training precision
    model_dtype = torch.float16
    if config.bf16:
        model_dtype = torch.bfloat16
        print("Initializing model in bfloat16")
    elif config.fp16:
        model_dtype = torch.float16
        print("Initializing model in float16")
        
    is_policy_only = (config.model.mode == "policy_only")
    lm_runtime_toggle_supported = (config.model.mode == "chess_fusion")
    
    model_kwargs = {}
    if is_ddp:
        model_kwargs["device_map"] = {"": local_rank}
    
    try:
        if is_policy_only:
            model = PolicyOnlyModel(config.model, torch_dtype=model_dtype, **model_kwargs)
        else:
            model = ChessCommentaryModel(config.model, torch_dtype=model_dtype, **model_kwargs)
    except Exception as e:
        rank = os.environ.get("RANK", "-1")
        local = os.environ.get("LOCAL_RANK", "-1")
        print(f"[INIT ERROR][rank {rank}][local {local}] Model init failed: {e}")
        import traceback
        traceback.print_exc()
        raise

    # Disable KV cache during training to avoid torch.compile recompiles from
    # dynamic past_key_values initialization in decoder layers.
    if not is_policy_only and hasattr(model.llm, "config"):
        if getattr(model.llm.config, "use_cache", None) is not False:
            model.llm.config.use_cache = False
            print("[Memory] Disabled LLM KV cache for training")
    
    # Resume model weights from a prior checkpoint
    if config.resume_from_checkpoint:
        model.resume_checkpoint_weights(config.resume_from_checkpoint)
        if config.resume_training_state:
            if pending_training_state is None:
                print(
                    f"[Resume] WARNING: resume_training_state=true but no training state found at {pending_training_state_dir}. "
                    "Continuing with fresh optimizer/scheduler state."
                )

    # Backward-compat init fix: old checkpoints may contain dead recurrent text-state init
    # (zero projection + zero scale), which never receives gradients.
    if hasattr(model, "adapter") and hasattr(model.adapter, "revive_recurrent_text_state_if_stuck"):
        _ = model.adapter.revive_recurrent_text_state_if_stuck(scale_value=1.0)
    
    # Enable Gradient Checkpointing (Critical for VRAM with large batches)
    if not is_policy_only and hasattr(config, "gradient_checkpointing") and config.gradient_checkpointing:
        print("\n[Memory] Enabling Gradient Checkpointing (non-reentrant)")
        model.llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        # Required for LoRA/Adapter training with frozen base
        if model._use_lora:
            model.llm.enable_input_require_grads()
        # (no-LoRA mode already has a forward hook for input require grads)
    
    # Progressive LoRA settings
    lora_start_frozen = config.model.lora.start_frozen
    lora_unfreeze_epoch = config.model.lora.unfreeze_epoch  # 0 = manual only
    progressive_lora_merge = config.model.lora.progressive_merge
    lora_manual_override = False  # Set True when live control changes freeze state; disables auto-unfreeze
    
    llm_label = "LoRA" if model._use_lora else "LLM"
    if lora_start_frozen:
        model.freeze_lora()
        lora_unfrozen = False
        if lora_unfreeze_epoch > 0:
            print(f"\n[{llm_label}] Starting frozen, auto-unfreeze at epoch {lora_unfreeze_epoch}")
        else:
            print(f"\n[{llm_label}] Starting frozen (use live control to unfreeze)")
    else:
        model.unfreeze_lora()  # Explicitly unfreeze (critical for no-LoRA mode where __init__ freezes all params)
        lora_unfrozen = True
        print(f"\n[{llm_label}] Starting unfrozen (training {llm_label} from epoch 1)")

    # Optionally restore frozen/unfrozen status from checkpoint training state.
    if pending_training_state is not None and "lora_unfrozen" in pending_training_state:
        resume_lora_unfrozen = bool(pending_training_state.get("lora_unfrozen", lora_unfrozen))
        if resume_lora_unfrozen and not lora_unfrozen:
            model.unfreeze_lora()
            lora_unfrozen = True
            print(f"[Resume] Restored {llm_label} state: unfrozen")
        elif (not resume_lora_unfrozen) and lora_unfrozen:
            model.freeze_lora()
            lora_unfrozen = False
            print(f"[Resume] Restored {llm_label} state: frozen")

    # Print parameter summary after initial freeze/unfreeze policy is applied.
    model.print_trainable_parameters()
    
    # Determine learning rate
    learning_rate = config.learning_rate
    
    # Initialize live training controller
    control_port = config.control_port
    control_poll_steps = config.control_poll_steps
    controller = TrainingController(
        output_dir=str(output_path),
        port=control_port,
        poll_steps=control_poll_steps,
    )
    controller.init_control_file(config)
    # Set initial freeze status for dashboard
    if config.model.mode in ("chess_fusion", "policy_only"):
        maia_cfg = config.model.chess_fusion
        cnn_frozen = getattr(maia_cfg, 'freeze_cnn', True)
        transformer_frozen = getattr(maia_cfg, 'freeze_transformer', True)
        csmp_frozen = getattr(maia_cfg, 'freeze_csmp', False)
        perceiver_frozen = getattr(maia_cfg, 'freeze_perceiver', False)
        prepend_latents_frozen = getattr(maia_cfg, 'freeze_prepend_latents', False)
        lm_pseudotokens_frozen = getattr(maia_cfg, 'freeze_lm_pseudotokens', False)
    else:
        maia_cfg = config.model.maia
        cnn_frozen = getattr(maia_cfg, 'freeze_cnn', False) or getattr(maia_cfg, 'freeze_backbone', False)
        transformer_frozen = getattr(maia_cfg, 'freeze_transformer', False) or getattr(maia_cfg, 'freeze_backbone', False)
        csmp_frozen = False
        perceiver_frozen = getattr(maia_cfg, 'freeze_perceiver', False)
        prepend_latents_frozen = False
        lm_pseudotokens_frozen = False
    controller.update_status(
        lm_enabled=(False if is_policy_only else (bool(getattr(config.model, "enable_lm", True)) if lm_runtime_toggle_supported else True)),
        lora_frozen=not lora_unfrozen,
        cnn_frozen=cnn_frozen,
        transformer_frozen=transformer_frozen,
        csmp_frozen=csmp_frozen,
        perceiver_frozen=perceiver_frozen,
        prepend_latents_frozen=prepend_latents_frozen,
        lm_pseudotokens_frozen=lm_pseudotokens_frozen,
    )
    controller.start(rank=global_rank)
    
    def build_optimizer():
        """Build optimizer with differential learning rates."""
        target_model = model.module if hasattr(model, "module") else model
        # Helper lists for parameter grouping
        cnn_params = []
        transformer_params = []
        csmp_params = []
        head_params = []
        xattn_params = []
        prepend_latent_params = []
        text_gate_params = []
        pseudotoken_params = []
        
        for name, param in target_model.adapter.named_parameters():
             if not param.requires_grad:
                 continue
             
             if name.startswith("backbone."):
                 if "chess_cnn" in name:
                     cnn_params.append(param)
                 else:
                     transformer_params.append(param)
             elif name.startswith("multi_scale.chess_mp."):
                 csmp_params.append(param)
             elif name.startswith("lm_pseudotoken_layers."):
                 pseudotoken_params.append(param)
             elif name.startswith("prepend_latent_readout."):
                 prepend_latent_params.append(param)
             elif (
                 name.startswith("gated_xattns.")
                 or name.startswith("shared_readout.")
                 or name.startswith("shared_recurrent_query_gru.")
             ):
                 if ".text_gate_mlp." in name:
                     text_gate_params.append(param)
                 else:
                     xattn_params.append(param)
             else:
                 head_params.append(param)
        
        # Get ratios from the appropriate config section
        if config.model.mode in ("chess_fusion", "policy_only") and hasattr(config.model, 'chess_fusion'):
            cfg_ratios = config.model.chess_fusion
            cnn_ratio = cfg_ratios.cnn_lr_ratio
            cnn_learning_rate = getattr(cfg_ratios, 'cnn_learning_rate', None)
            trans_ratio = cfg_ratios.transformer_lr_ratio
            perceiver_ratio = cfg_ratios.perceiver_lr_ratio
            csmp_ratio = getattr(cfg_ratios, 'csmp_lr_ratio', None)
            if csmp_ratio is None:
                csmp_ratio = perceiver_ratio
            xattn_ratio = cfg_ratios.xattn_lr_ratio
            text_gate_ratio = getattr(cfg_ratios, 'text_gate_lr_ratio', None)
            if text_gate_ratio is None:
                text_gate_ratio = xattn_ratio
            pseudotoken_ratio = getattr(cfg_ratios, 'pseudotoken_lr_ratio', None)
            if pseudotoken_ratio is None:
                pseudotoken_ratio = xattn_ratio
            prepend_latent_ratio = getattr(cfg_ratios, 'prepend_latent_lr_ratio', None)
            if prepend_latent_ratio is None:
                prepend_latent_ratio = xattn_ratio
            lora_ratio = cfg_ratios.lora_lr_ratio
        elif hasattr(config.model, 'maia'):
            cnn_ratio = getattr(config.model.maia, 'cnn_lr_ratio', 0.1)
            cnn_learning_rate = getattr(config.model.maia, 'cnn_learning_rate', None)
            trans_ratio = getattr(config.model.maia, 'transformer_lr_ratio', 0.1)
            perceiver_ratio = getattr(config.model.maia, 'perceiver_lr_ratio', 1.0)
            csmp_ratio = perceiver_ratio
            xattn_ratio = 1.0
            text_gate_ratio = xattn_ratio
            pseudotoken_ratio = xattn_ratio
            prepend_latent_ratio = xattn_ratio
            lora_ratio = getattr(config.model.maia, 'lora_lr_ratio', 1.0)
        else:
            cnn_ratio = 0.1
            cnn_learning_rate = None
            trans_ratio = 0.1
            perceiver_ratio = 1.0
            csmp_ratio = 1.0
            xattn_ratio = 1.0
            text_gate_ratio = 1.0
            pseudotoken_ratio = 1.0
            prepend_latent_ratio = 1.0
            lora_ratio = 1.0

        # LoRA params — exclude adapter params that live inside LLM layers
        # (inject_into_llm places FusionDecoderLayer wrappers containing xattn
        # modules into the LLM's layer list, so llm.parameters() sees them too)
        adapter_param_ids = {id(p) for p in target_model.adapter.parameters()}
        lora_params = []
        if not target_model.is_lora_frozen():
            lora_params = [p for p in target_model.llm.parameters()
                           if p.requires_grad and id(p) not in adapter_param_ids]

        # Parameter groups
        groups = []

        # 1. CNN Backbone
        if cnn_params:
            if cnn_learning_rate is not None:
                lr = cnn_learning_rate
                lr_note = f"fixed {lr:.2e}"
            else:
                lr = learning_rate * cnn_ratio
                lr_note = f"x{cnn_ratio}"
            num = sum(p.numel() for p in cnn_params)
            print(f"\n  [Optimizer] Group: Maia CNN (LR={lr:.2e} [{lr_note}], Params={num:,.0f})")
            groups.append({"params": cnn_params, "lr": lr})

        # 2. Transformer Backbone
        if transformer_params:
            lr = learning_rate * trans_ratio
            num = sum(p.numel() for p in transformer_params)
            print(f"  [Optimizer] Group: Maia Transformer (LR={lr:.2e} [x{trans_ratio}], Params={num:,.0f})")
            groups.append({"params": transformer_params, "lr": lr})
            
        # 3. CSMP (multi-scale chess structure message passing)
        if csmp_params:
            lr = learning_rate * csmp_ratio
            num = sum(p.numel() for p in csmp_params)
            print(f"  [Optimizer] Group: CSMP (LR={lr:.2e} [x{csmp_ratio}], Params={num:,.0f})")
            groups.append({"params": csmp_params, "lr": lr})

        # 4. Adapter Head / Latents (Perceiver + pools + aux heads)
        if head_params:
            lr = learning_rate * perceiver_ratio
            num = sum(p.numel() for p in head_params)
            print(f"  [Optimizer] Group: Adapter/Head (LR={lr:.2e} [x{perceiver_ratio}], Params={num:,.0f})")
            groups.append({"params": head_params, "lr": lr})

        # 5. Gated Cross-Attention (fusion-specific)
        if xattn_params:
            lr = learning_rate * xattn_ratio
            num = sum(p.numel() for p in xattn_params)
            print(f"  [Optimizer] Group: Gated X-Attn (LR={lr:.2e} [x{xattn_ratio}], Params={num:,.0f})")
            groups.append({"params": xattn_params, "lr": lr})

        # 6. Prefix prepend-latent readout (fusion-specific)
        if prepend_latent_params:
            lr = learning_rate * prepend_latent_ratio
            num = sum(p.numel() for p in prepend_latent_params)
            print(f"  [Optimizer] Group: Prepend Latent Readout (LR={lr:.2e} [x{prepend_latent_ratio}], Params={num:,.0f})")
            groups.append({"params": prepend_latent_params, "lr": lr})

        # 7. Text Gate MLP (freshly-initialized, benefits from higher LR)
        if text_gate_params:
            lr = learning_rate * text_gate_ratio
            num = sum(p.numel() for p in text_gate_params)
            print(f"  [Optimizer] Group: Text Gate MLP (LR={lr:.2e} [x{text_gate_ratio}], Params={num:,.0f})")
            groups.append({"params": text_gate_params, "lr": lr})

        # 8. LM pseudotokens (fusion-specific style memory)
        if pseudotoken_params:
            lr = learning_rate * pseudotoken_ratio
            num = sum(p.numel() for p in pseudotoken_params)
            print(f"  [Optimizer] Group: LM Pseudotokens (LR={lr:.2e} [x{pseudotoken_ratio}], Params={num:,.0f})")
            groups.append({"params": pseudotoken_params, "lr": lr})
            
        # 9. LoRA / LLM fine-tuning params
        if lora_params:
            lr = learning_rate * lora_ratio
            num = sum(p.numel() for p in lora_params)
            label = "LoRA" if target_model._use_lora else "LLM (full ft)"
            print(f"  [Optimizer] Group: {label} (LR={lr:.2e} [x{lora_ratio}], Params={num:,.0f})")
            groups.append({"params": lora_params, "lr": lr})
            
        if config.use_8bit_optimizer:
            try:
                import bitsandbytes as bnb
                print(f"  [Optimizer] Using 8-bit AdamW (bitsandbytes)")
                return bnb.optim.AdamW8bit(groups, lr=learning_rate)
            except ImportError:
                print(f"  [Optimizer] bitsandbytes not available, falling back to standard AdamW")
                return torch.optim.AdamW(groups, lr=learning_rate)
        return torch.optim.AdamW(groups, lr=learning_rate)
    
    # Create dataset with train-val split
    print(f"\nLoading training data from: {config.samples_dir}")
    eval_bucket_count = int(getattr(config.model.chess_fusion, "num_eval_buckets", 5))
    if is_policy_only:
        primary_dataset = PolicyOnlyDataset(
            samples_dir=config.samples_dir,
            preload=config.preload_dataset,
            source_tag="primary",
            num_eval_buckets=eval_bucket_count,
        )
        active_collate_fn = policy_only_collate_fn
    else:
        primary_dataset = ChessCommentaryTrainingDataset(
            samples_dir=config.samples_dir,
            tokenizer=model.tokenizer,
            max_length=config.max_length,
            use_engineered_features=(config.model.mode == "engineered"),
            use_hybrid_features=(config.model.mode == "hybrid"),
            use_perceiver_features=(config.model.mode == "perceiver"),
            use_maia_features=(
                config.model.mode == "maia"
                or (
                    config.model.mode == "chess_fusion"
                    and not getattr(config.model.chess_fusion, "engineered_only_xattn_ablation", False)
                )
            ),
            feature_mode=getattr(config.model, 'engineered_features_type', 'sparse'),
            use_perceiver_main_engineered_concat=getattr(config.model.perceiver, "use_main_engineered_concat", False),
            use_maia_main_engineered_concat=(
                getattr(config.model.maia, "use_main_engineered_concat", False)
                or (config.model.mode == "chess_fusion" and getattr(config.model.chess_fusion, "use_engineered_concat", False))
            ),
            use_chess_fusion_main_engineered_source=(
                config.model.mode == "chess_fusion"
                and (
                    getattr(config.model.chess_fusion, "xattn_structured_use_engineered_source", False)
                    or getattr(config.model.chess_fusion, "engineered_only_xattn_ablation", False)
                )
            ),
            preload=config.preload_dataset,
            use_last_move_in_prompt=config.use_last_move_in_prompt,
            use_pgn_in_prompt=config.use_pgn_in_prompt,
            prepend_fen_in_prompt=getattr(config, "prepend_fen_in_prompt", False),
            pgn_prompt_last_n_moves=config.pgn_prompt_last_n_moves,
            reserved_prefix_tokens=model.num_prefix_tokens,
            source_tag="primary",
            num_eval_buckets=eval_bucket_count,
            training_config=config,
        )
        active_collate_fn = collate_fn
    
    # --- Secondary data mixing ---
    secondary_dir = getattr(config, 'secondary_samples_dir', None)
    if secondary_dir:
        from torch.utils.data import ConcatDataset, Subset
        secondary_mix_ratio = float(getattr(config, 'secondary_mix_ratio', 0.15))
        secondary_mix_seed = int(getattr(config, 'secondary_mix_seed', 42))
        n_secondary_target = max(1, int(len(primary_dataset) * secondary_mix_ratio))
        
        print(f"\n  [Data Mix] Secondary data from: {secondary_dir}")
        if is_policy_only:
            secondary_full = PolicyOnlyDataset(
                samples_dir=secondary_dir,
                preload=config.preload_dataset,
                source_tag="secondary",
                num_eval_buckets=eval_bucket_count,
            )
        else:
            secondary_full = ChessCommentaryTrainingDataset(
                samples_dir=secondary_dir,
                tokenizer=model.tokenizer,
                max_length=config.max_length,
                use_engineered_features=(config.model.mode == "engineered"),
                use_hybrid_features=(config.model.mode == "hybrid"),
                use_perceiver_features=(config.model.mode == "perceiver"),
                use_maia_features=(
                    config.model.mode == "maia"
                    or (
                        config.model.mode == "chess_fusion"
                        and not getattr(config.model.chess_fusion, "engineered_only_xattn_ablation", False)
                    )
                ),
                feature_mode=getattr(config.model, 'engineered_features_type', 'sparse'),
                use_perceiver_main_engineered_concat=getattr(config.model.perceiver, "use_main_engineered_concat", False),
                use_maia_main_engineered_concat=(
                    getattr(config.model.maia, "use_main_engineered_concat", False)
                    or (config.model.mode == "chess_fusion" and getattr(config.model.chess_fusion, "use_engineered_concat", False))
                ),
                use_chess_fusion_main_engineered_source=(
                    config.model.mode == "chess_fusion"
                    and (
                        getattr(config.model.chess_fusion, "xattn_structured_use_engineered_source", False)
                        or getattr(config.model.chess_fusion, "engineered_only_xattn_ablation", False)
                    )
                ),
                preload=config.preload_dataset,
                use_last_move_in_prompt=config.use_last_move_in_prompt,
                use_pgn_in_prompt=config.use_pgn_in_prompt,
                prepend_fen_in_prompt=getattr(config, "prepend_fen_in_prompt", False),
                pgn_prompt_last_n_moves=config.pgn_prompt_last_n_moves,
                reserved_prefix_tokens=model.num_prefix_tokens,
                source_tag="secondary",
                num_eval_buckets=eval_bucket_count,
                training_config=config,
            )
        
        n_available = len(secondary_full)
        n_secondary = min(n_secondary_target, n_available)
        rng = torch.Generator().manual_seed(secondary_mix_seed)
        perm = torch.randperm(n_available, generator=rng)[:n_secondary].tolist()
        secondary_subset = Subset(secondary_full, perm)
        
        full_dataset = ConcatDataset([primary_dataset, secondary_subset])
        print(f"  [Data Mix] Primary: {len(primary_dataset)}, Secondary subset: {n_secondary}/{n_available} "
              f"(ratio={secondary_mix_ratio}, seed={secondary_mix_seed})")
        print(f"  [Data Mix] Combined dataset: {len(full_dataset)} samples")
    else:
        full_dataset = primary_dataset
    
    # Split into train and validation
    from torch.utils.data import random_split
    total_size = len(full_dataset)
    val_size = int(total_size * config.val_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)  # Reproducible split
    )
    print(f"  Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Determine number of workers based on platform and preload status
    import platform
    is_linux = platform.system() == "Linux"
    dataset_preloaded = getattr(primary_dataset, '_cache', None) is not None

    configured_workers = getattr(config, 'dataloader_num_workers', None)
    configured_prefetch = max(1, int(getattr(config, 'dataloader_prefetch_factor', 4)))
    configured_persistent = getattr(config, 'dataloader_persistent_workers', None)

    if dataset_preloaded:
        # __getitem__ is a list index — worker IPC overhead > trivial lookup
        actual_num_workers = 0
    elif configured_workers is not None:
        actual_num_workers = max(0, int(configured_workers))
    else:
        actual_num_workers = min(os.cpu_count() or 4, 8) if is_linux else 0

    if actual_num_workers > 0:
        train_prefetch_factor = configured_prefetch
        persistent_workers = (
            configured_persistent
            if configured_persistent is not None
            else True
        )
    else:
        train_prefetch_factor = None
        persistent_workers = False
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_ddp else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=(train_sampler is None),
        collate_fn=active_collate_fn,
        num_workers=actual_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        prefetch_factor=train_prefetch_factor,
        persistent_workers=persistent_workers,
        sampler=train_sampler
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=active_collate_fn,
        num_workers=actual_num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        sampler=val_sampler
    )

    if actual_num_workers > 0:
        print(
            f"  DataLoader: {actual_num_workers} workers, pin_memory=True, "
            f"prefetch_factor={train_prefetch_factor}, persistent_workers={persistent_workers}"
        )
    elif dataset_preloaded:
        print(f"  DataLoader: single-threaded (preloaded — worker IPC unnecessary)")
    else:
        print(f"  DataLoader: single-threaded (Windows mode)")

    max_steps_per_epoch = config.max_steps_per_epoch
    if max_steps_per_epoch is not None and max_steps_per_epoch > 0:
        steps_per_epoch = min(len(train_dataloader), max_steps_per_epoch)
    else:
        steps_per_epoch = len(train_dataloader)

    # Calculate training steps
    total_steps = (steps_per_epoch * config.num_epochs)
    # Note: len(train_dataloader) is already adjusted by DistributedSampler (total / world_size)
    
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps * world_size
    
    if global_rank == 0:
        print(f"\nTraining configuration:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Epochs: {config.num_epochs}")
        print(f"  Batch size (per GPU): {config.batch_size}")
        print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
        print(f"  World Size: {world_size}")
        print(f"  Effective batch: {effective_batch_size}")
        print(f"  Total steps (per GPU): {total_steps}")
        if max_steps_per_epoch is not None and max_steps_per_epoch > 0:
            print(f"  Max steps per epoch: {max_steps_per_epoch}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Gradient clip: {config.gradient_clip_val}")
        if is_ddp and config.gradient_accumulation_steps > 1:
            print("  DDP grad accumulation: no_sync on non-step microbatches")
    
    # Initial optimizer (adapter only)
    optimizer = build_optimizer()
    
    # Learning rate scheduler (warmup + linear decay)
    epoch_warmup_ratio = config.epoch_warmup_ratio
    if epoch_warmup_ratio is not None and epoch_warmup_ratio > 0:
        warmup_steps = int(steps_per_epoch * epoch_warmup_ratio)
        print(f"  Using per-epoch warmup: {epoch_warmup_ratio:.1%} of epoch = {warmup_steps} steps")
    else:
        warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    
    # ReduceLROnPlateau scheduler for when val loss gets worse
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=0,  # Reduce immediately when val loss increases
    )
    best_val_loss = float('inf')
    start_epoch = 0
    resume_step_in_epoch = 0
    resume_global_step = 0

    if pending_training_state is not None:
        try:
            optimizer.load_state_dict(pending_training_state["optimizer_state_dict"])
            scheduler.load_state_dict(pending_training_state["scheduler_state_dict"])
            plateau_scheduler.load_state_dict(pending_training_state["plateau_scheduler_state_dict"])
            state_epoch = int(pending_training_state.get("epoch", -1))
            state_is_epoch_end = pending_training_state.get("is_epoch_end")
            state_step_in_epoch = int(pending_training_state.get("step_in_epoch", 0) or 0)

            if state_is_epoch_end is None:
                # Backward compatibility for older checkpoints without step metadata.
                state_dir_name = (pending_training_state_dir.name if pending_training_state_dir is not None else "")
                state_is_epoch_end = state_dir_name.startswith("epoch-")
                if not state_is_epoch_end and state_dir_name.startswith("checkpoint-") and steps_per_epoch > 0:
                    state_step_in_epoch = int(pending_training_state.get("global_step", 0)) % steps_per_epoch

            if bool(state_is_epoch_end):
                start_epoch = state_epoch + 1
                resume_step_in_epoch = 0
            else:
                start_epoch = state_epoch
                resume_step_in_epoch = state_step_in_epoch

            resume_global_step = int(pending_training_state.get("global_step", 0))
            best_val_loss = float(pending_training_state.get("best_val_loss", best_val_loss))
            if start_epoch < 0:
                start_epoch = 0
            if resume_step_in_epoch < 0:
                resume_step_in_epoch = 0
            if steps_per_epoch > 0 and resume_step_in_epoch >= steps_per_epoch:
                start_epoch += resume_step_in_epoch // steps_per_epoch
                resume_step_in_epoch = resume_step_in_epoch % steps_per_epoch
            if start_epoch >= config.num_epochs:
                print(
                    f"[Resume] WARNING: start_epoch={start_epoch} is beyond configured num_epochs={config.num_epochs}. "
                    "No further epochs will run."
                )
            state_src = pending_training_state_dir if pending_training_state_dir is not None else config.resume_from_checkpoint
            print(
                f"[Resume] Restored training state from {state_src}: "
                f"start_epoch={start_epoch}, step_in_epoch={resume_step_in_epoch}, "
                f"global_step={resume_global_step}, best_val_loss={best_val_loss:.6f}"
            )
        except KeyError as e:
            print(f"[Resume] WARNING: Missing key in training state ({e}); continuing with fresh optimizer/scheduler state")
        except Exception as e:
            print(f"[Resume] WARNING: Failed to restore training state: {e}")

    last_lr = learning_rate  # Track LR for manual logging
    
    # Mixed precision setup
    use_amp = (config.fp16 or config.bf16) and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if config.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler() if config.fp16 and torch.cuda.is_available() else None
    
    if config.bf16:
        print(f"  Using bf16 mixed precision (no GradScaler needed)")
    elif config.fp16:
        print(f"  Using fp16 mixed precision with GradScaler")
    
    # Training loop
    print("\n" + "-" * 60)
    print("Starting training...")
    print("-" * 60)
    
    # Setup device
    if is_ddp:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if is_policy_only or not (config.model.load_in_8bit or config.model.load_in_4bit):
            model.to(device)
            
    ddp_debug = os.environ.get("DEBUG_DDP", "0") == "1"
    if is_ddp and ddp_debug:
        try:
            rank = dist.get_rank()
            param_count = sum(1 for _ in model.parameters())
            trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
            total_elems = sum(p.numel() for p in model.parameters())
            print(
                f"[DDP DEBUG][rank {rank}] params={param_count}, trainable={trainable_count}, total_elems={total_elems}"
            )
            dist.barrier()
        except Exception as e:
            print(f"[DDP DEBUG][rank {local_rank}] pre-DDP debug failed: {e}")
            raise

    if is_ddp:
        # Important: find_unused_parameters=False generally faster, but needed?
        # If adapter has params not used (e.g. conditional branches), might need True.
        # But here we use most things.
        find_unused = os.environ.get("DDP_FIND_UNUSED", "0") == "1"
        static_graph = os.environ.get("DDP_STATIC_GRAPH", "0") == "1"
        model = DDP(
            model,
            device_ids=[local_rank],
            find_unused_parameters=find_unused,
            static_graph=static_graph,
        )
        if global_rank == 0 and static_graph:
            print("[DDP] static_graph=True")
    
    model.train()
    global_step = resume_global_step
    running_loss = 0.0
    running_lm_loss = 0.0
    running_policy_loss = 0.0
    running_structured_xattn_sparse_loss = 0.0
    running_structured_xattn_square_diversity_loss = 0.0
    running_structured_xattn_square_usage_entropy = 0.0
    running_structured_xattn_gate_usage_loss = 0.0
    running_structured_xattn_gate_usage_mean_abs = 0.0
    running_bsr_loss = 0.0
    running_spp_loss = 0.0
    running_move_eval_loss = 0.0
    running_move_eval_mse = 0.0
    running_move_eval_ce = 0.0
    running_move_eval_pairwise = 0.0
    running_move_eval_mate = 0.0
    running_entropy_sum = 0.0
    running_entropy_count = 0
    running_entropy_per_layer = None
    running_entropy_per_head = None
    last_grad_norm = 0.0
    
    # Track accumulated LoRA stages for progressive merge checkpoints
    lora_stage_dirs = []  # List of (stage_index, Path) for prior merged LoRA weights
    
    # Cache trainable params list (avoids rebuilding every step)
    _cached_trainable_params = None
    
    # Profiling stats
    import time
    prof_cfg = config.profiling
    _prof_cuda   = prof_cfg.enabled and prof_cfg.cuda_event_timing and torch.cuda.is_available()
    _prof_memory = prof_cfg.enabled and prof_cfg.memory_snapshots   and torch.cuda.is_available()
    _prof_nvtx = prof_cfg.enabled and getattr(prof_cfg, "nvtx_ranges", False) and torch.cuda.is_available()
    _prof_emit_interval = prof_cfg.emit_interval if prof_cfg.emit_interval > 0 else config.logging_steps
    profile_stats = {
        "data_time": 0.0,
        "forward_time": 0.0,
        "backward_time": 0.0,
        "grad_clip_time": 0.0,
        "optimizer_time": 0.0,
        "step_time": 0.0,
        # Extended profiling (populated when profiling.enabled=True)
        "mem_fwd_peak_mb": 0.0,
        "mem_bwd_peak_mb": 0.0,
        "adapter_multiscale_ms": 0.0,
        "adapter_perceiver_ms": 0.0,
        "policy_head_ms": 0.0,
        "llm_xattn_ms": 0.0,
        "llm_orig_ms": 0.0,
        "_prof_sample_count": 0,
    }
    if _prof_nvtx and global_rank == 0:
        print("[Profiler] NVTX ranges enabled (h2d, forward, backward, grad_clip, optimizer)")
    last_log_time = time.time()
    seq_length_stats = {"total_padded": 0, "sum_real": 0.0, "max_real": 0, "min_real": float("inf"), "count": 0, "n_batches": 0}
    batch_start_time = time.time()

    # Set up torch.profiler (wraps the entire epoch loop)
    _torch_prof = None
    if prof_cfg.enabled and prof_cfg.torch_profiler and global_rank == 0:
        import torch.profiler as _tprof
        import pathlib as _pathlib
        _tprof_out = str(_pathlib.Path(prof_cfg.torch_profiler_output_dir))
        _torch_prof = _tprof.profile(
            activities=[_tprof.ProfilerActivity.CPU, _tprof.ProfilerActivity.CUDA],
            schedule=_tprof.schedule(
                wait=prof_cfg.torch_profiler_wait,
                warmup=prof_cfg.torch_profiler_warmup,
                active=prof_cfg.torch_profiler_active,
                repeat=1,
            ),
            on_trace_ready=_tprof.tensorboard_trace_handler(_tprof_out),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        _torch_prof.__enter__()
        print(f"[Profiler] torch.profiler active — trace will be written to: {_tprof_out}")
    
    for epoch in range(start_epoch, config.num_epochs):
        if is_ddp:
            train_sampler.set_epoch(epoch)

        skip_batches = resume_step_in_epoch if epoch == start_epoch else 0
        if skip_batches > 0 and global_rank == 0:
            print(f"[Resume] Skipping {skip_batches} already-completed batch(es) in epoch {epoch + 1}")
            
        epoch_loss = 0.0
        num_train_batches = 0
        
        debug_dataloader = os.environ.get("DEBUG_DATALOADER", "0") == "1"
        if debug_dataloader:
            import itertools
            data_iter = iter(train_dataloader)
            first_batch_start = time.time()
            first_batch = next(data_iter)
            first_batch_time = time.time() - first_batch_start
            print(f"[DDP DEBUG][rank {global_rank}] First batch fetch: {first_batch_time:.2f}s")
            data_iter = itertools.chain([first_batch], data_iter)
        else:
            data_iter = train_dataloader
            
        progress_bar = None
        if global_rank == 0:
            progress_bar = tqdm(data_iter, total=steps_per_epoch, desc=f"Epoch {epoch+1}/{config.num_epochs}")
            data_iter = progress_bar
        
        for step, batch in enumerate(data_iter):
            if skip_batches > 0 and step < skip_batches:
                continue
            if max_steps_per_epoch is not None and max_steps_per_epoch > 0 and step >= max_steps_per_epoch:
                break
            step_timing_debug = os.environ.get("DEBUG_STEP_TIMING", "0") == "1"
            is_first_step = step == 0
            # Measure data loading time (time since last loop end)
            current_time = time.time()
            profile_stats["data_time"] += current_time - batch_start_time
            if step_timing_debug and is_first_step:
                print(f"[STEP DEBUG][rank {global_rank}] data_wait={(current_time - batch_start_time):.2f}s")
            
            # Move batch to device (non_blocking overlaps H2D transfer with CPU work when pin_memory=True)
            with _nvtx_range("h2d", _prof_nvtx):
                batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            # Track sequence length stats from attention_mask (skip for policy_only)
            if not is_policy_only:
                real_lengths = batch["attention_mask"].sum(dim=1).float()  # (B,)
                seq_length_stats["total_padded"] += batch["input_ids"].shape[1]
                seq_length_stats["sum_real"] += real_lengths.sum().item()
                seq_length_stats["max_real"] = max(seq_length_stats["max_real"], real_lengths.max().item())
                seq_length_stats["min_real"] = min(seq_length_stats["min_real"], real_lengths.min().item())
                seq_length_stats["count"] += real_lengths.numel()
                seq_length_stats["n_batches"] += 1
            
            # Toggle entropy logging only on the step that will be logged
            # (avoids need_weights=True overhead on 99% of steps)
            if config.model.mode in ("chess_fusion", "policy_only"):
                _target_m = model.module if is_ddp else model
                _perc = getattr(getattr(_target_m, 'adapter', None), 'perceiver', None)
                if _perc is not None:
                    # Log entropy on last accumulation sub-step before a logging boundary
                    _is_logging_step = (
                        (step + 1) % config.gradient_accumulation_steps == 0
                        and (global_step + 1) % config.logging_steps == 0
                    )
                    _perc.log_entropy = _is_logging_step

            # Forward pass (enable component profiling for steps 3-7 to identify bottleneck,
            # or at emit_interval when profiling is enabled)
            _target_for_profile = model.module if is_ddp else model
            _do_profile = 3 <= step <= 7 and global_rank == 0
            if prof_cfg.enabled and global_rank == 0 and global_step % _prof_emit_interval == 0:
                _do_profile = True
            if hasattr(_target_for_profile, 'adapter'):
                _target_for_profile.adapter._profile = _do_profile
            _target_for_profile._profile_forward = _do_profile

            # CUDA event timing (accurate under async GPU execution)
            if _prof_cuda:
                _ev_fwd_s = torch.cuda.Event(enable_timing=True)
                _ev_fwd_e = torch.cuda.Event(enable_timing=True)
                _ev_bwd_s = torch.cuda.Event(enable_timing=True)
                _ev_bwd_e = torch.cuda.Event(enable_timing=True)
                _ev_fwd_s.record()
            else:
                _ev_fwd_s = _ev_fwd_e = _ev_bwd_s = _ev_bwd_e = None
            accum_boundary = ((step + 1) % config.gradient_accumulation_steps == 0)
            use_no_sync = bool(is_ddp and config.gradient_accumulation_steps > 1 and not accum_boundary)
            ddp_sync_ctx = model.no_sync() if use_no_sync else nullcontext()

            t0 = time.time()
            with ddp_sync_ctx:
                if use_amp:
                    with _nvtx_range("forward", _prof_nvtx):
                        with torch.amp.autocast('cuda', dtype=amp_dtype):
                            if is_policy_only:
                                outputs = model(
                                    maia_features=batch["maia_features"],
                                    side_to_move=batch["side_to_move"],
                                    maia_policy=batch.get("maia_policy"),
                                    maia_policy_mask=batch.get("maia_policy_mask"),
                                    eval_targets=batch.get("eval_targets"),
                                    move_eval_indices=batch.get("move_eval_indices"),
                                    move_eval_targets=batch.get("move_eval_targets"),
                                    move_eval_mask=batch.get("move_eval_mask"),
                                    move_ce_indices=batch.get("move_ce_indices"),
                                    move_ce_targets=batch.get("move_ce_targets"),
                                    move_ce_mask=batch.get("move_ce_mask"),
                                )
                            else:
                                outputs = model(
                                    lc0_hidden_states=batch["lc0_hidden_states"],
                                    input_ids=batch["input_ids"],
                                    attention_mask=batch["attention_mask"],
                                    labels=batch["labels"],
                                    eval_targets=batch.get("eval_targets"),
                                    side_to_move=batch["side_to_move"],
                                    fen=batch["fen"],
                                    engineered_features=batch.get("engineered_features"),
                                    perceiver_features=batch.get("perceiver_features"),
                                    maia_features=batch.get("maia_features"),
                                    maia_policy=batch.get("maia_policy"),
                                    maia_policy_mask=batch.get("maia_policy_mask"),
                                    move_eval_indices=batch.get("move_eval_indices"),
                                    move_eval_targets=batch.get("move_eval_targets"),
                                    move_eval_mask=batch.get("move_eval_mask"),
                                    move_ce_indices=batch.get("move_ce_indices"),
                                    move_ce_targets=batch.get("move_ce_targets"),
                                    move_ce_mask=batch.get("move_ce_mask"),
                                    loss_weights=batch.get("loss_weights"),
                                )
                            loss = outputs.loss / config.gradient_accumulation_steps

                    if _prof_cuda:
                        _ev_fwd_e.record()
                        torch.cuda.synchronize()
                        forward_time = _ev_fwd_s.elapsed_time(_ev_fwd_e) / 1000.0
                        _ev_bwd_s.record()
                    else:
                        forward_time = time.time() - t0
                    profile_stats["forward_time"] += forward_time
                    if _prof_memory:
                        profile_stats["mem_fwd_peak_mb"] = max(
                            profile_stats["mem_fwd_peak_mb"],
                            torch.cuda.max_memory_allocated() / 1024**2
                        )
                        torch.cuda.reset_peak_memory_stats()
                    if step_timing_debug and is_first_step:
                        print(f"[STEP DEBUG][rank {global_rank}] forward={forward_time:.2f}s")

                    # Backward pass
                    t0 = time.time()
                    with _nvtx_range("backward", _prof_nvtx):
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            if accum_boundary:
                                scaler.unscale_(optimizer)
                        else:
                            loss.backward()

                    if _prof_cuda:
                        _ev_bwd_e.record()
                        torch.cuda.synchronize()
                        backward_time = _ev_bwd_s.elapsed_time(_ev_bwd_e) / 1000.0
                    else:
                        backward_time = time.time() - t0
                    profile_stats["backward_time"] += backward_time
                    if _prof_memory:
                        profile_stats["mem_bwd_peak_mb"] = max(
                            profile_stats["mem_bwd_peak_mb"],
                            torch.cuda.max_memory_allocated() / 1024**2
                        )
                        torch.cuda.reset_peak_memory_stats()
                    if step_timing_debug and is_first_step:
                        print(f"[STEP DEBUG][rank {global_rank}] backward={backward_time:.2f}s")
                else:
                    with _nvtx_range("forward", _prof_nvtx):
                        if is_policy_only:
                            outputs = model(
                                maia_features=batch["maia_features"],
                                side_to_move=batch["side_to_move"],
                                maia_policy=batch.get("maia_policy"),
                                maia_policy_mask=batch.get("maia_policy_mask"),
                                eval_targets=batch.get("eval_targets"),
                                move_eval_indices=batch.get("move_eval_indices"),
                                move_eval_targets=batch.get("move_eval_targets"),
                                move_eval_mask=batch.get("move_eval_mask"),
                                move_ce_indices=batch.get("move_ce_indices"),
                                move_ce_targets=batch.get("move_ce_targets"),
                                move_ce_mask=batch.get("move_ce_mask"),
                            )
                        else:
                            outputs = model(
                                lc0_hidden_states=batch["lc0_hidden_states"],
                                input_ids=batch["input_ids"],
                                attention_mask=batch["attention_mask"],
                                labels=batch["labels"],
                                eval_targets=batch.get("eval_targets"),
                                side_to_move=batch["side_to_move"],
                                fen=batch["fen"],
                                engineered_features=batch.get("engineered_features"),
                                perceiver_features=batch.get("perceiver_features"),
                                maia_features=batch.get("maia_features"),
                                maia_policy=batch.get("maia_policy"),
                                maia_policy_mask=batch.get("maia_policy_mask"),
                                move_eval_indices=batch.get("move_eval_indices"),
                                move_eval_targets=batch.get("move_eval_targets"),
                                move_eval_mask=batch.get("move_eval_mask"),
                                move_ce_indices=batch.get("move_ce_indices"),
                                move_ce_targets=batch.get("move_ce_targets"),
                                move_ce_mask=batch.get("move_ce_mask"),
                                loss_weights=batch.get("loss_weights"),
                            )
                        loss = outputs.loss / config.gradient_accumulation_steps

                    if _prof_cuda:
                        _ev_fwd_e.record()
                        torch.cuda.synchronize()
                        forward_time = _ev_fwd_s.elapsed_time(_ev_fwd_e) / 1000.0
                        _ev_bwd_s.record()
                    else:
                        forward_time = time.time() - t0
                    profile_stats["forward_time"] += forward_time
                    if _prof_memory:
                        profile_stats["mem_fwd_peak_mb"] = max(
                            profile_stats["mem_fwd_peak_mb"],
                            torch.cuda.max_memory_allocated() / 1024**2
                        )
                        torch.cuda.reset_peak_memory_stats()
                    if step_timing_debug and is_first_step:
                        print(f"[STEP DEBUG][rank {global_rank}] forward={forward_time:.2f}s")

                    # Backward pass
                    t0 = time.time()
                    with _nvtx_range("backward", _prof_nvtx):
                        loss.backward()

                    if _prof_cuda:
                        _ev_bwd_e.record()
                        torch.cuda.synchronize()
                        backward_time = _ev_bwd_s.elapsed_time(_ev_bwd_e) / 1000.0
                    else:
                        backward_time = time.time() - t0
                    profile_stats["backward_time"] += backward_time
                    if _prof_memory:
                        profile_stats["mem_bwd_peak_mb"] = max(
                            profile_stats["mem_bwd_peak_mb"],
                            torch.cuda.max_memory_allocated() / 1024**2
                        )
                        torch.cuda.reset_peak_memory_stats()
                    if step_timing_debug and is_first_step:
                        print(f"[STEP DEBUG][rank {global_rank}] backward={backward_time:.2f}s")

            # Gradient clipping should happen once per optimizer step (after accumulation)
            if config.gradient_clip_val is not None and accum_boundary:
                t0 = time.time()
                with _nvtx_range("grad_clip", _prof_nvtx):
                    if _cached_trainable_params is None:
                        _cached_trainable_params = [p for p in model.parameters() if p.requires_grad]
                    trainable_params = [p for p in _cached_trainable_params if p.grad is not None]
                    total_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.gradient_clip_val)
                last_grad_norm = total_norm.item()
                grad_clip_time = time.time() - t0
                profile_stats["grad_clip_time"] += grad_clip_time
                if step_timing_debug and is_first_step:
                    print(f"[STEP DEBUG][rank {global_rank}] grad_clip={grad_clip_time:.2f}s")
            
            running_loss += loss.item()
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            num_train_batches += 1

            # Capture adapter/xAttn profiling data into profile_stats (when enabled)
            if _do_profile and prof_cfg.enabled and config.model.mode in ("chess_fusion", "policy_only"):
                _prof_tgt = model.module if is_ddp else model
                _prof_adapter = getattr(_prof_tgt, 'adapter', None)
                if _prof_adapter is not None and hasattr(_prof_adapter, '_last_profile_ms'):
                    _pm = _prof_adapter._last_profile_ms
                    profile_stats["adapter_multiscale_ms"] += _pm.get("multi_scale", 0.0)
                    profile_stats["adapter_perceiver_ms"]  += _pm.get("perceiver", 0.0)
                    profile_stats["policy_head_ms"] += _pm.get("policy_head", 0.0)
                    profile_stats["bsr_spp_ms"]             = profile_stats.get("bsr_spp_ms", 0.0) + _pm.get("bsr_spp", 0.0)
                    if "csmp" not in profile_stats:
                        profile_stats["csmp_ms"] = 0.0
                    profile_stats["csmp_ms"] = profile_stats.get("csmp_ms", 0.0) + _pm.get("csmp", 0.0)
                try:
                    from training.chess_fusion_model import FusionDecoderLayer as _FDL
                    _xattn_list = _FDL._profile_xattn_ms
                    _orig_list  = _FDL._profile_original_ms
                    if isinstance(_xattn_list, list) and _xattn_list:
                        profile_stats["llm_xattn_ms"] += sum(_xattn_list)
                        profile_stats["llm_orig_ms"]  += sum(_orig_list)
                    elif isinstance(_xattn_list, float):
                        profile_stats["llm_xattn_ms"] += _xattn_list
                        profile_stats["llm_orig_ms"]  += _orig_list
                except Exception:
                    pass
                profile_stats["_prof_sample_count"] += 1

            # Track per-component losses for chess_fusion / policy_only
            target_model = model.module if hasattr(model, "module") else model
            if config.model.mode in ("chess_fusion", "policy_only") and hasattr(target_model, '_last_aux_losses'):
                aux = target_model._last_aux_losses
                ga = config.gradient_accumulation_steps
                running_policy_loss += aux['policy_loss'].item() / ga
                running_structured_xattn_sparse_loss += aux.get('structured_xattn_sparse_loss', torch.tensor(0.0)).item() / ga
                running_structured_xattn_square_diversity_loss += aux.get('structured_xattn_square_diversity_loss', torch.tensor(0.0)).item() / ga
                running_structured_xattn_square_usage_entropy += aux.get('structured_xattn_square_usage_entropy', torch.tensor(0.0)).item() / ga
                running_structured_xattn_gate_usage_loss += aux.get('structured_xattn_gate_usage_loss', torch.tensor(0.0)).item() / ga
                running_structured_xattn_gate_usage_mean_abs += aux.get('structured_xattn_gate_usage_mean_abs', torch.tensor(0.0)).item() / ga
                running_bsr_loss += aux.get('bsr_loss', torch.tensor(0.0)).item() / ga
                running_spp_loss += aux.get('spp_loss', torch.tensor(0.0)).item() / ga
                running_move_eval_loss += aux.get('move_eval_loss', torch.tensor(0.0)).item() / ga
                running_move_eval_mse += aux.get('move_eval_mse', torch.tensor(0.0)).item() / ga
                running_move_eval_ce += aux.get('move_eval_ce', torch.tensor(0.0)).item() / ga
                running_move_eval_pairwise += aux.get('move_eval_pairwise', torch.tensor(0.0)).item() / ga
                running_move_eval_mate += aux.get('move_eval_mate', torch.tensor(0.0)).item() / ga
                lm_active_now = (not is_policy_only) and bool(getattr(target_model, 'is_lm_enabled', lambda: True)())
                if not lm_active_now:
                    running_lm_loss += 0.0  # LM disabled or policy-only mode
                else:
                    # outputs.loss is unscaled total (LM + weighted aux); `loss` is scaled for grad accumulation.
                    # Log LM in unscaled units to match aux component logs.
                    running_lm_loss += (outputs.loss.item() - aux['total_aux_loss'].item()) / ga
                # Accumulate entropy metrics
                entropy = getattr(target_model, '_fusion_entropy_metrics', {})
                if 'cross_attn_entropy_mean' in entropy:
                    running_entropy_sum += entropy['cross_attn_entropy_mean'].item()
                    running_entropy_count += 1
                    pl = entropy.get('cross_attn_entropy_per_layer')
                    ph = entropy.get('cross_attn_entropy_per_head')
                    if pl is not None:
                        if running_entropy_per_layer is None:
                            running_entropy_per_layer = pl.cpu().clone()
                        else:
                            running_entropy_per_layer += pl.cpu()
                    if ph is not None:
                        if running_entropy_per_head is None:
                            running_entropy_per_head = ph.cpu().clone()
                        else:
                            running_entropy_per_head += ph.cpu()
            
            # Gradient accumulation
            if accum_boundary:
                # Optimizer step
                t0 = time.time()
                with _nvtx_range("optimizer", _prof_nvtx):
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                    else:
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                scheduler.step()
                optimizer_time = time.time() - t0
                profile_stats["optimizer_time"] += optimizer_time

                # Advance torch.profiler schedule (no-op when profiler is inactive)
                if _torch_prof is not None:
                    _torch_prof.step()

                # Periodic CUDA cache cleanup to combat memory fragmentation
                # from dynamic padding (different sequence lengths per batch)
                if global_step % 500 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                if step_timing_debug and is_first_step:
                    print(f"[STEP DEBUG][rank {global_rank}] optimizer={optimizer_time:.2f}s")
                    if is_ddp:
                        sync_start = time.time()
                        dist.barrier()
                        print(f"[STEP DEBUG][rank {global_rank}] ddp_barrier={(time.time() - sync_start):.2f}s")
                
                global_step += 1
                
                # ── Live Control Polling ──────────────────────────
                if global_step % control_poll_steps == 0 and global_rank == 0:
                    changes = controller.poll()
                    if changes:
                        need_rebuild = False
                        target_model = model.module if is_ddp else model
                        
                        # Apply base LR change
                        if "base_learning_rate" in changes:
                            new_base = changes["base_learning_rate"]
                            print(f"\n[Live Control] Base LR: {learning_rate:.2e} -> {new_base:.2e}")
                            learning_rate = new_base
                            need_rebuild = True
                            # Clear from file so it doesn't re-apply
                            controller.update_status(active_base_lr=learning_rate)
                        
                        # Apply LR ratio changes
                        if "lr_ratios" in changes:
                            ratios = changes["lr_ratios"]
                            if config.model.mode == "chess_fusion":
                                lr_cfg = config.model.chess_fusion
                            else:
                                lr_cfg = config.model.maia
                            changed = []
                            for key in ["cnn_lr_ratio", "transformer_lr_ratio", "perceiver_lr_ratio", "csmp_lr_ratio", "xattn_lr_ratio", "text_gate_lr_ratio", "pseudotoken_lr_ratio", "prepend_latent_lr_ratio", "lora_lr_ratio"]:
                                new_val = ratios.get(key)
                                if new_val is not None and hasattr(lr_cfg, key) and new_val != getattr(lr_cfg, key):
                                    old_val = getattr(lr_cfg, key)
                                    setattr(lr_cfg, key, new_val)
                                    changed.append(f"{key}: {old_val} -> {new_val}")
                            if changed:
                                print(f"\n[Live Control] LR ratios updated: {', '.join(changed)}")
                                need_rebuild = True
                        
                        # Apply aux policy weight change
                        if "aux_policy_weight" in changes:
                            new_pw = changes["aux_policy_weight"]
                            if config.model.mode in ("chess_fusion", "policy_only"):
                                pw_cfg = config.model.chess_fusion
                            else:
                                pw_cfg = config.model.maia
                            old_pw = getattr(pw_cfg, 'aux_policy_weight', 0.1)
                            if new_pw != old_pw:
                                pw_cfg.aux_policy_weight = new_pw
                                print(f"\n[Live Control] Policy weight: {old_pw} -> {new_pw}")
                                controller.update_status(
                                    active_aux_policy_weight=new_pw,
                                    last_command_applied="set_policy_weight",
                                )
                        if "structured_xattn_sparse_weight" in changes:
                            new_sxw = changes["structured_xattn_sparse_weight"]
                            sxw_cfg = config.model.chess_fusion if config.model.mode in ("chess_fusion", "policy_only") else config.model.maia
                            old_sxw = getattr(sxw_cfg, 'structured_xattn_sparse_weight', 0.0)
                            if new_sxw != old_sxw:
                                sxw_cfg.structured_xattn_sparse_weight = new_sxw
                                print(f"\n[Live Control] Structured x-attn sparse weight: {old_sxw} -> {new_sxw}")
                                controller.update_status(
                                    active_structured_xattn_sparse_weight=new_sxw,
                                    last_command_applied="set_structured_xattn_sparse_weight",
                                )
                                if config.use_wandb:
                                    wandb.log({
                                        "live_control/structured_xattn_sparse_weight": new_sxw,
                                    }, step=global_step)
                        if "structured_xattn_square_diversity_weight" in changes:
                            new_sxdw = changes["structured_xattn_square_diversity_weight"]
                            sxdw_cfg = config.model.chess_fusion if config.model.mode in ("chess_fusion", "policy_only") else config.model.maia
                            old_sxdw = getattr(sxdw_cfg, 'structured_xattn_square_diversity_weight', 0.0)
                            if new_sxdw != old_sxdw:
                                sxdw_cfg.structured_xattn_square_diversity_weight = new_sxdw
                                print(f"\n[Live Control] Structured x-attn square diversity weight: {old_sxdw} -> {new_sxdw}")
                                controller.update_status(
                                    active_structured_xattn_square_diversity_weight=new_sxdw,
                                    last_command_applied="set_structured_xattn_square_diversity_weight",
                                )
                                if config.use_wandb:
                                    wandb.log({
                                        "live_control/structured_xattn_square_diversity_weight": new_sxdw,
                                    }, step=global_step)
                        if "structured_xattn_square_diversity_target_entropy" in changes:
                            new_sxdt = float(changes["structured_xattn_square_diversity_target_entropy"])
                            new_sxdt = max(0.0, min(1.0, new_sxdt))
                            sxdt_cfg = config.model.chess_fusion if config.model.mode in ("chess_fusion", "policy_only") else config.model.maia
                            old_sxdt = getattr(sxdt_cfg, 'structured_xattn_square_diversity_target_entropy', 0.5)
                            if new_sxdt != old_sxdt:
                                sxdt_cfg.structured_xattn_square_diversity_target_entropy = new_sxdt
                                print(f"\n[Live Control] Structured x-attn square diversity target entropy: {old_sxdt} -> {new_sxdt}")
                                controller.update_status(
                                    active_structured_xattn_square_diversity_target_entropy=new_sxdt,
                                    last_command_applied="set_structured_xattn_square_diversity_target_entropy",
                                )
                                if config.use_wandb:
                                    wandb.log({
                                        "live_control/structured_xattn_square_diversity_target_entropy": new_sxdt,
                                    }, step=global_step)
                        if "structured_xattn_gate_usage_weight" in changes:
                            new_sxguw = changes["structured_xattn_gate_usage_weight"]
                            sxguw_cfg = config.model.chess_fusion if config.model.mode in ("chess_fusion", "policy_only") else config.model.maia
                            old_sxguw = getattr(sxguw_cfg, 'structured_xattn_gate_usage_weight', 0.0)
                            if new_sxguw != old_sxguw:
                                sxguw_cfg.structured_xattn_gate_usage_weight = new_sxguw
                                print(f"\n[Live Control] Structured x-attn gate usage weight: {old_sxguw} -> {new_sxguw}")
                                controller.update_status(
                                    active_structured_xattn_gate_usage_weight=new_sxguw,
                                    last_command_applied="set_structured_xattn_gate_usage_weight",
                                )
                                if config.use_wandb:
                                    wandb.log({
                                        "live_control/structured_xattn_gate_usage_weight": new_sxguw,
                                    }, step=global_step)
                        if "structured_xattn_gate_usage_target" in changes:
                            new_sxgut = float(changes["structured_xattn_gate_usage_target"])
                            new_sxgut = max(0.0, min(1.0, new_sxgut))
                            sxgut_cfg = config.model.chess_fusion if config.model.mode in ("chess_fusion", "policy_only") else config.model.maia
                            old_sxgut = getattr(sxgut_cfg, 'structured_xattn_gate_usage_target', 0.1)
                            if new_sxgut != old_sxgut:
                                sxgut_cfg.structured_xattn_gate_usage_target = new_sxgut
                                print(f"\n[Live Control] Structured x-attn gate usage target: {old_sxgut} -> {new_sxgut}")
                                controller.update_status(
                                    active_structured_xattn_gate_usage_target=new_sxgut,
                                    last_command_applied="set_structured_xattn_gate_usage_target",
                                )
                                if config.use_wandb:
                                    wandb.log({
                                        "live_control/structured_xattn_gate_usage_target": new_sxgut,
                                    }, step=global_step)

                        # Apply move-eval objective weights
                        me_cfg = config.model.chess_fusion if config.model.mode in ("chess_fusion", "policy_only") else config.model.maia
                        if "aux_move_eval_weight" in changes:
                            new_mew = changes["aux_move_eval_weight"]
                            old_mew = getattr(me_cfg, 'aux_move_eval_weight', 0.0)
                            if new_mew != old_mew:
                                me_cfg.aux_move_eval_weight = new_mew
                                print(f"\n[Live Control] Move-eval weight: {old_mew} -> {new_mew}")
                                controller.update_status(
                                    active_aux_move_eval_weight=new_mew,
                                    last_command_applied="set_move_eval_weight",
                                )
                        if "move_eval_mse_weight" in changes:
                            new_msew = changes["move_eval_mse_weight"]
                            old_msew = getattr(me_cfg, 'move_eval_mse_weight', 0.5)
                            if new_msew != old_msew:
                                me_cfg.move_eval_mse_weight = new_msew
                                print(f"\n[Live Control] Move-eval MSE weight: {old_msew} -> {new_msew}")
                                controller.update_status(
                                    active_move_eval_mse_weight=new_msew,
                                    last_command_applied="set_move_eval_mse_weight",
                                )
                        if "move_eval_ce_weight" in changes:
                            new_cew = changes["move_eval_ce_weight"]
                            old_cew = getattr(me_cfg, 'move_eval_ce_weight', 0.5)
                            if new_cew != old_cew:
                                me_cfg.move_eval_ce_weight = new_cew
                                print(f"\n[Live Control] Move-eval CE weight: {old_cew} -> {new_cew}")
                                controller.update_status(
                                    active_move_eval_ce_weight=new_cew,
                                    last_command_applied="set_move_eval_ce_weight",
                                )
                        if "move_eval_pairwise_weight" in changes:
                            new_pww = changes["move_eval_pairwise_weight"]
                            old_pww = getattr(me_cfg, 'move_eval_pairwise_weight', 0.0)
                            if new_pww != old_pww:
                                me_cfg.move_eval_pairwise_weight = new_pww
                                print(f"\n[Live Control] Move-eval pairwise weight: {old_pww} -> {new_pww}")
                                controller.update_status(
                                    active_move_eval_pairwise_weight=new_pww,
                                    last_command_applied="set_move_eval_pairwise_weight",
                                )
                        
                        # Apply BSR weight change
                        if "bsr_weight" in changes:
                            new_bw = changes["bsr_weight"]
                            bw_cfg = config.model.chess_fusion if config.model.mode in ("chess_fusion", "policy_only") else config.model.maia
                            old_bw = getattr(bw_cfg, 'bsr_weight', 0.0)
                            if new_bw != old_bw:
                                bw_cfg.bsr_weight = new_bw
                                print(f"\n[Live Control] BSR weight: {old_bw} -> {new_bw}")
                                controller.update_status(
                                    active_bsr_weight=new_bw,
                                    last_command_applied="set_bsr_weight",
                                )
                        
                        # Apply SPP weight change
                        if "spp_weight" in changes:
                            new_sw = changes["spp_weight"]
                            sw_cfg = config.model.chess_fusion if config.model.mode in ("chess_fusion", "policy_only") else config.model.maia
                            old_sw = getattr(sw_cfg, 'spp_weight', 0.0)
                            if new_sw != old_sw:
                                sw_cfg.spp_weight = new_sw
                                print(f"\n[Live Control] SPP weight: {old_sw} -> {new_sw}")
                                controller.update_status(
                                    active_spp_weight=new_sw,
                                    last_command_applied="set_spp_weight",
                                )

                        # Apply x-attn gate override (tanh-space value in (-1, 1))
                        if "xattn_gate_tanh_value" in changes:
                            requested_gate = float(changes["xattn_gate_tanh_value"])
                            requested_gate = max(min(requested_gate, 0.999), -0.999)
                            gate_raw = torch.atanh(torch.tensor(requested_gate)).item()
                            applied_vals = []
                            with torch.no_grad():
                                for xattn in target_model.adapter.gated_xattns:
                                    xattn.gate.fill_(gate_raw)
                                    applied_vals.append(torch.tanh(xattn.gate).mean().item())
                            applied_mean = float(sum(applied_vals) / max(1, len(applied_vals)))
                            print(
                                f"\n[Live Control] X-Attn gate set (tanh-space): requested={requested_gate:.4f}, "
                                f"applied_mean={applied_mean:.4f}"
                            )
                            controller.update_status(
                                active_xattn_gate_tanh_mean=applied_mean,
                                last_command_applied="set_xattn_gate_tanh_value",
                            )
                            if config.use_wandb:
                                wandb.log({
                                    "live_control/xattn_gate_tanh_requested": requested_gate,
                                    "live_control/xattn_gate_tanh_applied_mean": applied_mean,
                                }, step=global_step)

                        # Apply FFN gate override (tanh-space value in (-1, 1))
                        if "ffn_gate_tanh_value" in changes:
                            requested_gate = float(changes["ffn_gate_tanh_value"])
                            requested_gate = max(min(requested_gate, 0.999), -0.999)
                            gate_raw = torch.atanh(torch.tensor(requested_gate)).item()
                            applied_vals = []
                            with torch.no_grad():
                                for xattn in target_model.adapter.gated_xattns:
                                    xattn.ffn_gate.fill_(gate_raw)
                                    applied_vals.append(torch.tanh(xattn.ffn_gate).item())
                            applied_mean = float(sum(applied_vals) / max(1, len(applied_vals)))
                            print(
                                f"\n[Live Control] FFN gate set (tanh-space): requested={requested_gate:.4f}, "
                                f"applied_mean={applied_mean:.4f}"
                            )
                            controller.update_status(
                                active_ffn_gate_tanh_mean=applied_mean,
                                last_command_applied="set_ffn_gate_tanh_value",
                            )
                            if config.use_wandb:
                                wandb.log({
                                    "live_control/ffn_gate_tanh_requested": requested_gate,
                                    "live_control/ffn_gate_tanh_applied_mean": applied_mean,
                                }, step=global_step)
                        
                        # Process commands
                        cmds = changes.get("commands", {})

                        if cmds.get("enable_lm"):
                            if not lm_runtime_toggle_supported:
                                mode_name = "policy_only" if is_policy_only else config.model.mode
                                print(f"\n[Live Control] enable_lm ignored in {mode_name} mode (runtime LM toggle is chess_fusion-only)")
                            else:
                                config.model.enable_lm = True
                                target_model.set_lm_enabled(True)
                                print("\n[Live Control] LM objective/generation enabled")
                                controller.update_status(lm_enabled=True, last_command_applied="enable_lm")
                                if config.use_wandb:
                                    wandb.log({"live_control/lm_enabled": 1}, step=global_step)

                        if cmds.get("disable_lm"):
                            if not lm_runtime_toggle_supported:
                                mode_name = "policy_only" if is_policy_only else config.model.mode
                                print(f"\n[Live Control] disable_lm ignored in {mode_name} mode (runtime LM toggle is chess_fusion-only)")
                            else:
                                config.model.enable_lm = False
                                target_model.set_lm_enabled(False)
                                print("\n[Live Control] LM objective/generation disabled (adapter-only training)")
                                controller.update_status(lm_enabled=False, last_command_applied="disable_lm")
                                if config.use_wandb:
                                    wandb.log({"live_control/lm_enabled": 0}, step=global_step)
                        
                        if cmds.get("freeze_lora"):
                            print(f"\n[Live Control] Freezing LoRA")
                            target_model.freeze_lora()
                            lora_unfrozen = False
                            lora_manual_override = True
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(lora_frozen=True, last_command_applied="freeze_lora")
                        
                        if cmds.get("unfreeze_lora"):
                            print(f"\n[Live Control] Unfreezing LoRA")
                            target_model.unfreeze_lora()
                            lora_unfrozen = True
                            lora_manual_override = True
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(lora_frozen=False, last_command_applied="unfreeze_lora")
                        
                        if cmds.get("freeze_cnn"):
                            print(f"\n[Live Control] Freezing CNN backbone")
                            target_model.adapter._freeze_cnn_params()
                            target_model.adapter.freeze_cnn = True
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(cnn_frozen=True, last_command_applied="freeze_cnn")
                        
                        if cmds.get("unfreeze_cnn"):
                            print(f"\n[Live Control] Unfreezing CNN backbone")
                            target_model.adapter._unfreeze_cnn_params()
                            target_model.adapter.freeze_cnn = False
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(cnn_frozen=False, last_command_applied="unfreeze_cnn")
                        
                        if cmds.get("freeze_transformer"):
                            print(f"\n[Live Control] Freezing Transformer backbone")
                            target_model.adapter._freeze_transformer_params()
                            target_model.adapter.freeze_transformer = True
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(transformer_frozen=True, last_command_applied="freeze_transformer")
                        
                        if cmds.get("unfreeze_transformer"):
                            print(f"\n[Live Control] Unfreezing Transformer backbone")
                            target_model.adapter._unfreeze_transformer_params()
                            target_model.adapter.freeze_transformer = False
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(transformer_frozen=False, last_command_applied="unfreeze_transformer")

                        if cmds.get("freeze_csmp") and hasattr(target_model.adapter, '_freeze_csmp_params'):
                            print(f"\n[Live Control] Freezing CSMP")
                            target_model.adapter._freeze_csmp_params()
                            target_model.adapter.freeze_csmp = True
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(csmp_frozen=True, last_command_applied="freeze_csmp")

                        if cmds.get("unfreeze_csmp") and hasattr(target_model.adapter, '_unfreeze_csmp_params'):
                            print(f"\n[Live Control] Unfreezing CSMP")
                            target_model.adapter._unfreeze_csmp_params()
                            target_model.adapter.freeze_csmp = False
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(csmp_frozen=False, last_command_applied="unfreeze_csmp")

                        if cmds.get("freeze_perceiver") and hasattr(target_model.adapter, '_freeze_perceiver_params'):
                            print(f"\n[Live Control] Freezing Perceiver")
                            target_model.adapter._freeze_perceiver_params()
                            target_model.adapter.freeze_perceiver = True
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(perceiver_frozen=True, last_command_applied="freeze_perceiver")

                        if cmds.get("unfreeze_perceiver") and hasattr(target_model.adapter, '_unfreeze_perceiver_params'):
                            print(f"\n[Live Control] Unfreezing Perceiver")
                            target_model.adapter._unfreeze_perceiver_params()
                            target_model.adapter.freeze_perceiver = False
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(perceiver_frozen=False, last_command_applied="unfreeze_perceiver")
                        
                        if cmds.get("freeze_xattn") and hasattr(target_model.adapter, '_freeze_xattn_params'):
                            print(f"\n[Live Control] Freezing Gated X-Attn")
                            target_model.adapter._freeze_xattn_params()
                            target_model.adapter.freeze_xattn = True
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(xattn_frozen=True, last_command_applied="freeze_xattn")
                        
                        if cmds.get("unfreeze_xattn") and hasattr(target_model.adapter, '_unfreeze_xattn_params'):
                            print(f"\n[Live Control] Unfreezing Gated X-Attn")
                            target_model.adapter._unfreeze_xattn_params()
                            target_model.adapter.freeze_xattn = False
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(xattn_frozen=False, last_command_applied="unfreeze_xattn")

                        if cmds.get("freeze_prepend_latents") and hasattr(target_model.adapter, '_freeze_prepend_latent_params'):
                            print(f"\n[Live Control] Freezing prepend latents")
                            target_model.adapter._freeze_prepend_latent_params()
                            target_model.adapter.freeze_prepend_latents = True
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(prepend_latents_frozen=True, last_command_applied="freeze_prepend_latents")

                        if cmds.get("unfreeze_prepend_latents") and hasattr(target_model.adapter, '_unfreeze_prepend_latent_params'):
                            print(f"\n[Live Control] Unfreezing prepend latents")
                            target_model.adapter._unfreeze_prepend_latent_params()
                            target_model.adapter.freeze_prepend_latents = False
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(prepend_latents_frozen=False, last_command_applied="unfreeze_prepend_latents")

                        if cmds.get("freeze_lm_pseudotokens") and hasattr(target_model.adapter, '_freeze_lm_pseudotoken_params'):
                            print(f"\n[Live Control] Freezing LM pseudotokens")
                            target_model.adapter._freeze_lm_pseudotoken_params()
                            target_model.adapter.freeze_lm_pseudotokens = True
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(lm_pseudotokens_frozen=True, last_command_applied="freeze_lm_pseudotokens")

                        if cmds.get("unfreeze_lm_pseudotokens") and hasattr(target_model.adapter, '_unfreeze_lm_pseudotoken_params'):
                            print(f"\n[Live Control] Unfreezing LM pseudotokens")
                            target_model.adapter._unfreeze_lm_pseudotoken_params()
                            target_model.adapter.freeze_lm_pseudotokens = False
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(lm_pseudotokens_frozen=False, last_command_applied="unfreeze_lm_pseudotokens")
                        
                        if cmds.get("merge_and_reinit_lora"):
                            print(f"\n[Live Control] Merging and reinitializing LoRA")
                            # Save pre-merge LoRA as a stage (so inference can replay merges)
                            stage_idx = len(lora_stage_dirs)
                            stage_save_dir = output_path / f"_lora_stage_{stage_idx}"
                            stage_save_dir.mkdir(parents=True, exist_ok=True)
                            target_model.llm.save_pretrained(str(stage_save_dir))
                            lora_stage_dirs.append((stage_idx, stage_save_dir))
                            print(f"  Saved LoRA stage {stage_idx} to {stage_save_dir}")
                            target_model.merge_and_reinit_lora()
                            _cached_trainable_params = None
                            need_rebuild = True
                            controller.update_status(last_command_applied="merge_and_reinit_lora")
                            if config.use_wandb:
                                wandb.log({"live_control/merge_at_step": global_step}, step=global_step)
                        
                        if cmds.get("run_inference_sample"):
                            print(f"\n[Live Control] Running inference on a validation sample...")
                            import random as _rng
                            try:
                                val_idx = _rng.randint(0, len(val_dataset) - 1)
                                val_sample = val_dataset[val_idx]
                                # Collate single sample into a batch
                                val_batch_single = active_collate_fn([val_sample])
                                val_batch_single = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch_single.items()}
                                sample_fen = val_batch_single["fen"][0] if "fen" in val_batch_single else "N/A"
                                sample_pgn_moves = val_batch_single["pgn_moves"][0] if "pgn_moves" in val_batch_single else ""
                                inference_prompt = build_commentary_prompt(
                                    fen=sample_fen,
                                    pgn_moves=sample_pgn_moves,
                                    use_pgn_in_prompt=bool(getattr(config, "use_pgn_in_prompt", False)),
                                    prepend_fen_in_prompt=bool(getattr(config, "prepend_fen_in_prompt", False)),
                                )
                                prompt_messages = [{"role": "user", "content": inference_prompt}]
                                prompt_chat_text = _safe_apply_chat_template(
                                    target_model.tokenizer,
                                    prompt_messages,
                                    add_generation_prompt=True,
                                )
                                prompt_user_tokens = int(
                                    target_model.tokenizer(inference_prompt, return_tensors="pt")["input_ids"].shape[1]
                                )
                                prompt_chat_tokens = int(
                                    target_model.tokenizer(prompt_chat_text, return_tensors="pt")["input_ids"].shape[1]
                                )
                                target_model.eval()
                                gen_text = None
                                live_max_new_tokens = int(getattr(config, "val_generation_max_new_tokens", 256))
                                try:
                                    while True:
                                        try:
                                            with torch.no_grad():
                                                ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if use_amp else nullcontext()
                                                with ctx:
                                                    if is_policy_only:
                                                        # Policy-only: no LLM, just run adapter forward pass
                                                        _ = target_model(
                                                            maia_features=val_batch_single["maia_features"],
                                                            side_to_move=val_batch_single["side_to_move"],
                                                            maia_policy=val_batch_single.get("maia_policy"),
                                                            maia_policy_mask=val_batch_single.get("maia_policy_mask"),
                                                            eval_targets=val_batch_single.get("eval_targets"),
                                                            move_eval_indices=val_batch_single.get("move_eval_indices"),
                                                            move_eval_targets=val_batch_single.get("move_eval_targets"),
                                                            move_eval_mask=val_batch_single.get("move_eval_mask"),
                                                            move_ce_indices=val_batch_single.get("move_ce_indices"),
                                                            move_ce_targets=val_batch_single.get("move_ce_targets"),
                                                            move_ce_mask=val_batch_single.get("move_ce_mask"),
                                                        )
                                                    else:
                                                        gen_text = target_model.generate(
                                                            lc0_hidden_states=val_batch_single["lc0_hidden_states"],
                                                            side_to_move=bool(val_batch_single["side_to_move"][0].item()) if isinstance(val_batch_single["side_to_move"], torch.Tensor) else val_batch_single["side_to_move"][0],
                                                            prompt=inference_prompt,
                                                            max_new_tokens=live_max_new_tokens,
                                                            temperature=getattr(config, "val_generation_temperature", 0.7),
                                                            fen=sample_fen,
                                                            maia_policy=val_batch_single.get("maia_policy"),
                                                        )
                                            break
                                        except torch.cuda.OutOfMemoryError as oom_err:
                                            if is_policy_only:
                                                print(f"  [Live Control] CUDA OOM in policy-only inference: {oom_err}")
                                                raise
                                            next_tokens = max(1, live_max_new_tokens // 2)
                                            print(
                                                "  [Live Control] CUDA OOM during live inference "
                                                f"(max_new_tokens={live_max_new_tokens}): {oom_err}"
                                            )
                                            if next_tokens == live_max_new_tokens:
                                                print("  [Live Control] Retry budget exhausted (max_new_tokens=1).")
                                                raise
                                            print(
                                                "  [Live Control] Retrying live inference with reduced token budget: "
                                                f"{live_max_new_tokens} -> {next_tokens}"
                                            )
                                            torch.cuda.empty_cache()
                                            live_max_new_tokens = next_tokens
                                finally:
                                    target_model.train()
                                print(f"  FEN: {sample_fen}")
                                print("  [Live Control] User prompt:")
                                print(inference_prompt)
                                print(f"  [Live Control] Prompt tokens: user={prompt_user_tokens}, chat_template={prompt_chat_tokens}")
                                if gen_text is not None:
                                    print(f"  Generation: {gen_text}")
                                else:
                                    print(f"  (Policy-only mode — no text generation)")

                                # Extract auxiliary predictions if available (chess_fusion / policy_only)
                                aux_info = {}
                                adapter_out = getattr(target_model, '_last_adapter_out', None)
                                if adapter_out is not None and 'policy_logits' in adapter_out:
                                    try:
                                        from training.maia_model import get_maia_mapping, unmirror_policy_move
                                        mapping = get_maia_mapping()

                                        legal_moves = None
                                        stm_from_fen = None
                                        fen_parse_error = None
                                        if isinstance(sample_fen, str) and sample_fen and sample_fen != "N/A":
                                            try:
                                                sample_board = chess.Board(sample_fen)
                                                stm_from_fen = bool(sample_board.turn)
                                                legal_moves = {m.uci() for m in sample_board.legal_moves}
                                                aux_info["num_legal_moves"] = len(legal_moves)
                                                aux_info["position_terminal"] = (len(legal_moves) == 0)
                                            except Exception as fen_parse_err:
                                                fen_parse_error = str(fen_parse_err)
                                                print(f"  [Live Control] Could not parse FEN for legality checks: {fen_parse_error}")
                                        else:
                                            fen_parse_error = "missing FEN"

                                        # Resolve side-to-move for un-mirroring policy outputs.
                                        # Prefer FEN-derived value (source of truth), fall back to batch tensor.
                                        _stm = val_batch_single["side_to_move"]
                                        _stm_batch = _stm[0] if isinstance(_stm, (list, torch.Tensor)) else _stm
                                        if isinstance(_stm_batch, torch.Tensor):
                                            _stm_batch = bool(_stm_batch.item())
                                        else:
                                            _stm_batch = bool(_stm_batch)
                                        if stm_from_fen is not None and _stm_batch != stm_from_fen:
                                            print(
                                                "  [Live Control] WARNING: side_to_move mismatch "
                                                f"(batch={_stm_batch}, fen={stm_from_fen}); using FEN side"
                                            )
                                        _stm_val = stm_from_fen if stm_from_fen is not None else _stm_batch

                                        decode_stm = _stm_val
                                        legal_vocab_mask = None

                                        def _build_legal_vocab_mask(vocab_size: int, device: torch.device, stm_val: bool) -> torch.Tensor:
                                            mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                                            for idx in range(vocab_size):
                                                rel_uci = mapping.decode(idx)
                                                if rel_uci is None:
                                                    continue
                                                abs_uci = unmirror_policy_move(rel_uci, stm_val)
                                                if abs_uci in legal_moves:
                                                    mask[idx] = True
                                            return mask

                                        if legal_moves is not None:
                                            vocab_size = int(adapter_out['policy_logits'][0].numel())
                                            legal_vocab_mask = _build_legal_vocab_mask(
                                                vocab_size=vocab_size,
                                                device=adapter_out['policy_logits'][0].device,
                                                stm_val=decode_stm,
                                            )
                                            primary_hits = int(legal_vocab_mask.sum().item())
                                            aux_info["legal_vocab_hits_primary"] = primary_hits
                                            if primary_hits == 0 and len(legal_moves) > 0:
                                                flipped_stm = not decode_stm
                                                flipped_mask = _build_legal_vocab_mask(
                                                    vocab_size=vocab_size,
                                                    device=adapter_out['policy_logits'][0].device,
                                                    stm_val=flipped_stm,
                                                )
                                                flipped_hits = int(flipped_mask.sum().item())
                                                aux_info["legal_vocab_hits_flipped_stm"] = flipped_hits
                                                if flipped_hits > primary_hits:
                                                    print(
                                                        "  [Live Control] WARNING: decode side fallback engaged "
                                                        f"(primary_hits={primary_hits}, flipped_hits={flipped_hits})"
                                                    )
                                                    decode_stm = flipped_stm
                                                    legal_vocab_mask = flipped_mask
                                                    aux_info["decode_side_fallback_used"] = True

                                        def _decode_top_moves(logits_tensor: torch.Tensor, top_k: int = 5, legal_only: bool = False):
                                            if legal_only and legal_moves is None:
                                                return []
                                            if legal_only:
                                                if legal_vocab_mask is None:
                                                    return []
                                                legal_count = int(legal_vocab_mask.sum().item())
                                                if legal_count <= 0:
                                                    return []
                                                probs = torch.softmax(
                                                    logits_tensor.masked_fill(~legal_vocab_mask, float('-inf')),
                                                    dim=-1,
                                                )
                                                scan_k = min(top_k, legal_count)
                                            else:
                                                probs = torch.softmax(logits_tensor, dim=-1)
                                                scan_k = min(int(probs.numel()), top_k)
                                            vals, idxs = probs.topk(scan_k)
                                            decoded = []
                                            for prob, idx in zip(vals.tolist(), idxs.tolist()):
                                                uci = mapping.decode(idx) or f"?{idx}"
                                                uci = unmirror_policy_move(uci, decode_stm)
                                                is_legal = (legal_moves is None) or (uci in legal_moves)
                                                if legal_only and legal_moves is not None and not is_legal:
                                                    continue
                                                decoded.append({
                                                    "move": uci,
                                                    "prob": round(prob * 100, 1),
                                                    "legal": bool(is_legal),
                                                })
                                                if len(decoded) >= top_k:
                                                    break
                                            return decoded

                                        # Adapter policy top-5
                                        pol_logits = adapter_out['policy_logits'][0]  # (1880,)
                                        if legal_moves is not None:
                                            top_moves = _decode_top_moves(pol_logits, top_k=5, legal_only=True)
                                            top_moves_raw = _decode_top_moves(pol_logits, top_k=5, legal_only=False)
                                        else:
                                            top_moves = []
                                            top_moves_raw = []
                                        aux_info["top_moves"] = [
                                            {"move": m["move"], "prob": m["prob"]}
                                            for m in top_moves
                                        ]
                                        if legal_moves is not None:
                                            aux_info["top_moves_illegal_in_raw_top5"] = int(
                                                sum(1 for m in top_moves_raw if not m["legal"])
                                            )
                                        else:
                                            aux_info["legality_unavailable"] = True
                                            aux_info["legality_error"] = fen_parse_error or "unknown FEN parse failure"

                                        # Maia backbone top-5 (distillation target)
                                        tgt_moves = []
                                        _mp_mask = val_batch_single.get("maia_policy_mask")
                                        if _mp_mask is not None:
                                            try:
                                                aux_info["has_maia_policy_supervision"] = bool(_mp_mask[0].item())
                                            except Exception:
                                                pass
                                        tgt_logits_batch = adapter_out.get('policy_targets', None)
                                        if tgt_logits_batch is not None:
                                            tgt_logits = tgt_logits_batch[0]
                                            if legal_moves is not None:
                                                tgt_moves = _decode_top_moves(tgt_logits, top_k=5, legal_only=True)
                                                tgt_moves_raw = _decode_top_moves(tgt_logits, top_k=5, legal_only=False)
                                            else:
                                                tgt_moves = []
                                                tgt_moves_raw = []
                                            aux_info["maia_top_moves"] = [
                                                {"move": m["move"], "prob": m["prob"]}
                                                for m in tgt_moves
                                            ]
                                            if legal_moves is not None:
                                                aux_info["maia_top_moves_illegal_in_raw_top5"] = int(
                                                    sum(1 for m in tgt_moves_raw if not m["legal"])
                                                )
                                        else:
                                            aux_info["maia_top_moves"] = []
                                            if aux_info.get("has_maia_policy_supervision") is False:
                                                aux_info["maia_target_unavailable_reason"] = (
                                                    "No maia_policy target on this sampled position."
                                                )
                                            else:
                                                aux_info["maia_target_unavailable_reason"] = (
                                                    "policy_targets unavailable (no precomputed_policy in sample and teacher disabled)."
                                                )

                                        # Move-eval predictions on supervised CE top-k moves
                                        move_eval_logits_batch = adapter_out.get('move_eval_logits', None)
                                        move_eval_mse_logits_batch = adapter_out.get('move_eval_mse_logits', None)
                                        move_eval_cp_logits_batch = (
                                            move_eval_mse_logits_batch
                                            if move_eval_mse_logits_batch is not None
                                            else move_eval_logits_batch
                                        )
                                        if move_eval_logits_batch is not None or move_eval_cp_logits_batch is not None:
                                            aux_info["move_eval_topk"] = []
                                            ce_indices_batch = val_batch_single.get("move_ce_indices")
                                            ce_targets_batch = val_batch_single.get("move_ce_targets")
                                            ce_mask_batch = val_batch_single.get("move_ce_mask")
                                            if ce_indices_batch is None:
                                                ce_indices_batch = val_batch_single.get("move_eval_indices")
                                            if ce_targets_batch is None:
                                                ce_targets_batch = val_batch_single.get("move_eval_targets")
                                            if ce_mask_batch is None:
                                                ce_mask_batch = val_batch_single.get("move_eval_mask")
                                            if ce_indices_batch is not None and ce_targets_batch is not None and ce_mask_batch is not None:
                                                ce_indices_row = ce_indices_batch[0]
                                                ce_targets_row = ce_targets_batch[0]
                                                ce_mask_row = ce_mask_batch[0].bool()
                                                if ce_mask_row.any():
                                                    cand_idx = ce_indices_row[ce_mask_row].long()
                                                    cand_cp = ce_targets_row[ce_mask_row].float()
                                                    ce_topk = int(max(1, getattr(target_model.adapter.cfg, "move_eval_ce_topk", 5)))
                                                    ce_temp = float(max(1e-6, getattr(target_model.adapter.cfg, "move_eval_ce_cp_temperature", 128.0)))
                                                    cp_scale = float(getattr(target_model.adapter.cfg, "move_eval_cp_scale", 512.0))
                                                    k = min(ce_topk, int(cand_idx.numel()))
                                                    if k > 0:
                                                        top_cp, top_pos = torch.topk(cand_cp, k=k, largest=True, sorted=True)
                                                        top_idx = cand_idx[top_pos]
                                                        target_probs = torch.softmax(top_cp / ce_temp, dim=0)
                                                        policy_probs = torch.softmax(pol_logits, dim=-1).gather(0, top_idx)
                                                        pred_norm = move_eval_cp_logits_batch[0].gather(0, top_idx)
                                                        pred_cp = pred_norm * cp_scale
                                                        decoded_rows = []
                                                        for i in range(k):
                                                            idx_i = int(top_idx[i].item())
                                                            rel_uci = mapping.decode(idx_i) or f"?{idx_i}"
                                                            abs_uci = unmirror_policy_move(rel_uci, decode_stm)
                                                            decoded_rows.append({
                                                                "move": abs_uci,
                                                                "target_cp": round(float(top_cp[i].item()), 1),
                                                                "pred_cp": round(float(pred_cp[i].item()), 1),
                                                                "target_prob": round(float(target_probs[i].item() * 100.0), 1),
                                                                "policy_prob": round(float(policy_probs[i].item() * 100.0), 1),
                                                            })
                                                        aux_info["move_eval_topk"] = decoded_rows
                                                    else:
                                                        aux_info["move_eval_unavailable_reason"] = "No supervised CE/eval candidates after top-k selection."
                                                else:
                                                    aux_info["move_eval_unavailable_reason"] = "Sample has zero supervised move-eval targets."
                                            else:
                                                aux_info["move_eval_unavailable_reason"] = "No move-eval targets in sample (.pt likely missing move_evals/stockfish_best_moves)."

                                            # Always expose model-predicted evals for adapter policy top-5 moves (no targets required).
                                            try:
                                                cp_scale = float(getattr(target_model.adapter.cfg, "move_eval_cp_scale", 512.0))
                                                if legal_moves is not None and legal_vocab_mask is not None and bool(legal_vocab_mask.any().item()):
                                                    pol_probs = torch.softmax(
                                                        pol_logits.masked_fill(~legal_vocab_mask, float('-inf')),
                                                        dim=-1,
                                                    )
                                                else:
                                                    pol_probs = torch.softmax(pol_logits, dim=-1)
                                                k_pol = min(5, int(pol_probs.numel()))
                                                vals_pol, idxs_pol = pol_probs.topk(k_pol)
                                                pred_eval_pol = move_eval_cp_logits_batch[0].gather(0, idxs_pol) * cp_scale
                                                policy_rows = []
                                                for i in range(k_pol):
                                                    idx_i = int(idxs_pol[i].item())
                                                    rel_uci = mapping.decode(idx_i) or f"?{idx_i}"
                                                    abs_uci = unmirror_policy_move(rel_uci, decode_stm)
                                                    policy_rows.append({
                                                        "move": abs_uci,
                                                        "policy_prob": round(float(vals_pol[i].item() * 100.0), 1),
                                                        "pred_cp": round(float(pred_eval_pol[i].item()), 1),
                                                    })
                                                aux_info["move_eval_policy_top5"] = policy_rows
                                            except Exception:
                                                pass
                                        else:
                                            aux_info["move_eval_topk"] = []
                                            aux_info["move_eval_unavailable_reason"] = "Move-eval head inactive (check use_structured_policy_head / checkpoint load)."

                                        # --- BSR: Board State Reconstruction ---
                                        bsr_piece_symbols = ['P','N','B','R','Q','K','p','n','b','r','q','k','.']
                                        try:
                                            bsr_logits = adapter_out.get('bsr_logits')
                                            if bsr_logits is None:
                                                # Run BSR head manually from perceiver latents
                                                latents = adapter_out.get('perceiver_latents')
                                                if latents is not None and hasattr(target_model.adapter, 'bsr_head'):
                                                    bsr_logits = target_model.adapter.bsr_head(latents)
                                            if bsr_logits is not None:
                                                bsr_pred = bsr_logits[0].argmax(dim=-1)  # (64,)
                                                # Get ground truth from bsr_targets or compute from FEN
                                                bsr_targets = adapter_out.get('bsr_targets')
                                                if bsr_targets is None:
                                                    from training.chess_fusion_model import extract_piece_types, extract_maia_features
                                                    gt_boards = extract_maia_features(sample_fen).unsqueeze(0).to(bsr_logits.device)
                                                    bsr_targets = extract_piece_types(gt_boards)
                                                bsr_gt = bsr_targets[0] if bsr_targets is not None else None  # (64,)
                                                # Build 8x8 predicted board string (rank 8 at top)
                                                pred_rows = []
                                                correct = 0
                                                total = 64
                                                for rank in range(7, -1, -1):  # rank 8 down to rank 1
                                                    row = []
                                                    for file in range(8):
                                                        sq = rank * 8 + file
                                                        p = bsr_pred[sq].item()
                                                        row.append(bsr_piece_symbols[p])
                                                        if bsr_gt is not None and p == bsr_gt[sq].item():
                                                            correct += 1
                                                    pred_rows.append(' '.join(row))
                                                aux_info["bsr_board"] = pred_rows  # list of 8 strings
                                                if bsr_gt is not None:
                                                    aux_info["bsr_accuracy"] = round(correct / total * 100, 1)
                                                    # Ground truth board for comparison
                                                    gt_rows = []
                                                    for rank in range(7, -1, -1):
                                                        row = []
                                                        for file in range(8):
                                                            sq = rank * 8 + file
                                                            row.append(bsr_piece_symbols[bsr_gt[sq].item()])
                                                        gt_rows.append(' '.join(row))
                                                    aux_info["bsr_gt_board"] = gt_rows
                                        except Exception as bsr_e:
                                            print(f"  [Live Control] BSR extraction failed: {bsr_e}")

                                        # --- SPP: Square Property Prediction ---
                                        try:
                                            spp_preds = adapter_out.get('spp_preds')
                                            if spp_preds is None:
                                                latents = adapter_out.get('perceiver_latents')
                                                if latents is not None and hasattr(target_model.adapter, 'spp_head'):
                                                    spp_preds = target_model.adapter.spp_head(latents)
                                            if spp_preds is not None:
                                                sp = spp_preds[0]  # (64, 10)
                                                # Channels: [0]=white_attack_count, [1]=black_attack_count, [2:10]=ray distances
                                                w_atk = sp[:, 0].sum().item()  # total white attacks across board
                                                b_atk = sp[:, 1].sum().item()  # total black attacks across board
                                                avg_ray = sp[:, 2:].mean().item()  # avg ray distance
                                                aux_info["spp_summary"] = {
                                                    "white_attacks": round(w_atk, 1),
                                                    "black_attacks": round(b_atk, 1),
                                                    "avg_ray_dist": round(avg_ray, 2),
                                                }
                                                # Get ground truth for comparison
                                                spp_targets = adapter_out.get('spp_targets')
                                                if spp_targets is None:
                                                    try:
                                                        from training.chess_fusion_model import compute_spp_targets, extract_maia_features, DynamicMaskBuilder
                                                        gt_boards = extract_maia_features(sample_fen).unsqueeze(0).to(spp_preds.device)
                                                        mask_builder = target_model.adapter.spp_mask_builder
                                                        spp_targets = compute_spp_targets(gt_boards, mask_builder)
                                                    except Exception:
                                                        spp_targets = None
                                                if spp_targets is not None:
                                                    gt = spp_targets[0]  # (64, 10)
                                                    gt_w_atk = gt[:, 0].sum().item()
                                                    gt_b_atk = gt[:, 1].sum().item()
                                                    gt_avg_ray = gt[:, 2:].mean().item()
                                                    aux_info["spp_gt_summary"] = {
                                                        "white_attacks": round(gt_w_atk, 1),
                                                        "black_attacks": round(gt_b_atk, 1),
                                                        "avg_ray_dist": round(gt_avg_ray, 2),
                                                    }
                                        except Exception as spp_e:
                                            print(f"  [Live Control] SPP extraction failed: {spp_e}")

                                        # Print summary
                                        moves_str = "  ".join(f"{m['move']}={m['prob']}%" for m in top_moves)
                                        if moves_str:
                                            print(f"  Adapter policy (top-5): {moves_str}")
                                        elif aux_info.get("legality_unavailable"):
                                            print("  Adapter policy (top-5): unavailable (legality check unavailable)")
                                        elif aux_info.get("position_terminal"):
                                            print("  Adapter policy (top-5): (terminal position — no legal moves)")
                                        else:
                                            print("  Adapter policy (top-5): unavailable (no legal move found)")
                                        if 'maia_top_moves' in aux_info:
                                            maia_str = "  ".join(f"{m['move']}={m['prob']}%" for m in tgt_moves)
                                            if maia_str:
                                                print(f"  Maia backbone  (top-5): {maia_str}")
                                            elif aux_info.get("maia_target_unavailable_reason"):
                                                print(
                                                    "  Maia backbone  (top-5): unavailable - "
                                                    f"{aux_info.get('maia_target_unavailable_reason')}"
                                                )
                                            elif aux_info.get("legality_unavailable"):
                                                print("  Maia backbone  (top-5): unavailable (legality check unavailable)")
                                            elif aux_info.get("position_terminal"):
                                                print("  Maia backbone  (top-5): (terminal position — no legal moves)")
                                            else:
                                                print("  Maia backbone  (top-5): unavailable (no legal move found)")
                                        if "num_legal_moves" in aux_info:
                                            terminal_suffix = " (terminal)" if aux_info.get("position_terminal") else ""
                                            print(f"  Legal moves in position: {aux_info['num_legal_moves']}{terminal_suffix}")
                                        if aux_info.get("legality_unavailable"):
                                            print(f"  Legality checks unavailable: {aux_info.get('legality_error', 'unknown error')}")
                                        if "top_moves_illegal_in_raw_top5" in aux_info:
                                            print(
                                                "  Adapter raw top-5 illegal moves: "
                                                f"{aux_info['top_moves_illegal_in_raw_top5']}"
                                            )
                                        if "maia_top_moves_illegal_in_raw_top5" in aux_info:
                                            print(
                                                "  Maia raw top-5 illegal moves: "
                                                f"{aux_info['maia_top_moves_illegal_in_raw_top5']}"
                                            )
                                        if "move_eval_topk" in aux_info:
                                            if aux_info["move_eval_topk"]:
                                                rows = "  ".join(
                                                    f"{m['move']}(tgt={m['target_cp']}cp pred={m['pred_cp']}cp tgtP={m['target_prob']}% polP={m['policy_prob']}%)"
                                                    for m in aux_info["move_eval_topk"]
                                                )
                                                print(f"  Move-eval top-k: {rows}")
                                            else:
                                                print(
                                                    "  Move-eval top-k: unavailable - "
                                                    f"{aux_info.get('move_eval_unavailable_reason', 'unknown reason')}"
                                                )
                                        if "move_eval_policy_top5" in aux_info and aux_info["move_eval_policy_top5"]:
                                            rows = "  ".join(
                                                f"{m['move']}(pred={m['pred_cp']}cp polP={m['policy_prob']}%)"
                                                for m in aux_info["move_eval_policy_top5"]
                                            )
                                            print(f"  Move-eval on policy top-5: {rows}")
                                        if 'bsr_accuracy' in aux_info:
                                            print(f"  BSR accuracy: {aux_info['bsr_accuracy']}%")
                                        if 'spp_summary' in aux_info:
                                            s = aux_info['spp_summary']
                                            print(f"  SPP: W_atk={s['white_attacks']}, B_atk={s['black_attacks']}, avg_ray={s['avg_ray_dist']}")
                                    except Exception as aux_e:
                                        print(f"  [Live Control] Aux extraction failed: {aux_e}")

                                inference_result = {
                                    "fen": sample_fen,
                                    "commentary": gen_text if gen_text is not None else "(policy-only mode — no text generation)",
                                    "step": global_step,
                                    "timestamp": time.strftime("%H:%M:%S"),
                                }
                                if aux_info:
                                    inference_result["aux"] = aux_info
                                controller.update_status(
                                    inference_result=inference_result,
                                    last_command_applied="run_inference_sample",
                                )
                                if config.use_wandb:
                                    try:
                                        table = wandb.Table(columns=["step", "fen", "commentary"])
                                        table.add_data(global_step, sample_fen, gen_text)
                                        wandb.log({"live_control/inference_sample": table}, step=global_step)
                                    except Exception:
                                        pass
                            except Exception as e:
                                print(f"  [Live Control] Inference failed: {e}")
                                controller.update_status(
                                    inference_result={"error": str(e), "step": global_step, "timestamp": time.strftime("%H:%M:%S")},
                                    last_command_applied="run_inference_sample",
                                )
                        
                        if need_rebuild or cmds.get("rebuild_optimizer"):
                            print(f"[Live Control] Rebuilding optimizer at step {global_step}")
                            optimizer = build_optimizer()
                            remaining_steps = (config.num_epochs - epoch) * steps_per_epoch - step
                            rebuild_warmup = int(steps_per_epoch * epoch_warmup_ratio) if epoch_warmup_ratio else 0
                            scheduler = get_linear_schedule_with_warmup(
                                optimizer,
                                num_warmup_steps=rebuild_warmup,
                                num_training_steps=max(remaining_steps, 1),
                            )
                            rebuild_cfg = config.model.chess_fusion if config.model.mode in ("chess_fusion", "policy_only") else config.model.maia
                            active_csmp_ratio = getattr(rebuild_cfg, 'csmp_lr_ratio', None)
                            if active_csmp_ratio is None:
                                active_csmp_ratio = rebuild_cfg.perceiver_lr_ratio
                            active_text_gate_ratio = getattr(rebuild_cfg, 'text_gate_lr_ratio', None)
                            if active_text_gate_ratio is None:
                                active_text_gate_ratio = getattr(rebuild_cfg, 'xattn_lr_ratio', 1.0)
                            active_pseudotoken_ratio = getattr(rebuild_cfg, 'pseudotoken_lr_ratio', None)
                            if active_pseudotoken_ratio is None:
                                active_pseudotoken_ratio = getattr(rebuild_cfg, 'xattn_lr_ratio', 1.0)
                            active_prepend_latent_ratio = getattr(rebuild_cfg, 'prepend_latent_lr_ratio', None)
                            if active_prepend_latent_ratio is None:
                                active_prepend_latent_ratio = getattr(rebuild_cfg, 'xattn_lr_ratio', 1.0)
                            controller.update_status(
                                active_base_lr=learning_rate,
                                active_lr_ratios={
                                    "cnn_lr_ratio": rebuild_cfg.cnn_lr_ratio,
                                    "transformer_lr_ratio": rebuild_cfg.transformer_lr_ratio,
                                    "perceiver_lr_ratio": rebuild_cfg.perceiver_lr_ratio,
                                    "csmp_lr_ratio": active_csmp_ratio,
                                    "xattn_lr_ratio": getattr(rebuild_cfg, 'xattn_lr_ratio', 1.0),
                                    "text_gate_lr_ratio": active_text_gate_ratio,
                                    "pseudotoken_lr_ratio": active_pseudotoken_ratio,
                                    "prepend_latent_lr_ratio": active_prepend_latent_ratio,
                                    "lora_lr_ratio": rebuild_cfg.lora_lr_ratio,
                                },
                                last_command_applied="rebuild_optimizer",
                            )
                            if config.use_wandb:
                                wandb.log({
                                    "live_control/rebuild_step": global_step,
                                    "live_control/base_lr": learning_rate,
                                    "live_control/cnn_lr_ratio": rebuild_cfg.cnn_lr_ratio,
                                    "live_control/transformer_lr_ratio": rebuild_cfg.transformer_lr_ratio,
                                    "live_control/perceiver_lr_ratio": rebuild_cfg.perceiver_lr_ratio,
                                    "live_control/csmp_lr_ratio": active_csmp_ratio,
                                    "live_control/xattn_lr_ratio": getattr(rebuild_cfg, 'xattn_lr_ratio', 1.0),
                                    "live_control/text_gate_lr_ratio": active_text_gate_ratio,
                                    "live_control/pseudotoken_lr_ratio": active_pseudotoken_ratio,
                                    "live_control/prepend_latent_lr_ratio": active_prepend_latent_ratio,
                                    "live_control/lora_lr_ratio": rebuild_cfg.lora_lr_ratio,
                                }, step=global_step)
                
                # Logging
                if global_step % config.logging_steps == 0 and global_rank == 0:
                    avg_loss = running_loss / config.logging_steps
                    lr = scheduler.get_last_lr()[0]
                    
                    # Calculate timing metrics
                    log_interval_duration = time.time() - last_log_time
                    steps_in_interval = config.logging_steps
                    
                    # Prepare logs
                    if config.model.mode in ("chess_fusion", "policy_only"):
                        maia_cfg = config.model.chess_fusion
                    else:
                        maia_cfg = config.model.maia
                    logs = {
                        "train/loss": avg_loss,
                        "train/grad_norm": last_grad_norm,
                        "train/learning_rate": lr,
                        "train/base_learning_rate": learning_rate,
                        "train/cnn_lr_ratio": maia_cfg.cnn_lr_ratio,
                        "train/transformer_lr_ratio": maia_cfg.transformer_lr_ratio,
                        "train/perceiver_lr_ratio": maia_cfg.perceiver_lr_ratio,
                        "train/lora_lr_ratio": maia_cfg.lora_lr_ratio,
                        "train/global_step": global_step,
                        "train/epoch": epoch + 1,
                        # Performance metrics
                        "perf/data_loading_sec": profile_stats["data_time"] / steps_in_interval,
                        "perf/forward_pass_sec": profile_stats["forward_time"] / steps_in_interval,
                        "perf/backward_pass_sec": profile_stats["backward_time"] / steps_in_interval,
                        "perf/grad_clip_sec": profile_stats["grad_clip_time"] / steps_in_interval,
                        "perf/optimizer_sec": profile_stats["optimizer_time"] / steps_in_interval,
                        "perf/step_total_sec": log_interval_duration / steps_in_interval,
                        "perf/samples_per_sec": (steps_in_interval * effective_batch_size) / log_interval_duration,
                        # Sequence length stats (dynamic padding)
                        "seq/mean_real_len": seq_length_stats["sum_real"] / max(seq_length_stats["count"], 1),
                        "seq/max_real_len": seq_length_stats["max_real"],
                        "seq/min_real_len": seq_length_stats["min_real"] if seq_length_stats["min_real"] != float("inf") else 0,
                        "seq/padded_len": seq_length_stats["total_padded"] / max(seq_length_stats["n_batches"], 1),
                        "seq/padding_efficiency": seq_length_stats["sum_real"] / max(seq_length_stats["total_padded"] * seq_length_stats["count"] / max(seq_length_stats["n_batches"], 1), 1),
                        "seq/waste_tokens_per_step": (
                            seq_length_stats["total_padded"] * seq_length_stats["count"] / max(seq_length_stats["n_batches"], 1)
                            - seq_length_stats["sum_real"]
                        ) / max(seq_length_stats["n_batches"], 1),
                    }
                    # Throughput: tokens/sec and approximate MFU (only for LM mode with real sequences)
                    if not is_policy_only and log_interval_duration > 0:
                        _tokens_per_sec = seq_length_stats["sum_real"] / log_interval_duration
                        logs["perf/tokens_per_sec"] = _tokens_per_sec
                        _gpu_peak_flops = prof_cfg.gpu_peak_tflops * 1e12
                        if _gpu_peak_flops > 0:
                            # Approximate: 6 * N_params * tokens (forward + backward ≈ 3× forward)
                            if '_n_params_llm' not in dir():
                                _n_params_llm = sum(p.numel() for p in target_model.llm.parameters())
                            logs["perf/mfu"] = (6 * _n_params_llm * _tokens_per_sec) / _gpu_peak_flops
                    # Extended profiling metrics (when profiling.enabled=True)
                    if prof_cfg.enabled:
                        _n_prof = max(profile_stats["_prof_sample_count"], 1)
                        if _prof_memory:
                            logs["prof/mem_fwd_peak_mb"] = profile_stats["mem_fwd_peak_mb"]
                            logs["prof/mem_bwd_peak_mb"] = profile_stats["mem_bwd_peak_mb"]
                        if config.model.mode in ("chess_fusion", "policy_only") and profile_stats["_prof_sample_count"] > 0:
                            logs["prof/adapter_multiscale_ms"] = profile_stats["adapter_multiscale_ms"] / _n_prof
                            logs["prof/adapter_perceiver_ms"]  = profile_stats["adapter_perceiver_ms"]  / _n_prof
                            logs["prof/policy_head_ms"]        = profile_stats["policy_head_ms"]        / _n_prof
                            logs["prof/llm_xattn_ms"]          = profile_stats["llm_xattn_ms"]          / _n_prof
                            logs["prof/llm_orig_ms"]           = profile_stats["llm_orig_ms"]            / _n_prof
                            if profile_stats.get("csmp_ms", 0.0) > 0:
                                logs["prof/csmp_ms"] = profile_stats["csmp_ms"] / _n_prof
                            if profile_stats.get("bsr_spp_ms", 0.0) > 0:
                                logs["prof/bsr_spp_ms"] = profile_stats["bsr_spp_ms"] / _n_prof
                    
                    # Add per-component losses for chess_fusion / policy_only
                    if config.model.mode in ("chess_fusion", "policy_only"):
                        n = config.logging_steps
                        logs["train/lm_loss"] = running_lm_loss / n
                        logs["train/policy_loss"] = running_policy_loss / n
                        if getattr(maia_cfg, 'structured_xattn_sparse_weight', 0.0) > 0:
                            logs["train/structured_xattn_sparse_loss"] = running_structured_xattn_sparse_loss / n
                        if getattr(maia_cfg, 'structured_xattn_square_diversity_weight', 0.0) > 0:
                            logs["train/structured_xattn_square_diversity_loss"] = running_structured_xattn_square_diversity_loss / n
                            logs["train/structured_xattn_square_usage_entropy"] = running_structured_xattn_square_usage_entropy / n
                        if getattr(maia_cfg, 'structured_xattn_gate_usage_weight', 0.0) > 0:
                            logs["train/structured_xattn_gate_usage_loss"] = running_structured_xattn_gate_usage_loss / n
                            logs["train/structured_xattn_gate_usage_mean_abs"] = running_structured_xattn_gate_usage_mean_abs / n
                        if getattr(maia_cfg, 'bsr_weight', 0.0) > 0:
                            logs["train/bsr_loss"] = running_bsr_loss / n
                        if getattr(maia_cfg, 'spp_weight', 0.0) > 0:
                            logs["train/spp_loss"] = running_spp_loss / n
                        if getattr(maia_cfg, 'aux_move_eval_weight', 0.0) > 0:
                            logs["train/move_eval_loss"] = running_move_eval_loss / n
                            logs["train/move_eval_mse"] = running_move_eval_mse / n
                            logs["train/move_eval_ce"] = running_move_eval_ce / n
                            logs["train/move_eval_pairwise"] = running_move_eval_pairwise / n
                            logs["train/move_eval_mate"] = running_move_eval_mate / n
                        logs["train/xattn_lr_ratio"] = maia_cfg.xattn_lr_ratio
                        prepend_latent_log_ratio = getattr(maia_cfg, "prepend_latent_lr_ratio", None)
                        if prepend_latent_log_ratio is None:
                            prepend_latent_log_ratio = maia_cfg.xattn_lr_ratio
                        logs["train/prepend_latent_lr_ratio"] = prepend_latent_log_ratio
                        logs["train/aux_policy_weight"] = maia_cfg.aux_policy_weight
                        logs["train/structured_xattn_sparse_weight"] = getattr(maia_cfg, 'structured_xattn_sparse_weight', 0.0)
                        logs["train/structured_xattn_square_diversity_weight"] = getattr(maia_cfg, 'structured_xattn_square_diversity_weight', 0.0)
                        logs["train/structured_xattn_square_diversity_target_entropy"] = getattr(maia_cfg, 'structured_xattn_square_diversity_target_entropy', 0.5)
                        logs["train/structured_xattn_gate_usage_weight"] = getattr(maia_cfg, 'structured_xattn_gate_usage_weight', 0.0)
                        logs["train/structured_xattn_gate_usage_target"] = getattr(maia_cfg, 'structured_xattn_gate_usage_target', 0.1)
                        # Perceiver cross-attention entropy
                        if running_entropy_count > 0:
                            logs["entropy/cross_attn_mean"] = running_entropy_sum / running_entropy_count
                            if running_entropy_per_layer is not None:
                                for li, val in enumerate((running_entropy_per_layer / running_entropy_count).tolist()):
                                    logs[f"entropy/cross_attn_layer_{li}"] = val
                            if running_entropy_per_head is not None:
                                for hi, val in enumerate((running_entropy_per_head / running_entropy_count).tolist()):
                                    logs[f"entropy/cross_attn_head_{hi}"] = val
                        
                        # Add effective gate values
                        _target = model.module if is_ddp else model
                        gate_vals = _target.adapter.get_gate_values()
                        logs.update(gate_vals)
                        structured_xattn_metrics = _target.adapter.get_structured_xattn_metrics()
                        logs.update(structured_xattn_metrics)

                    # Chess token weight metrics
                    if getattr(config, 'chess_token_weight_enabled', False):
                        _target = model.module if is_ddp else model
                        _ctw = getattr(_target, '_chess_token_weight_metrics', None)
                        if _ctw is not None:
                            logs["train/chess_token_mean_weight"] = _ctw['mean_weight'].item()
                            logs["train/chess_token_unweighted_lm_loss"] = _ctw['unweighted_lm_loss'].item()
                            logs["train/chess_token_weighted_frac"] = _ctw['weighted_frac'].item()

                    if progress_bar:
                        progress_bar.set_postfix({
                            "loss": f"{avg_loss:.4f}",
                            "lr": f"{lr:.2e}",
                            "t/step": f"{logs['perf/step_total_sec']:.2f}s"
                        })
                    
                    # Log to wandb
                    if config.use_wandb:
                        wandb.log(logs, step=global_step)
                    
                    running_loss = 0.0
                    running_lm_loss = 0.0
                    running_policy_loss = 0.0
                    running_structured_xattn_sparse_loss = 0.0
                    running_structured_xattn_square_diversity_loss = 0.0
                    running_structured_xattn_square_usage_entropy = 0.0
                    running_structured_xattn_gate_usage_loss = 0.0
                    running_structured_xattn_gate_usage_mean_abs = 0.0
                    running_bsr_loss = 0.0
                    running_spp_loss = 0.0
                    running_move_eval_loss = 0.0
                    running_move_eval_mse = 0.0
                    running_move_eval_ce = 0.0
                    running_move_eval_pairwise = 0.0
                    running_move_eval_mate = 0.0
                    running_entropy_sum = 0.0
                    running_entropy_count = 0
                    running_entropy_per_layer = None
                    running_entropy_per_head = None
                    
                    # Reset stats
                    for k in profile_stats:
                        profile_stats[k] = 0.0
                    seq_length_stats = {"total_padded": 0, "sum_real": 0.0, "max_real": 0, "min_real": float("inf"), "count": 0, "n_batches": 0}
                    # Reset peak memory trackers
                    if _prof_memory and torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    last_log_time = time.time()
                    
                    # Update live control status
                    _target = model.module if is_ddp else model
                    controller.update_status(
                        current_epoch=epoch + 1,
                        current_step=global_step,
                        train_loss=avg_loss,
                        active_aux_policy_weight=getattr(_target.adapter.cfg, "aux_policy_weight", 0.0),
                        active_structured_xattn_sparse_weight=getattr(_target.adapter.cfg, "structured_xattn_sparse_weight", 0.0),
                        active_structured_xattn_square_diversity_weight=getattr(_target.adapter.cfg, "structured_xattn_square_diversity_weight", 0.0),
                        active_structured_xattn_square_diversity_target_entropy=getattr(_target.adapter.cfg, "structured_xattn_square_diversity_target_entropy", 0.5),
                        active_structured_xattn_gate_usage_weight=getattr(_target.adapter.cfg, "structured_xattn_gate_usage_weight", 0.0),
                        active_structured_xattn_gate_usage_target=getattr(_target.adapter.cfg, "structured_xattn_gate_usage_target", 0.1),
                        active_aux_move_eval_weight=getattr(_target.adapter.cfg, "aux_move_eval_weight", 0.0),
                        active_move_eval_mse_weight=getattr(_target.adapter.cfg, "move_eval_mse_weight", 0.5),
                        active_move_eval_ce_weight=getattr(_target.adapter.cfg, "move_eval_ce_weight", 0.5),
                        active_move_eval_pairwise_weight=getattr(_target.adapter.cfg, "move_eval_pairwise_weight", 0.0),
                        active_bsr_weight=getattr(_target.adapter.cfg, "bsr_weight", 0.0),
                        active_spp_weight=getattr(_target.adapter.cfg, "spp_weight", 0.0),
                        lora_frozen=not lora_unfrozen,
                        cnn_frozen=_target.adapter.freeze_cnn,
                        transformer_frozen=_target.adapter.freeze_transformer,
                        csmp_frozen=getattr(_target.adapter, 'freeze_csmp', False),
                        perceiver_frozen=getattr(_target.adapter, 'freeze_perceiver', False),
                        prepend_latents_frozen=getattr(_target.adapter, 'freeze_prepend_latents', False),
                    )
                
                # Save checkpoint (keep only latest to save disk space)
                if global_step % config.save_steps == 0 and global_rank == 0:
                    checkpoint_path = output_path / f"checkpoint-{global_step}"
                    target_model = model.module if is_ddp else model
                    target_model.save_pretrained(
                        str(checkpoint_path),
                        save_merged_base=progressive_lora_merge,
                    )
                    # Copy accumulated LoRA stages so inference can replay merges
                    for stage_idx, stage_src in lora_stage_dirs:
                        stage_dest = checkpoint_path / f"lora_stage_{stage_idx}"
                        if not stage_dest.exists():
                            shutil.copytree(str(stage_src), str(stage_dest))
                    if lora_stage_dirs:
                        print(f"  Included {len(lora_stage_dirs)} LoRA stage(s) in checkpoint")
                    # Save training config for reproducibility
                    from dataclasses import asdict
                    config_save_path = checkpoint_path / "config.yaml"
                    if not config_save_path.exists():
                        with open(config_save_path, "w", encoding="utf-8") as _cf:
                            yaml.dump(asdict(config), _cf, default_flow_style=False)
                    save_training_state(
                        checkpoint_path, optimizer, scheduler, plateau_scheduler,
                        epoch, step + 1, False, global_step, best_val_loss, lora_unfrozen,
                        config.wandb_run_id if config.use_wandb else None,
                    )
                    
                    # Delete previous step checkpoint
                    if not hasattr(train, '_last_step_checkpoint'):
                        train._last_step_checkpoint = None
                    if train._last_step_checkpoint and train._last_step_checkpoint.exists():
                        shutil.rmtree(train._last_step_checkpoint)
                        print(f"  Deleted old checkpoint: {train._last_step_checkpoint.name}")
                    train._last_step_checkpoint = checkpoint_path
            
            # Reset batch start time for next iteration
            batch_start_time = time.time()

            if max_steps_per_epoch > 0 and (step + 1) >= max_steps_per_epoch:
                if is_ddp:
                    dist.barrier()
                if global_rank == 0:
                    print(f"[DEBUG] Reached max steps per epoch: {max_steps_per_epoch}")
                break
        
        # Epoch summary
        avg_epoch_loss = epoch_loss / num_train_batches
        if global_rank == 0:
            print(f"\nEpoch {epoch+1} complete. Train loss: {avg_epoch_loss:.4f}")
        
        # Validation evaluation
        skip_val = os.environ.get("DEBUG_SKIP_VAL", "0") == "1"
        if skip_val:
            if global_rank == 0:
                print("[DEBUG] Skipping validation (DEBUG_SKIP_VAL=1)")
            avg_val_loss = 0.0
            val_perplexity = 0.0
            val_loss_per_source = {}
        else:
            model.eval()
            val_loss = 0.0
            num_val_batches = 0
            val_loss_by_source = {}   # source_tag -> [sum, count]
            val_generations = []
            log_val_generations = getattr(config, "log_val_generations", False)
            val_generation_samples = getattr(config, "val_generation_samples", 1)
            val_generation_max_new_tokens = getattr(config, "val_generation_max_new_tokens", 128)
            val_generation_temperature = getattr(config, "val_generation_temperature", 0.7)
            val_generation_log_path = getattr(config, "val_generation_log_path", None)
            val_generation_log_wandb = getattr(config, "val_generation_log_wandb", True)
            
            with torch.no_grad():
                for val_batch in tqdm(val_dataloader, desc="Validation", leave=False):
                    val_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in val_batch.items()}
                    
                    if is_policy_only:
                        # Policy-only: forward only needs maia_features
                        if use_amp:
                            with torch.amp.autocast('cuda', dtype=amp_dtype):
                                val_outputs = model(
                                    maia_features=val_batch["maia_features"],
                                    side_to_move=val_batch["side_to_move"],
                                    maia_policy=val_batch.get("maia_policy"),
                                    maia_policy_mask=val_batch.get("maia_policy_mask"),
                                    eval_targets=val_batch.get("eval_targets"),
                                    move_eval_indices=val_batch.get("move_eval_indices"),
                                    move_eval_targets=val_batch.get("move_eval_targets"),
                                    move_eval_mask=val_batch.get("move_eval_mask"),
                                    move_ce_indices=val_batch.get("move_ce_indices"),
                                    move_ce_targets=val_batch.get("move_ce_targets"),
                                    move_ce_mask=val_batch.get("move_ce_mask"),
                                )
                        else:
                            val_outputs = model(
                                maia_features=val_batch["maia_features"],
                                side_to_move=val_batch["side_to_move"],
                                maia_policy=val_batch.get("maia_policy"),
                                maia_policy_mask=val_batch.get("maia_policy_mask"),
                                eval_targets=val_batch.get("eval_targets"),
                                move_eval_indices=val_batch.get("move_eval_indices"),
                                move_eval_targets=val_batch.get("move_eval_targets"),
                                move_eval_mask=val_batch.get("move_eval_mask"),
                                move_ce_indices=val_batch.get("move_ce_indices"),
                                move_ce_targets=val_batch.get("move_ce_targets"),
                                move_ce_mask=val_batch.get("move_ce_mask"),
                            )
                    else:
                        if use_amp:
                            with torch.amp.autocast('cuda', dtype=amp_dtype):
                                val_outputs = model(
                                    lc0_hidden_states=val_batch["lc0_hidden_states"],
                                    input_ids=val_batch["input_ids"],
                                    attention_mask=val_batch["attention_mask"],
                                    labels=val_batch["labels"],
                                    eval_targets=val_batch.get("eval_targets"),
                                    side_to_move=val_batch["side_to_move"],
                                    fen=val_batch["fen"],
                                    engineered_features=val_batch.get("engineered_features"),
                                    perceiver_features=val_batch.get("perceiver_features"),
                                    maia_features=val_batch.get("maia_features"),
                                    maia_policy=val_batch.get("maia_policy"),
                                    maia_policy_mask=val_batch.get("maia_policy_mask"),
                                    move_eval_indices=val_batch.get("move_eval_indices"),
                                    move_eval_targets=val_batch.get("move_eval_targets"),
                                    move_eval_mask=val_batch.get("move_eval_mask"),
                                    move_ce_indices=val_batch.get("move_ce_indices"),
                                    move_ce_targets=val_batch.get("move_ce_targets"),
                                    move_ce_mask=val_batch.get("move_ce_mask"),
                                    loss_weights=val_batch.get("loss_weights"),
                                )
                        else:
                            val_outputs = model(
                                lc0_hidden_states=val_batch["lc0_hidden_states"],
                                input_ids=val_batch["input_ids"],
                                attention_mask=val_batch["attention_mask"],
                                labels=val_batch["labels"],
                                eval_targets=val_batch.get("eval_targets"),
                                side_to_move=val_batch["side_to_move"],
                                fen=val_batch["fen"],
                                engineered_features=val_batch.get("engineered_features"),
                                perceiver_features=val_batch.get("perceiver_features"),
                                maia_features=val_batch.get("maia_features"),
                                maia_policy=val_batch.get("maia_policy"),
                                maia_policy_mask=val_batch.get("maia_policy_mask"),
                                move_eval_indices=val_batch.get("move_eval_indices"),
                                move_eval_targets=val_batch.get("move_eval_targets"),
                                move_eval_mask=val_batch.get("move_eval_mask"),
                                move_ce_indices=val_batch.get("move_ce_indices"),
                                move_ce_targets=val_batch.get("move_ce_targets"),
                                move_ce_mask=val_batch.get("move_ce_mask"),
                                loss_weights=val_batch.get("loss_weights"),
                            )
                    
                    batch_loss = val_outputs.loss.item()
                    val_loss += batch_loss
                    num_val_batches += 1

                    # Per-source loss tracking
                    source_tags = val_batch.get("source_tag", None)
                    if source_tags:
                        # Determine dominant source for this batch
                        # (mixed batches are rare; attribute to majority source)
                        tag_counts = Counter(source_tags)
                        dominant_tag = tag_counts.most_common(1)[0][0]
                        if dominant_tag not in val_loss_by_source:
                            val_loss_by_source[dominant_tag] = [0.0, 0]
                        val_loss_by_source[dominant_tag][0] += batch_loss
                        val_loss_by_source[dominant_tag][1] += 1

                    lm_active_now = (not is_policy_only) and bool(getattr((model.module if is_ddp else model), 'is_lm_enabled', lambda: True)())
                    if lm_active_now and log_val_generations and len(val_generations) < val_generation_samples:
                        target_model = model.module if is_ddp else model
                        gen_ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if use_amp else nullcontext()
                        with gen_ctx:
                            gen = target_model.generate(
                                lc0_hidden_states=val_batch["lc0_hidden_states"],
                                side_to_move=bool(val_batch["side_to_move"][0].item()) if isinstance(val_batch["side_to_move"], torch.Tensor) else val_batch["side_to_move"][0],
                                prompt=build_commentary_prompt(
                                    fen=val_batch["fen"][0],
                                    pgn_moves=val_batch["pgn_moves"][0] if "pgn_moves" in val_batch else "",
                                    use_pgn_in_prompt=bool(getattr(config, "use_pgn_in_prompt", False)),
                                    prepend_fen_in_prompt=bool(getattr(config, "prepend_fen_in_prompt", False)),
                                ),
                                max_new_tokens=val_generation_max_new_tokens,
                                temperature=val_generation_temperature,
                                fen=val_batch["fen"][0],
                            )
                        val_generations.append({
                            "fen": val_batch["fen"][0],
                            "commentary": gen,
                        })
            
            model.train()
            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
            val_perplexity = torch.exp(torch.tensor(avg_val_loss)).item()
            
            # Per-source validation summary
            val_loss_per_source = {}
            for src_tag, (src_sum, src_count) in val_loss_by_source.items():
                src_avg = src_sum / src_count if src_count > 0 else 0.0
                val_loss_per_source[src_tag] = src_avg
            
            source_detail = ""
            if len(val_loss_per_source) > 1:
                parts = [f"{tag}={loss:.4f}" for tag, loss in sorted(val_loss_per_source.items())]
                source_detail = f" ({', '.join(parts)})"
            
            print(f"  Validation loss: {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}{source_detail}")
            if global_rank == 0:
                controller.update_status(val_loss=avg_val_loss)
            if log_val_generations and val_generations and global_rank == 0:
                print("\n[Validation Generations]")
                for idx, sample in enumerate(val_generations, start=1):
                    print(f"--- Sample {idx} ---")
                    print(f"FEN: {sample['fen']}")
                    print(sample["commentary"])

                if val_generation_log_path:
                    log_path = Path(val_generation_log_path)
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"\n=== Epoch {epoch + 1} (step {global_step}) ===\n")
                        for idx, sample in enumerate(val_generations, start=1):
                            f.write(f"--- Sample {idx} ---\n")
                            f.write(f"FEN: {sample['fen']}\n")
                            f.write(sample["commentary"])
                            f.write("\n")

                if config.use_wandb and val_generation_log_wandb:
                    try:
                        table = wandb.Table(columns=["epoch", "step", "fen", "commentary"])
                        for sample in val_generations:
                            table.add_data(epoch + 1, global_step, sample["fen"], sample["commentary"])
                        wandb.log({"val/generations": table}, step=global_step)
                    except Exception as exc:
                        print(f"[Validation Generations] WandB logging failed: {exc}")
        
        # Step plateau scheduler based on validation loss (each rank does this independently or sync?)
        # Ideally we sync val loss. For now, let's assume they are similar or just run on rank 0?
        # Problem: Scheduler stepping needs to happen on all ranks to keep optimizers in sync.
        # So we should valid on all, and they will get approx same loss.
        # Or even better, all-reduce the loss.
        if is_ddp:
             val_loss_tensor = torch.tensor(avg_val_loss, device=device)
             dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
             avg_val_loss = val_loss_tensor.item()
             val_perplexity = torch.exp(val_loss_tensor).item()
             if global_rank == 0:
                 print(f"  Validation loss (avg): {avg_val_loss:.4f}, Perplexity: {val_perplexity:.2f}")

        # Step plateau scheduler based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        plateau_scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
            print(f"  [ReduceLROnPlateau] LR reduced: {old_lr:.2e} -> {new_lr:.2e}")
        
        # Track best validation loss
        is_new_best = avg_val_loss < best_val_loss
        if is_new_best:
            best_val_loss = avg_val_loss
            print(f"  New best validation loss!")

        # Keep a running best checkpoint (single directory, overwritten on improvement)
        if global_rank == 0:
            running_best_path = output_path / "running_best"
            should_save_running_best = is_new_best or (not running_best_path.exists())
            if should_save_running_best:
                if running_best_path.exists():
                    shutil.rmtree(running_best_path)
                if is_ddp:
                    model.module.save_pretrained(
                        str(running_best_path),
                        save_merged_base=False,
                    )
                else:
                    model.save_pretrained(
                        str(running_best_path),
                        save_merged_base=False,
                    )
                for stage_idx, stage_src in lora_stage_dirs:
                    stage_dest = running_best_path / f"lora_stage_{stage_idx}"
                    if not stage_dest.exists():
                        shutil.copytree(str(stage_src), str(stage_dest))
                from dataclasses import asdict
                config_save_path = running_best_path / "config.yaml"
                if not config_save_path.exists():
                    with open(config_save_path, "w", encoding="utf-8") as _cf:
                        yaml.dump(asdict(config), _cf, default_flow_style=False)
                save_training_state(
                    running_best_path, optimizer, scheduler, plateau_scheduler,
                    epoch, steps_per_epoch, True, global_step, best_val_loss, lora_unfrozen,
                    config.wandb_run_id if config.use_wandb else None,
                )
                print(f"Updated running best checkpoint: {running_best_path}")
        
        # Progressive LoRA: Auto-unfreeze at specified epoch (skipped if live control already changed freeze state)
        if not lora_unfrozen and not lora_manual_override and lora_unfreeze_epoch > 0 and (epoch + 1) >= lora_unfreeze_epoch:
            print(f"\n[LoRA] Auto-unfreezing at epoch {epoch + 1} (per unfreeze_epoch={lora_unfreeze_epoch})")
            target_model = model.module if is_ddp else model
            target_model.unfreeze_lora()
            lora_unfrozen = True
            _cached_trainable_params = None  # Invalidate cache
            
            # Rebuild optimizer with LoRA parameters
            optimizer = build_optimizer()
            # Reset scheduler
            remaining_steps = (config.num_epochs - epoch - 1) * len(train_dataloader)
            unfreeze_warmup = int(steps_per_epoch * epoch_warmup_ratio) if epoch_warmup_ratio else 0
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=unfreeze_warmup,
                num_training_steps=remaining_steps,
            )
            target_model.print_trainable_parameters()
            
            if config.use_wandb and global_rank == 0:
                wandb.log({"progressive_lora/stage": 2, "progressive_lora/unfreeze_epoch": epoch + 1}, step=global_step)
        
        # Log epoch metrics to wandb
        lora_stage = epoch + 1 if progressive_lora_merge else (2 if lora_unfrozen else 1)
        if config.use_wandb and global_rank == 0:
            epoch_logs = {
                "epoch/train_loss": avg_epoch_loss,
                "epoch/val_loss": avg_val_loss,
                "epoch/val_perplexity": val_perplexity,
                "epoch/epoch": epoch + 1,
                "progressive_lora/lora_active": lora_unfrozen or progressive_lora_merge,
                "progressive_lora/stage": lora_stage,
            }
            # Per-source validation losses (when secondary data mixing is active)
            if not skip_val and val_loss_per_source:
                for src_tag, src_loss in val_loss_per_source.items():
                    epoch_logs[f"epoch/val_loss_{src_tag}"] = src_loss
                    epoch_logs[f"epoch/val_perplexity_{src_tag}"] = torch.exp(torch.tensor(src_loss)).item()
            wandb.log(epoch_logs, step=global_step)
        
        # Save epoch checkpoint BEFORE progressive merge so the trained LoRA
        # (not the reinitialized one) is included in the checkpoint.
        epoch_checkpoint_path = output_path / f"epoch-{epoch+1}"
        if global_rank == 0:
            if is_ddp:
                model.module.save_pretrained(
                    str(epoch_checkpoint_path),
                    save_merged_base=False,
                )
            else:
                model.save_pretrained(
                    str(epoch_checkpoint_path),
                    save_merged_base=False,
                )
            # Copy accumulated LoRA stages into the checkpoint so inference
            # can replay the progressive merges without needing sibling dirs
            for stage_idx, stage_src in lora_stage_dirs:
                stage_dest = epoch_checkpoint_path / f"lora_stage_{stage_idx}"
                if not stage_dest.exists():
                    shutil.copytree(str(stage_src), str(stage_dest))
            if lora_stage_dirs:
                print(f"  Included {len(lora_stage_dirs)} prior LoRA stage(s) in checkpoint")
            # Save training config for reproducibility
            from dataclasses import asdict
            config_save_path = epoch_checkpoint_path / "config.yaml"
            if not config_save_path.exists():
                with open(config_save_path, "w", encoding="utf-8") as _cf:
                    yaml.dump(asdict(config), _cf, default_flow_style=False)
            save_training_state(
                epoch_checkpoint_path, optimizer, scheduler, plateau_scheduler,
                epoch, steps_per_epoch, True, global_step, best_val_loss, lora_unfrozen,
                config.wandb_run_id if config.use_wandb else None,
            )
            print(f"Saved epoch {epoch+1} checkpoint to: {epoch_checkpoint_path}")
        
        # Progressive LoRA Merge (AFTER saving the epoch checkpoint)
        if progressive_lora_merge and epoch < config.num_epochs - 1:
            target_model = model.module if is_ddp else model

            # Save current LoRA weights as a stage before merging
            if global_rank == 0:
                stage_idx = len(lora_stage_dirs)
                stage_save_dir = output_path / f"_lora_stage_{stage_idx}"
                stage_save_dir.mkdir(parents=True, exist_ok=True)
                target_model.llm.save_pretrained(str(stage_save_dir))
                lora_stage_dirs.append((stage_idx, stage_save_dir))
                print(f"  Saved LoRA stage {stage_idx} to {stage_save_dir}")

            print(f"\n[Progressive LoRA] Merging and reinitializing LoRA for stage {epoch + 2}")
            target_model.merge_and_reinit_lora()
            _cached_trainable_params = None  # Invalidate cache
            
            optimizer = build_optimizer()
            remaining_steps = (config.num_epochs - epoch - 1) * len(train_dataloader)
            merge_warmup = int(steps_per_epoch * epoch_warmup_ratio) if epoch_warmup_ratio else 0
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=merge_warmup,
                num_training_steps=remaining_steps,
            )
            
            if config.use_wandb and global_rank == 0:
                wandb.log({"progressive_lora/merged_at_epoch": epoch + 1}, step=global_step)
        
        # Delete previous epoch checkpoint
        if global_rank == 0:
            if not hasattr(train, '_last_epoch_checkpoint'):
                train._last_epoch_checkpoint = None
            if train._last_epoch_checkpoint and train._last_epoch_checkpoint.exists():
                shutil.rmtree(train._last_epoch_checkpoint)
                print(f"  Deleted old epoch checkpoint: {train._last_epoch_checkpoint.name}")
            train._last_epoch_checkpoint = epoch_checkpoint_path
        
        # Upload epoch checkpoint to wandb
        if config.use_wandb and global_rank == 0:
            if not hasattr(train, '_epoch_checkpoints'):
                train._epoch_checkpoints = []
            
            artifact_name = f"epoch-{epoch+1}-checkpoint"
            artifact = wandb.Artifact(
                name=artifact_name,
                type="model",
                description=f"Epoch {epoch+1} checkpoint (loss: {avg_epoch_loss:.4f})",
                metadata={
                    "epoch": epoch + 1,
                    "loss": avg_epoch_loss,
                    "global_step": global_step,
                }
            )
            # Add only inference-relevant files (exclude training_state.pt ~1.4GB optimizer state)
            for item in epoch_checkpoint_path.iterdir():
                if item.name == "training_state.pt":
                    continue
                if item.is_dir():
                    artifact.add_dir(str(item), name=item.name)
                else:
                    artifact.add_file(str(item), name=item.name)
            wandb.log_artifact(artifact)
            print(f"Uploaded epoch {epoch+1} checkpoint to wandb")
            
            train._epoch_checkpoints.append(artifact_name)
            
            if len(train._epoch_checkpoints) > 5:
                old_artifact_name = train._epoch_checkpoints.pop(0)
                try:
                    api = wandb.Api()
                    old_artifact = api.artifact(f"{config.wandb_project}/{old_artifact_name}:latest")
                    old_artifact.delete()
                    print(f"Deleted old artifact: {old_artifact_name}")
                except Exception as e:
                    print(f"Note: Could not delete old artifact {old_artifact_name}: {e}")
    
    # Stop torch.profiler (exports final chrome trace if active)
    if _torch_prof is not None:
        _torch_prof.__exit__(None, None, None)
        print(f"[Profiler] Chrome trace written to: {prof_cfg.torch_profiler_output_dir}")
        print(f"[Profiler] View with: tensorboard --logdir={prof_cfg.torch_profiler_output_dir}")

    # Stop live control server
    controller.stop()
    
    # Finish wandb run
    if config.use_wandb and global_rank == 0:
        wandb.finish()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    if global_rank == 0:
        print("Checkpoint retention: epoch-latest, checkpoint-latest, running_best")
        if hasattr(train, "_last_epoch_checkpoint") and train._last_epoch_checkpoint is not None:
            print(f"Latest epoch checkpoint: {train._last_epoch_checkpoint}")
        print(f"Running best checkpoint: {output_path / 'running_best'}")
    print("=" * 60)
    
    return model


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train chess commentary model")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to YAML configuration file")
    
    args = parser.parse_args()
    
    try:
        config = load_config(args.config)
        train(config)
        if dist.is_initialized():
            dist.destroy_process_group()
    except Exception as e:
        print(f"Error starting training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

