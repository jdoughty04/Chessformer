from dataclasses import dataclass, field
from typing import Optional, Any, List
from pathlib import Path
import yaml

try:
    from typing import Literal  # Python 3.8+
except ImportError:  # pragma: no cover - compatibility path
    try:
        from typing_extensions import Literal  # type: ignore
    except ImportError:
        Literal = Any  # type: ignore

@dataclass
class LoRAConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Freeze behavior for the trainable LLM component at training start:
    # - model.use_lora=True  -> controls LoRA adapter params
    # - model.use_lora=False -> controls full LLM params
    # None keeps backward-compatible inference from unfreeze_epoch.
    start_frozen: Optional[bool] = None
    # Auto-unfreeze epoch for the same component as start_frozen above.
    # 0 = no auto-unfreeze (manual only via live control).
    unfreeze_epoch: int = 0
    # Progressive merge only applies when model.use_lora=True.
    progressive_merge: bool = False
    # LoRA target modules (used only when model.use_lora=True).
    # Set to "auto" to auto-detect based on model architecture (recommended for multi-model support).
    # Explicit list overrides auto-detection (e.g. ["q_proj", "v_proj"] for LLaMA).
    target_modules: Any = "auto"
    
    def __post_init__(self):
        if self.start_frozen is None:
            # Backward compat: unfreeze_epoch=1 used to mean "start unfrozen"
            self.start_frozen = self.unfreeze_epoch != 1

@dataclass
class HybridConfig:
    lc0_proj_dim: int = 128

@dataclass
class PerceiverConfig:
    d_model: int = 256
    n_layers_encoder: int = 12
    n_heads: int = 8
    n_latents: int = 64
    n_layers_pooling: int = 4
    mlp_expansion: int = 4
    use_main_engineered_concat: bool = False

@dataclass
class MaiaConfig:
    model_type: str = "rapid"
    elo_self: int = 1500
    elo_oppo: int = 1500
    freeze_backbone: bool = False
    freeze_cnn: bool = False
    freeze_transformer: bool = False
    freeze_perceiver: bool = False
    llm_projection_dim: int = 2048
    adapter_mode: str = "minimal"
    num_latents: int = 8
    perceiver_depth: int = 1
    perceiver_internal_dim: Optional[int] = None  # Internal perceiver dim; None = llm_projection_dim (legacy)
    use_mlp_projections: bool = False
    use_main_engineered_concat: bool = False
    cnn_lr_ratio: float = 0.1
    cnn_learning_rate: Optional[float] = None
    transformer_lr_ratio: float = 0.1
    perceiver_lr_ratio: float = 1.0
    lora_lr_ratio: float = 1.0

@dataclass
class ChessFusionConfig:
    model_type: str = "rapid"
    elo_self: int = 1500
    elo_oppo: int = 1500
    
    # CNN backbone
    use_cnn: bool = True                 # Load and run Maia CNN; False = board-only mode (pos + piece embeddings)
    # Student (trainable) backbone init when use_cnn=True:
    # - maia_pretrained: initialize from Maia pretrained weights
    # - random: reinitialize backbone weights randomly
    # Resume loading from adapter checkpoints is controlled separately by
    # load_checkpoint_backbone.
    backbone_init: Literal["maia_pretrained", "random"] = "maia_pretrained"
    
    # Multi-scale feature tapping (ignored when use_cnn=False)
    tap_projection_dim: int = 1024       # Projection dim for CNN spatial tokens
    cnn_tap_layers: List[int] = field(default_factory=lambda: [4])  # Which CNN block indices to tap (0-4, 5 blocks total)
    concat_cnn_taps: bool = False        # If True, concat all tap outputs per square â†’ single projection (64 tokens instead of N*64)
    use_transformer_taps: bool = True    # Also attend to Maia transformer's 8 channel-tokens
    
    # Chess Structure Message Passing (CSMP)
    use_chess_structure_mp: bool = False  # Enable CSMP between CNN taps and Perceiver
    csmp_layers: int = 4                 # Number of CSMP layers
    csmp_heads: int = 8                  # First 8 are structured (6 static + ray + attack), extra heads are global
    csmp_dim: int = 1024                 # Working dimension inside CSMP
    csmp_pos_dim: int = 32               # Positional embedding dimension (rank + file)
    csmp_piece_dim: int = 64             # Piece-type embedding dimension (13 types)
    csmp_cnn_proj_dim: int = 0           # Per-tap CNN projection dim (0 = no projection, use raw 256-dim)
    csmp_ffn_mult: int = 2               # FFN expansion multiplier in CSMP layers
    csmp_dropout: float = 0.1            # Dropout in CSMP attention + FFN
    csmp_use_ray_mask: bool = True       # Enable dynamic line-of-sight ray masking
    csmp_use_attack_mask: bool = True    # Enable dynamic attack/defense masking
    csmp_use_xy_coords: bool = False     # Add normalized (rank,file) coords to positional features
    csmp_relative_mode: Literal["none", "score_bias", "edge_modulation"] = "none"
                                         # Mutually exclusive CSMP relative-position modes:
                                         # none = masks only, score_bias = additive relative logits,
                                         # edge_modulation = pair-conditioned key/value modulation
    csmp_relative_edge_dim: int = 16     # Edge embedding dim used only by csmp_relative_mode=edge_modulation
    csmp_ablation_no_mask: bool = False  # Ablation: disable all per-head chess masks, use fully-connected attention
    
    # Perspective correction
    use_absolute_coords: bool = True     # Un-mirror CNN features + board tensor for Black-to-move
                                          # so CSMP/BSR/SPP operate in absolute coordinates (a1=square 0,
                                          # White always in channels 0-5). Set False for legacy behaviour.
    
    # Perceiver
    num_latents: int = 32
    perceiver_depth: int = 4
    perceiver_dim: int = 2048
    perceiver_heads: int = 16
    perceiver_dropout: float = 0.1
    use_engineered_concat: bool = False
    structured_latents: bool = False
    latent_context_mask_type: Literal["full", "strict_own_square"] = "full"
    global_latent_attends_all: bool = True
    square_latent_attends_side_token: bool = True
    square_heads_include_global_latent: bool = True
    use_structured_policy_head: bool = False
    structured_policy_query_layers: int = 4
    structured_policy_query_heads: Optional[int] = None  # None = perceiver_heads
    structured_policy_ffn_mult: int = 2
    structured_policy_use_move_bias: bool = True
    
    # Shared layer-conditioned readout (legacy recurrent_query_attn path only;
    # ignored when xattn_mode='structured_square_mixer')
    num_fusion_tokens: int = 16          # Number of learned latents in shared readout (output tokens for LLM cross-attention)
    readout_depth: int = 1               # Number of (text-xattn â†’ policy-xattn[opt] â†’ perc-xattn â†’ csmp-xattn â†’ self-attn â†’ FFN) layers in shared readout
    shared_readout_fourier_dim: int = 64 # Dimension of Fourier encoding for layer fraction conditioning
    readout_use_policy_latent_cross_attention: bool = False  # If True, add policy-latent cross-attention before perceiver xattn in shared readout
    
    # Decoder-layer chess fusion in the LLM
    xattn_layers: List[int] = field(default_factory=lambda: [5, 11, 17])
    xattn_mode: Literal["recurrent_query_attn", "structured_square_mixer"] = "recurrent_query_attn"
    xattn_heads: int = 16
    xattn_gate_init: float = 0.0         # tanh gate init value (0 = identity at start)
    xattn_dropout: float = 0.1           # Dropout in xattn attention + FFN
    xattn_ffn_mult: int = 2              # FFN expansion multiplier in xattn (2x instead of 4x)
    xattn_recurrent_query_state_dim: int = 256 # GRU hidden size for recurrent-query state used by both fusion modes
    xattn_recurrent_query_use_mlp: bool = False # If True, use MLP heads (instead of linear) from recurrent state
    xattn_recurrent_query_share_gru_across_layers: bool = False # Share recurrent-query GRU across all x-attn layers (projections remain per-layer)
    enable_lm_prepend_latents: bool = False  # If True, prepend learned chess latents into LLM text embedding stream
    num_lm_prepend_latents: int = 16         # Number of prepended learned latents (must be >0 when enabled)
    lm_prepend_latent_mode: Literal["cross_attn", "structured_mlp"] = "cross_attn"
    lm_prepend_structured_mlp_hidden_dim: Optional[int] = None  # None = use llm hidden size
    lm_prepend_latents_use_positional_encoding: bool = True  # If False, prepended latent prefix does not consume/distinguish LLM positions
    enable_lm_pseudotokens: bool = True  # Master on/off switch for per-layer LM pseudotoken attention
    num_lm_pseudotokens: int = 0         # Extra learned KV pseudotokens per active LM pseudotoken layer
    lm_pseudotoken_layers: Optional[List[int]] = None  # None = reuse xattn_layers (backward compatible)
    
    # Auxiliary losses
    aux_policy_weight: float = 0.1       # KL div vs Maia policy
    aux_eval_weight: float = 0.0         # Deprecated position-eval bucket CE (kept for backward compatibility)
    structured_xattn_sparse_weight: float = 0.0  # Entropy penalty on structured_square_mixer square marginals (lower = sparser over 64 squares)
    structured_xattn_square_diversity_weight: float = 0.0  # Encourage aggregate square usage to stay above a target entropy floor
    structured_xattn_square_diversity_target_entropy: float = 0.5  # Normalized target entropy floor in [0, 1] over mean 64-square usage
    num_eval_buckets: int = 5

    # Precomputed policy: use cached maia_policy from .pt files instead of live teacher
    use_precomputed_policy: bool = False

    # Per-move evaluation objective:
    # - CE: move-eval logits vs soft Stockfish target distribution on top-k supervised moves
    # - Pairwise: ranking loss on move-eval logits over top-k supervised moves
    # - MSE: predict centipawn eval per supervised move from dedicated eval projections
    aux_move_eval_weight: float = 0.0    # 0 = disabled; try 0.1 when data ready
    move_eval_dim: int = 128             # Projection dim for from/to eval head
    move_eval_cp_scale: float = 512.0    # Normalize cp targets by this (1 pawn â‰ˆ 0.2)
    move_eval_cp_clip: float = 2000.0    # Clip centipawn targets to [-clip, +clip] before CE/MSE
    move_eval_mse_weight: float = 0.5    # Internal balance: MSE fraction of move_eval loss
    move_eval_ce_weight: float = 0.5     # Internal balance: CE fraction
    move_eval_pairwise_weight: float = 0.0  # Internal balance: pairwise ranking fraction
    move_eval_pairwise_topk: int = 5        # Top-k supervised moves used for pairwise ranking
    move_eval_pairwise_cp_margin: float = 0.0  # Ignore pairs with |cp_i - cp_j| <= margin
    move_eval_pairwise_temperature: float = 1.0  # Temperature on predicted score differences
    move_eval_mate_weight: float = 0.25  # Internal balance: mate BCE fraction
    move_eval_mate_threshold_cp: float = 9000.0  # |cp| >= threshold counts as mate target
    move_eval_ce_topk: int = 5           # Number of supervised moves used for CE targets
    move_eval_ce_cp_temperature: float = 128.0  # Temperature (cp units) for soft CE target distribution

    # BSR (Board State Reconstruction) â€” predict piece type per square from Perceiver latents
    bsr_weight: float = 0.0              # 0 = disabled
    bsr_dim: int = 256                   # Internal dim of BSR cross-attention head
    bsr_heads: int = 4                   # Attention heads in BSR head
    bsr_layers: int = 2                  # Cross-attn â†’ self-attn â†’ FFN layers
    bsr_dropout: float = 0.1
    
    # SPP (Square Property Prediction) â€” predict attack counts + ray distances per square
    spp_weight: float = 0.0              # 0 = disabled
    spp_dim: int = 256                   # Internal dim of SPP cross-attention head
    spp_heads: int = 4                   # Attention heads in SPP head
    spp_layers: int = 2                  # Cross-attn â†’ self-attn â†’ FFN layers
    spp_dropout: float = 0.1
    
    # Freeze control
    freeze_cnn: bool = True
    freeze_transformer: bool = True
    freeze_csmp: bool = False            # Freeze CSMP in multi_scale.chess_mp (when CSMP is enabled)
    freeze_perceiver: bool = False       # Freeze SquareLatentEncoder + multi_scale side_token embedding
    freeze_xattn: bool = False
    freeze_prepend_latents: bool = False # Freeze prepend_latent_readout independently of xattn/shared_readout
    freeze_lm_pseudotokens: bool = False # Freeze only learnable LM pseudotokens (independent of freeze_xattn)
    
    # LR ratios
    cnn_lr_ratio: float = 0.001
    cnn_learning_rate: Optional[float] = None
    transformer_lr_ratio: float = 0.01
    perceiver_lr_ratio: float = 1.0
    csmp_lr_ratio: Optional[float] = None  # None = use perceiver_lr_ratio (backward compatible)
    xattn_lr_ratio: float = 1.0
    text_gate_lr_ratio: Optional[float] = None    # None = use xattn_lr_ratio; set higher for freshly-init text gate MLP
    pseudotoken_lr_ratio: Optional[float] = None  # None = use xattn_lr_ratio
    prepend_latent_lr_ratio: Optional[float] = None  # None = use xattn_lr_ratio
    # Multiplier for the trainable LLM component LR:
    # - model.use_lora=True  -> LoRA params LR
    # - model.use_lora=False -> full LLM params LR
    lora_lr_ratio: float = 0.1
    
    # Training phases
    phase1_epochs: int = 3               # Adapter-only warmup
    phase2_epochs: int = 4               # + LoRA + Maia transformer
    # Phase 3 = remaining epochs, unfreeze CNN

    # Checkpoint resume: selective module loading (adapter.pt)
    # Backbone controls whether trainable Maia CNN/transformer weights are
    # resumed from adapter checkpoints (True) or reinitialized from pretrained
    # Maia defaults (False).
    load_checkpoint_backbone: bool = True
    load_checkpoint_csmp: bool = True
    load_checkpoint_perceiver: bool = True
    load_checkpoint_xattn: bool = True
    load_checkpoint_prepend_latents: bool = True
    load_checkpoint_lm_pseudotokens: bool = True
    load_checkpoint_aux_heads: bool = True

    def __post_init__(self):
        valid_relative_modes = {"none", "score_bias", "edge_modulation"}
        if self.csmp_relative_mode not in valid_relative_modes:
            raise ValueError(
                "csmp_relative_mode must be one of "
                "{'none', 'score_bias', 'edge_modulation'} "
                f"(got {self.csmp_relative_mode!r})"
            )
        if int(self.csmp_relative_edge_dim) <= 0:
            raise ValueError(
                f"csmp_relative_edge_dim must be > 0 (got {self.csmp_relative_edge_dim})"
            )

@dataclass
class ProfilingConfig:
    # Master switch â€” if False, all sub-options are no-ops
    enabled: bool = False
    # Replace time.time() forward/backward measurements with CUDA event timing
    # (accurate under async GPU execution; adds one synchronize() at log interval)
    cuda_event_timing: bool = True
    # Add CUDA event timing inside ChessStructureMP.forward()
    # (adds two synchronize() calls per CSMP forward pass â€” keep disabled normally)
    csmp_timing: bool = False
    # Emit NVTX ranges around key train-step regions (for Nsight Systems/Compute).
    # Keep disabled for normal training runs to avoid extra instrumentation overhead.
    nvtx_ranges: bool = False
    # Export a chrome trace via torch.profiler (significant overhead while active)
    torch_profiler: bool = False
    torch_profiler_wait: int = 3       # steps to skip before profiling
    torch_profiler_warmup: int = 2     # steps to warm profiler buffers
    torch_profiler_active: int = 5     # steps to capture trace
    torch_profiler_output_dir: str = "profiler_output"
    # Track peak GPU memory (MB) after forward and backward passes
    memory_snapshots: bool = False
    # Log profiling metrics to wandb under prof/ prefix
    log_to_wandb: bool = True
    # Emit profiling metrics every N global steps (0 = follow logging_steps cadence)
    emit_interval: int = 50
    # Theoretical GPU peak TFLOPs for MFU calculation â€” update per machine
    # RTX 3090 (bf16) â‰ˆ 35.6, A100 SXM (bf16) â‰ˆ 312
    gpu_peak_tflops: float = 35.6

@dataclass
class ModelConfig:
    base_model: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    mode: Literal["hybrid", "engineered", "perceiver", "maia", "chess_fusion", "policy_only"] = "hybrid"
    
    # Universal settings
    load_in_8bit: bool = True
    load_in_4bit: bool = False
    bnb_4bit_compute_dtype: str = "bf16"  # bf16 or fp16
    use_flash_attention: bool = True
    use_torch_compile: bool = True
    # True: wrap LLM with PEFT LoRA and train LoRA params.
    # False: skip PEFT; train full LLM params instead (freeze/unfreeze still controlled by model.lora.*).
    use_lora: bool = True
    # Runtime LM objective/generation gate (primarily for chess_fusion).
    # False keeps LLM loaded but skips LM loss/generation path until re-enabled.
    enable_lm: bool = True
    # Checkpoint resume gate for LLM parameters (LoRA or full LLM).
    # False keeps adapter loading behavior but starts from fresh LM weights.
    load_lm_checkpoint: bool = True
    use_fen_tokens: bool = False
    
    # Feature settings
    engineered_features_type: Literal["simplified", "main"] = "simplified"
    
    # LC0 settings (only used for hybrid mode)
    lc0_dim: int = 768
    
    # Nested configs
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    perceiver: PerceiverConfig = field(default_factory=PerceiverConfig)
    maia: MaiaConfig = field(default_factory=MaiaConfig)
    chess_fusion: ChessFusionConfig = field(default_factory=ChessFusionConfig)

@dataclass
class TrainingConfig:
    experiment_name: str = "default_run"
    output_dir: str = "checkpoints"
    # Directory of `.pt` samples exported from Synthetic_commentary_generation.
    samples_dir: str = "data/preprocessed/samples"
    
    num_epochs: int = 10
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.1
    epoch_warmup_ratio: Optional[float] = None  # Per-epoch warmup; overrides warmup_ratio when set
    max_length: int = 512
    val_split: float = 0.1
    gradient_clip_val: Optional[float] = 30.0
    
    save_steps: int = 100
    logging_steps: int = 1
    max_steps_per_epoch: Optional[int] = None
    
    # Mixed precision
    fp16: bool = False
    bf16: bool = True
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "chess-commentary"
    wandb_run_name: Optional[str] = None
    # Resume existing wandb run when an id is available (from config or checkpoint state).
    wandb_resume: bool = True
    wandb_run_id: Optional[str] = None
    
    # Optimization
    gradient_checkpointing: bool = False
    use_8bit_optimizer: bool = False
    preload_dataset: bool = True          # Cache all samples in RAM at startup (eliminates per-step disk I/O)
    dataloader_num_workers: Optional[int] = None      # None=auto heuristic; set >0 to force worker count
    dataloader_prefetch_factor: int = 4               # Batches prefetched per worker (only when workers > 0)
    dataloader_persistent_workers: Optional[bool] = None  # None=auto (True when workers > 0)

    # Live control
    control_port: int = 8585
    control_poll_steps: int = 10

    # Validation generation logging
    log_val_generations: bool = False
    val_generation_samples: int = 1
    val_generation_max_new_tokens: int = 128
    val_generation_temperature: float = 0.7
    val_generation_log_path: Optional[str] = None
    val_generation_log_wandb: bool = True
    
    # Checkpoint resume (model weights only â€” no optimizer/scheduler state)
    resume_from_checkpoint: Optional[str] = None  # Path to checkpoint dir for warm-starting weights on a new dataset
    # When True, also restore optimizer/scheduler + epoch/step position from training_state.pt.
    # Keep False for weight-only warm starts on new data.
    resume_training_state: bool = False
    
    # Secondary data mixing (grounding retention)
    secondary_samples_dir: Optional[str] = None   # Path to secondary sample directory (None = disabled)
    secondary_mix_ratio: float = 0.15             # Fraction of primary dataset size to sample from secondary
    secondary_mix_seed: int = 42                  # Seed for reproducible subset selection from secondary
    
    # Last move context
    use_last_move_in_prompt: bool = False  # Include last move (from preprocessed data) in the text prompt
    use_pgn_in_prompt: bool = False        # Prepend PGN move list to the text prompt (requires pgn_moves in data)
    prepend_fen_in_prompt: bool = False    # Prepend raw FEN string to the text prompt when available
    pgn_prompt_last_n_moves: Optional[int] = None  # If set, only include last N SAN moves from pgn_moves in prompt

    # Chess token loss weighting: up-weight LM loss on chess-critical tokens
    # (squares, pieces, moves, sides, files) to penalize factual errors more heavily.
    chess_token_weight_enabled: bool = False   # Master switch
    chess_token_weight_squares: float = 3.0    # Tokens matching [a-h][1-8] (e.g. e4, g5)
    chess_token_weight_pieces: float = 2.0     # Piece names: king, queen, rook, bishop, knight, pawn
    chess_token_weight_moves: float = 2.5      # SAN move tokens: Nf3, Bxe5, O-O, exd5, etc.
    chess_token_weight_sides: float = 1.5      # Side references: White, Black, White's, Black's
    chess_token_weight_files: float = 1.5      # File references: a-file, h-file, etc.
    
    # Nested configs
    model: ModelConfig = field(default_factory=ModelConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)

    def __post_init__(self):
        # Allow nested dicts to be converted to dataclasses
        if isinstance(self.model, dict):
            self.model = ModelConfig(**self.model)
        if isinstance(self.profiling, dict):
            self.profiling = ProfilingConfig(**self.profiling)
        if isinstance(self.model.lora, dict):
            self.model.lora = LoRAConfig(**self.model.lora)
        if isinstance(self.model.hybrid, dict):
            self.model.hybrid = HybridConfig(**self.model.hybrid)
        if isinstance(self.model.perceiver, dict):
            self.model.perceiver = PerceiverConfig(**self.model.perceiver)
        if isinstance(self.model.maia, dict):
            self.model.maia = MaiaConfig(**self.model.maia)
        if isinstance(self.model.chess_fusion, dict):
            self.model.chess_fusion = ChessFusionConfig(**self.model.chess_fusion)
            
        # Ensure float types for numeric fields that might be parsed as strings (e.g. from YAML)
        if isinstance(self.learning_rate, str):
            self.learning_rate = float(self.learning_rate)
        if isinstance(self.warmup_ratio, str):
            self.warmup_ratio = float(self.warmup_ratio)
        if isinstance(self.val_split, str):
            self.val_split = float(self.val_split)

def load_config(path: str) -> TrainingConfig:
    """Load configuration from YAML file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
        
    # Force UTF-8 so config parsing is stable across Windows codepages.
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Flatten 'training' section if present (fixes compatibility with nested config structure)
    if "training" in data and isinstance(data["training"], dict):
        training_config = data.pop("training")
        data.update(training_config)

    # Handle nested dataclass initialization
    model_data = data.get("model", {})
    if not isinstance(model_data, dict):
        model_data = {}

    # Backward compatibility: allow model keys at top-level
    if "use_flash_attention" in data and "use_flash_attention" not in model_data:
        model_data["use_flash_attention"] = data.pop("use_flash_attention")
    
    # Handle misplaced gradient_clip_val (it belongs in TrainingConfig)
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

    if "maia" in model_data and isinstance(model_data["maia"], dict):
        # Handle nested LoRA config inside maia (user convenience)
        if "lora" in model_data["maia"]:
            lora_data = model_data["maia"].pop("lora")
            if isinstance(lora_data, dict):
                model_data["lora"] = LoRAConfig(**lora_data)
        
        # Helper: backward compatibility for backbone_lr_ratio
        if "backbone_lr_ratio" in model_data["maia"]:
            ratio = model_data["maia"].pop("backbone_lr_ratio")
            if "cnn_lr_ratio" not in model_data["maia"]:
                model_data["maia"]["cnn_lr_ratio"] = ratio
            if "transformer_lr_ratio" not in model_data["maia"]:
                model_data["maia"]["transformer_lr_ratio"] = ratio
                
        model_data["maia"] = MaiaConfig(**model_data["maia"])
    if "chess_fusion" in model_data and isinstance(model_data["chess_fusion"], dict):
        # Handle nested LoRA config inside chess_fusion (user convenience)
        if "lora" in model_data["chess_fusion"]:
            lora_data = model_data["chess_fusion"].pop("lora")
            if isinstance(lora_data, dict):
                model_data["lora"] = LoRAConfig(**lora_data)
        # Convert xattn_layers from yaml list to python list
        model_data["chess_fusion"] = ChessFusionConfig(**model_data["chess_fusion"])
    
    if model_data:
        data["model"] = ModelConfig(**model_data)

    # Handle top-level profiling section
    if "profiling" in data and isinstance(data["profiling"], dict):
        data["profiling"] = ProfilingConfig(**data["profiling"])
        
    return TrainingConfig(**data)


