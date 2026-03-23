"""
Maia Model Integration for Policy Training and LLM Embedding Extraction.

This module provides:
1. MaiaPolicyModel - Wrapper for pretrained Maia2 with from/to policy heads
2. MaiaEmbeddingExtractor - Extracts hidden states for LLM integration
3. extract_maia_features - Board tensor extraction compatible with Maia2
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import chess
import numpy as np

# Add src to path for imports
# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from maia2 import utils as maia_utils
except ImportError:
    maia_utils = None

from training.chess_adapter import ENGINEERED_FEATURE_DIM

# =============================================================================
# Move Vocabulary and Mapping
# =============================================================================

class MaiaMoveMapping:
    """Singleton to handle Maia2's move vocabulary."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MaiaMoveMapping, cls).__new__(cls)
            cls._instance._init_vocab()
        return cls._instance
    
    def _init_vocab(self):
        if maia_utils is None:
            raise ImportError("maia2 library required for move mapping")
            
        self.vocab = maia_utils.get_all_possible_moves()
        self.move_to_idx = {m: i for i, m in enumerate(self.vocab)}
        self.idx_to_move = {i: m for i, m in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)
        
    def encode(self, move_uci: str) -> int:
        """Convert UCI string (e.g., 'e2e4') to index."""
        return self.move_to_idx.get(move_uci, -1)
        
    def decode(self, idx: int) -> str:
        """Convert index to UCI string."""
        return self.idx_to_move.get(idx, None)

# Global instance
_MAIA_MAPPING = None

def get_maia_mapping() -> MaiaMoveMapping:
    global _MAIA_MAPPING
    if _MAIA_MAPPING is None:
        _MAIA_MAPPING = MaiaMoveMapping()
    return _MAIA_MAPPING


def unmirror_policy_move(uci: str, side_to_move) -> str:
    """Un-mirror a perspective-relative UCI move back to absolute coordinates.

    Maia2's policy head outputs moves in perspective-relative space (always
    from White's viewpoint).  When Black is to move, the predicted UCI string
    has its ranks mirrored (rank r -> 7-r in 0-indexed, i.e. '1'<->'8', '2'<->'7', …).
    This function reverses that transformation.

    For White-to-move positions the move is returned unchanged.

    Args:
        uci: UCI move string in perspective-relative space (e.g. 'd1d3')
        side_to_move: True / 1 for White, False / 0 for Black.
                      Also accepts torch.Tensor scalars.

    Returns:
        UCI move string in absolute board coordinates.
    """
    # Normalise side_to_move to bool
    import torch
    if isinstance(side_to_move, torch.Tensor):
        is_white = bool(side_to_move.item())
    else:
        is_white = bool(side_to_move)

    if is_white:
        return uci

    if maia_utils is not None:
        return maia_utils.mirror_move(uci)

    # Fallback manual mirror: flip rank digits ('1'<->'8', '2'<->'7', …)
    def flip_rank(c):
        if c.isdigit() and '1' <= c <= '8':
            return str(9 - int(c))
        return c
    return ''.join(flip_rank(c) for c in uci)


# =============================================================================
# ELO Mapping (matches maia2's create_elo_dict / map_to_category)
# =============================================================================

def elo_to_category(elo: int) -> int:
    """
    Map raw ELO rating to category index for Maia2 embedding.
    
    Maia2 uses ELO buckets:
        0: <1100
        1: 1100-1199
        2: 1200-1299
        ...
        9: 1900-1999
        10: >=2000
    
    Args:
        elo: Raw ELO rating (e.g., 1500)
        
    Returns:
        Category index (0-10)
    """
    if elo < 1100:
        return 0
    elif elo >= 2000:
        return 10
    else:
        # 1100-1999 maps to indices 1-9
        return (elo - 1100) // 100 + 1


# =============================================================================
# Feature Extraction (Maia2 Compatible)
# =============================================================================

def board_to_tensor(board: chess.Board) -> torch.Tensor:
    """
    Convert a chess.Board to an 18-channel 8x8 tensor.
    
    This matches maia2's internal board_to_tensor function.
    
    Channels (18 total):
        0-5:   White pieces (P, N, B, R, Q, K)
        6-11:  Black pieces (P, N, B, R, Q, K)
        12:    Side to move (1 if white, 0 if black)
        13:    White kingside castling rights
        14:    White queenside castling rights
        15:    Black kingside castling rights
        16:    Black queenside castling rights
        17:    En passant square (if any)
    """
    tensor = torch.zeros(18, 8, 8, dtype=torch.float32)
    
    # Piece channels
    piece_map = board.piece_map()
    for square, piece in piece_map.items():
        rank = square // 8
        file = square % 8
        
        # Channel index: piece_type - 1 for white, piece_type + 5 for black
        piece_idx = piece.piece_type - 1  # P=0, N=1, B=2, R=3, Q=4, K=5
        if piece.color == chess.WHITE:
            channel = piece_idx
        else:
            channel = piece_idx + 6
            
        tensor[channel, rank, file] = 1.0
    
    # Side to move (channel 12)
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    
    # Castling rights (channels 13-16)
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0
    
    # En passant (channel 17)
    if board.ep_square is not None:
        ep_rank = board.ep_square // 8
        ep_file = board.ep_square % 8
        tensor[17, ep_rank, ep_file] = 1.0
    
    return tensor


def extract_maia_features(fen: str) -> torch.Tensor:
    """
    Extract features from FEN string in Maia2-compatible format.
    
    Maia2 always processes positions from White's perspective.
    If it's Black to move, we mirror the board.
    
    Args:
        fen: FEN string of the position
        
    Returns:
        Tensor of shape (18, 8, 8)
    """
    board = chess.Board(fen)
    
    # Maia2 always sees from White's perspective
    if board.turn == chess.BLACK:
        board = board.mirror()
    
    return board_to_tensor(board)


# =============================================================================
# Maia Policy Model
# =============================================================================

class MaiaPolicyModel(nn.Module):
    """
    Wrapper around pretrained Maia2 model for policy training.
    
    Uses Maia2's native policy head (fc_1) which outputs 1880 logits
    corresponding to valid UCI moves.
    """
    
    def __init__(self, config=None, model_type: str = "rapid", elo_self: int = 1500, elo_oppo: int = 1500):
        super().__init__()
        
        # Config extraction
        if config and hasattr(config, 'maia'):
            maia_cfg = config.maia
            model_type = getattr(maia_cfg, 'model_type', model_type)
            elo_self = getattr(maia_cfg, 'elo_self', elo_self)
            elo_oppo = getattr(maia_cfg, 'elo_oppo', elo_oppo)
            freeze_backbone = getattr(maia_cfg, 'freeze_backbone', False)
        else:
            freeze_backbone = False
        
        self.model_type = model_type
        self._raw_elo_self = elo_self  # Store raw ELO for logging
        self._raw_elo_oppo = elo_oppo
        # Convert to category indices for embedding
        self.elo_self = elo_to_category(elo_self)
        self.elo_oppo = elo_to_category(elo_oppo)
        
        # Load pretrained Maia2
        print(f"Loading Maia2 model (type={model_type})...")
        from maia2 import model as maia_model
        
        # [DDP Fix] Monkeypatch os.makedirs inside maia2 to force exist_ok=True
        # This prevents race conditions where multiple DDP processes try to create the same dir simultaneousy
        import os
        original_makedirs = os.makedirs
        
        def safe_makedirs(name, mode=0o777, exist_ok=False):
            # Force exist_ok=True to prevent FileExistsError race condition
            return original_makedirs(name, mode, exist_ok=True)
            
        try:
            # Apply monkeypatch temporarily
            os.makedirs = safe_makedirs
            self.maia = maia_model.from_pretrained(type=model_type, device="cpu")
        except Exception as e:
            print(f"Error loading Maia2 model: {e}")
            raise e
        finally:
            # Restore original os.makedirs
            os.makedirs = original_makedirs
            
        # Ensure model is on CPU (checkpoint may have been saved on GPU)
        # Ensure model is on CPU (checkpoint may have been saved on GPU)
        self.maia = self.maia.cpu()
        
        # Get the dimension from Maia's transformer output
        # Maia2 uses dim_vit = 1024 by default
        self.hidden_dim = self.maia.cfg.dim_vit  # 1024
        
        
        # Policy heads
        # We use Maia2's original fc_1 head (1880 outputs)
        # It is already part of self.maia
        
        # For compatibility with earlier code, we might want to ensure it's exposed/used correctly
        # The forward method will return the single (B, 1880) tensor
        
        # We don't need from_head/to_head anymore
        self.output_dim = 1880
        
        # Optionally freeze the backbone (only train heads)
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_maia_backbone()
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.parameters())
            print(f"Backbone frozen: {trainable:,} / {total:,} params trainable ({100*trainable/total:.1f}%)")
        
        print(f"MaiaPolicyModel initialized (hidden_dim={self.hidden_dim}, elo={self._raw_elo_self}/{self._raw_elo_oppo} -> idx {self.elo_self}/{self.elo_oppo}, frozen={freeze_backbone})")
    
    def _freeze_maia_backbone(self):
        """
        Freeze parameters in the Maia backbone (CNN + Transformer),
        but keep the policy head (fc_1) trainable.
        """
        # Freeze everything first
        for param in self.maia.parameters():
            param.requires_grad = False
            
        # Unfreeze policy head (fc_1)
        for param in self.maia.fc_1.parameters():
            param.requires_grad = True
            
        # We might also want to unfreeze the final layer norm to allow adaptation
        for param in self.maia.last_ln.parameters():
            param.requires_grad = True
    
    def get_cnn_output(self, boards: torch.Tensor) -> torch.Tensor:
        """
        Run only the CNN backbone, without patch embedding or transformer.
        
        Useful when only CNN hook outputs are needed (e.g., when
        use_transformer_taps=False). This avoids storing transformer
        activations in the computation graph, saving significant VRAM.
        
        Args:
            boards: (B, 18, 8, 8) board tensors
            
        Returns:
            (B, 8, 8, 8) CNN output after conv_last
        """
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.maia.cfg.input_channels, 8, 8)
        return self.maia.chess_cnn(boards)

    def get_policy_from_cnn_output(
        self,
        cnn_embs: torch.Tensor,
        elo_self: torch.Tensor = None,
        elo_oppo: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute policy logits starting from CNN output, skipping CNN re-run.
        
        This avoids a redundant full backbone forward pass when the CNN
        has already been run (e.g., for multi-scale feature extraction).
        Runs: patch_embed -> pos_embed -> transformer -> mean -> LN -> fc_1.
        
        Args:
            cnn_embs: (B, 8, 8, 8) output from chess_cnn (should be detached)
            elo_self: (B,) or None
            elo_oppo: (B,) or None
            
        Returns:
            (B, 1880) policy logits
        """
        batch_size = cnn_embs.size(0)
        device = cnn_embs.device
        
        if elo_self is None:
            elo_self = torch.full((batch_size,), self.elo_self, dtype=torch.long, device=device)
        if elo_oppo is None:
            elo_oppo = torch.full((batch_size,), self.elo_oppo, dtype=torch.long, device=device)
        
        embs = cnn_embs.view(batch_size, cnn_embs.size(1), 8 * 8)
        x = self.maia.to_patch_embedding(embs)
        x += self.maia.pos_embedding
        x = self.maia.dropout(x)
        
        elos_emb_self = self.maia.elo_embedding(elo_self)
        elos_emb_oppo = self.maia.elo_embedding(elo_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        
        x = self.maia.transformer(x, elos_emb)
        x = x.mean(dim=1)
        x = self.maia.last_ln(x)
        return self.maia.fc_1(x)

    def get_transformer_output(self, boards: torch.Tensor, elo_self: torch.Tensor, elo_oppo: torch.Tensor, return_sequence: bool = False) -> torch.Tensor:
        """
        Get the transformer output (before Maia's heads).
        
        This replicates Maia2's forward pass up to the last layer norm.
        
        Args:
            boards: (B, 18, 8, 8) board tensors
            elo_self: (B,) self ELO category indices (0-10) or None
            elo_oppo: (B,) opponent ELO category indices (0-10) or None
            return_sequence: If True, returns (B, 256, 1024) unpooled sequence.
                             If False, returns (B, 1024) pooled vector.
            
        Returns:
            (B, 1024) or (B, 256, 1024) transformer output
        """
        batch_size = boards.size(0)
        device = boards.device
        
        # Default ELO values if not provided
        if elo_self is None:
            elo_self = torch.full((batch_size,), self.elo_self, dtype=torch.long, device=device)
        if elo_oppo is None:
            elo_oppo = torch.full((batch_size,), self.elo_oppo, dtype=torch.long, device=device)
            
        boards = boards.view(batch_size, self.maia.cfg.input_channels, 8, 8)
        
        # CNN backbone
        embs = self.maia.chess_cnn(boards)
        embs = embs.view(batch_size, embs.size(1), 8 * 8)
        
        # Patch embedding + position embedding
        x = self.maia.to_patch_embedding(embs)
        x += self.maia.pos_embedding
        x = self.maia.dropout(x)
        
        # ELO embedding
        elos_emb_self = self.maia.elo_embedding(elo_self)
        elos_emb_oppo = self.maia.elo_embedding(elo_oppo)
        elos_emb = torch.cat((elos_emb_self, elos_emb_oppo), dim=1)
        
        # Transformer
        # x is [B, 256, 1024]
        x = self.maia.transformer(x, elos_emb)
        
        if return_sequence:
            # Apply LayerNorm to the sequence
            x = self.maia.last_ln(x)
            return x
        else:
            # Mean pooling + LayerNorm
            x = x.mean(dim=1)
            x = self.maia.last_ln(x)
            return x
    
    def forward(self, boards: torch.Tensor, elo_self: torch.Tensor = None, elo_oppo: torch.Tensor = None):
        """
        Forward pass for policy prediction.
        
        Args:
            boards: (B, 18, 8, 8) or (B, 18*8*8) board tensors
            elo_self: (B,) or None - uses default if None
            elo_oppo: (B,) or None - uses default if None
            
        Returns:
            logits: (B, 1880) raw logits for each move in vocabulary
        """
        # Get transformer output (handles defaults for ELOs)
        hidden = self.get_transformer_output(boards, elo_self, elo_oppo)
        
        # Use Maia2's original policy head
        logits = self.maia.fc_1(hidden)
        
        return logits


# =============================================================================
# Maia Embedding Extractor (for LLM Integration)
# =============================================================================

class MaiaEmbeddingExtractor(nn.Module):
    """
    Extracts hidden states from Maia2's CNN backbone for LLM integration.
    
    Hooks into the final CNN block to get (B, 256, 8, 8) features,
    then projects to LLM dimension.
    """
    
    def __init__(self, config=None, model_type: str = "rapid", llm_dim: int = 2048,
                 elo_self: int = 1500, elo_oppo: int = 1500):
        super().__init__()
        
        # Config extraction
        if config and hasattr(config, 'maia'):
            maia_cfg = config.maia
            model_type = getattr(maia_cfg, 'model_type', model_type)
            llm_dim = getattr(maia_cfg, 'llm_projection_dim', llm_dim)
            elo_self = getattr(maia_cfg, 'elo_self', elo_self)
            elo_oppo = getattr(maia_cfg, 'elo_oppo', elo_oppo)
        
        self.model_type = model_type
        self.llm_dim = llm_dim
        self.elo_self = elo_self
        self.elo_oppo = elo_oppo
        
        # Load pretrained Maia2
        print(f"Loading Maia2 model for embedding extraction...")
        from maia2 import model as maia_model
        self.maia = maia_model.from_pretrained(type=model_type, device="cpu")
        
        # CNN hidden dimension (256 by default in Maia2)
        self.cnn_dim = self.maia.cfg.dim_cnn  # 256
        
        # Projection from CNN output [B, 256, 8, 8] -> [B, 64, llm_dim]
        # First flatten spatial dims partially: [B, 256, 8, 8] -> [B, 8*8, 256]
        # Then project channels: [B, 64, 256] -> [B, 64, llm_dim]
        self.projection = nn.Sequential(
            nn.Linear(self.cnn_dim, llm_dim),
            nn.LayerNorm(llm_dim),
            nn.GELU(),
        )
        
        # Hook storage
        self._hidden_states = None
        self._register_hooks()
        
        print(f"MaiaEmbeddingExtractor initialized (cnn_dim={self.cnn_dim}, llm_dim={llm_dim})")
    
    def _register_hooks(self):
        """Register forward hook on the last CNN layer."""
        def hook_fn(module, input, output):
            self._hidden_states = output
        
        # Hook into the end of CNN layers (before conv_last projection)
        self.maia.chess_cnn.layers.register_forward_hook(hook_fn)
    
    def extract_cnn_features(self, boards: torch.Tensor) -> torch.Tensor:
        """
        Extract CNN hidden states.
        
        Args:
            boards: (B, 18, 8, 8) board tensors
            
        Returns:
            (B, 256, 8, 8) hidden states from last CNN block
        """
        batch_size = boards.size(0)
        boards = boards.view(batch_size, self.maia.cfg.input_channels, 8, 8)
        
        # Run CNN (hooks will capture intermediate states)
        _ = self.maia.chess_cnn(boards)
        
        return self._hidden_states
    
    def forward(self, boards: torch.Tensor) -> torch.Tensor:
        """
        Extract and project hidden states for LLM integration.
        
        Args:
            boards: (B, 18, 8, 8) board tensors
            
        Returns:
            (B, 64, llm_dim) projected embeddings for LLM
        """
        # Extract CNN features: (B, 256, 8, 8)
        hidden = self.extract_cnn_features(boards)
        
        # Reshape: (B, 256, 8, 8) -> (B, 8, 8, 256) -> (B, 64, 256)
        batch_size = hidden.size(0)
        hidden = hidden.permute(0, 2, 3, 1)  # (B, 8, 8, 256)
        hidden = hidden.reshape(batch_size, 64, self.cnn_dim)  # (B, 64, 256)
        
        # Project to LLM dimension: (B, 64, 256) -> (B, 64, llm_dim)
        projected = self.projection(hidden)
        
        return projected



# =============================================================================
# Maia LLM Adapter (Integration Layer)
# =============================================================================

class MaiaLLMAdapter(nn.Module):
    """
    Adapter to project Maia2 features into LLM embedding space.
    
    Supports two modes:
    1. 'minimal': Projects pooled transformer output (1024) -> (1, llm_dim)
    2. 'perceiver': Uses Perceiver-like cross-attention to resample 
                    (256, 1024) -> (k, llm_dim)
    """
    
    def __init__(self, config=None):
        super().__init__()
        
        # Default config values
        self.mode = "minimal"
        self.llm_dim = 2048
        self.maia_dim = 1024
        self.num_latents = 8   # 'k' for perceiver
        self.num_cross_attn_layers = 1 # 'n' for perceiver
        self.freeze_cnn = False
        self.freeze_transformer = False
        self.freeze_perceiver = False
        self.use_main_engineered_concat = False
        
        # Load config
        if config and hasattr(config, 'maia'):
            cfg = config.maia
            self.mode = getattr(cfg, 'adapter_mode', self.mode)
            self.llm_dim = getattr(cfg, 'llm_projection_dim', self.llm_dim)
            self.num_latents = getattr(cfg, 'num_latents', self.num_latents)
            self.num_cross_attn_layers = getattr(cfg, 'perceiver_depth', self.num_cross_attn_layers)
            self.freeze_cnn = getattr(cfg, 'freeze_cnn', self.freeze_cnn)
            self.freeze_transformer = getattr(cfg, 'freeze_transformer', self.freeze_transformer)
            self.freeze_perceiver = getattr(cfg, 'freeze_perceiver', self.freeze_perceiver)
            self.use_main_engineered_concat = getattr(cfg, 'use_main_engineered_concat', self.use_main_engineered_concat)
            # Legacy config support
            if hasattr(cfg, 'freeze_backbone') and cfg.freeze_backbone:
                self.freeze_cnn = True
                self.freeze_transformer = True
        
        # Initialize Maia Policy Model (Backbone)
        self.backbone = MaiaPolicyModel(config)
        
        # Freeze all Maia2 prediction heads — only get_transformer_output is
        # used in the LLM adapter path.  Leaving these trainable causes DDP to
        # fail without find_unused_parameters=True.
        # Heads: fc_1 (policy), fc_2 (side info), fc_3/fc_3_1 (value)
        for head_name in ("fc_1", "fc_2", "fc_3", "fc_3_1"):
            head = getattr(self.backbone.maia, head_name, None)
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = False
        
        # Apply freezing logic
        if self.freeze_cnn or self.freeze_transformer:
            self._apply_freezing()
            
        print(f"MaiaLLMAdapter initialized (mode={self.mode}, llm_dim={self.llm_dim})")
        
        # Compute perceiver internal dimension early (needed for side embedding)
        # perceiver_internal_dim overrides the default; None = legacy behavior
        explicit_dim = getattr(cfg, 'perceiver_internal_dim', None) if cfg else None
        if self.mode == "perceiver" and self.use_main_engineered_concat:
            self.engineered_dim = ENGINEERED_FEATURE_DIM
            self.perceiver_dim = explicit_dim if explicit_dim is not None else self.llm_dim
            self.latent_base_dim = self.perceiver_dim - self.engineered_dim
        elif self.mode == "perceiver":
            self.perceiver_dim = explicit_dim if explicit_dim is not None else self.maia_dim
            self.latent_base_dim = self.perceiver_dim
            self.engineered_dim = 0
        else:
            self.perceiver_dim = explicit_dim if explicit_dim is not None else self.maia_dim

        # Side to move embedding (0=Black, 1=White)
        # Added to features before projection/attention
        side_dim = self.perceiver_dim if self.mode == "perceiver" else self.maia_dim
        self.side_embedding = nn.Embedding(2, side_dim)
        
        # Mode-specific components
        if self.mode == "minimal":
            self.use_mlp_projections = getattr(cfg, 'use_mlp_projections', False)
            
            if self.use_mlp_projections:
                print(f"  [Minimal] Using MLP projection (Linear -> GELU -> Linear)")
                self.projector = nn.Sequential(
                    nn.Linear(self.maia_dim, self.llm_dim),
                    nn.GELU(),
                    nn.Linear(self.llm_dim, self.llm_dim),
                    nn.LayerNorm(self.llm_dim)
                )
            else:
                # Linear projection: 1024 -> llm_dim
                self.projector = nn.Sequential(
                    nn.Linear(self.maia_dim, self.llm_dim),
                    nn.LayerNorm(self.llm_dim)
                )
            
        elif self.mode == "perceiver":
            if self.use_main_engineered_concat:
                self.num_latents = 64
            print(f"  Perceiver: {self.num_latents} latents, {self.num_cross_attn_layers} layers, dim={self.perceiver_dim}")
            if self.use_main_engineered_concat:
                print(f"    Latent base dim={self.latent_base_dim}, + engineered {self.engineered_dim} -> {self.perceiver_dim}")

            # Latent queries: (1, k, latent_base_dim) - learned
            # In concat mode, engineered features are cat'd at runtime -> perceiver_dim
            self.latents = nn.Parameter(torch.randn(1, self.num_latents, self.latent_base_dim))
            
            # Check for MLP Projections
            self.use_mlp_projections = getattr(cfg, 'use_mlp_projections', False)
            if self.use_mlp_projections:
                 print("  [Perceiver] Using MLP projections for Q/K/V generation")
            
            # Self-Attention Layers (latents attend to each other)
            self.self_attn_layers = nn.ModuleList([
                nn.MultiheadAttention(embed_dim=self.perceiver_dim, num_heads=8, batch_first=True)
                for _ in range(self.num_cross_attn_layers)
            ])
            self.self_attn_norms = nn.ModuleList([
                nn.LayerNorm(self.perceiver_dim) for _ in range(self.num_cross_attn_layers)
            ])

            # Cross-Attention Layers (latents attend to Maia context)
            # Q dim = perceiver_dim, K/V dim = maia_dim (1024)
            self.cross_attn_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=self.perceiver_dim, num_heads=8,
                    kdim=self.maia_dim, vdim=self.maia_dim,
                    batch_first=True
                )
                for _ in range(self.num_cross_attn_layers)
            ])
            self.cross_attn_norms = nn.ModuleList([
                nn.LayerNorm(self.perceiver_dim) for _ in range(self.num_cross_attn_layers)
            ])
            
            # Optional MLP Projectors for Q/K/V
            if self.use_mlp_projections:
                # Self-attention MLPs (all perceiver_dim)
                self.self_q_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.perceiver_dim, self.perceiver_dim),
                        nn.GELU(),
                        nn.Linear(self.perceiver_dim, self.perceiver_dim)
                    ) for _ in range(self.num_cross_attn_layers)
                ])
                self.self_kv_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.perceiver_dim, self.perceiver_dim),
                        nn.GELU(),
                        nn.Linear(self.perceiver_dim, self.perceiver_dim)
                    ) for _ in range(self.num_cross_attn_layers)
                ])
                # Cross-attention MLPs (Q=perceiver_dim, KV=maia_dim)
                self.cross_q_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.perceiver_dim, self.perceiver_dim),
                        nn.GELU(),
                        nn.Linear(self.perceiver_dim, self.perceiver_dim)
                    ) for _ in range(self.num_cross_attn_layers)
                ])
                self.cross_kv_mlps = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.maia_dim, self.maia_dim),
                        nn.GELU(),
                        nn.Linear(self.maia_dim, self.maia_dim)
                    ) for _ in range(self.num_cross_attn_layers)
                ])
            else:
                 self.self_q_mlps = None
                 self.self_kv_mlps = None
                 self.cross_q_mlps = None
                 self.cross_kv_mlps = None

            self.ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(self.perceiver_dim, self.perceiver_dim * 4),
                    nn.GELU(),
                    nn.Linear(self.perceiver_dim * 4, self.perceiver_dim)
                ) for _ in range(self.num_cross_attn_layers)
            ])
            self.ffn_norms = nn.ModuleList([
                nn.LayerNorm(self.perceiver_dim) for _ in range(self.num_cross_attn_layers)
            ])
            
            # Final LayerNorm (Pre-LN standard)
            self.perceiver_ln_f = nn.LayerNorm(self.perceiver_dim)
            
            # Final projection to LLM dim
            self.projector = nn.Sequential(
                nn.Linear(self.perceiver_dim, self.llm_dim),
                nn.LayerNorm(self.llm_dim)
            )
            
        else:
            raise ValueError(f"Unknown adapter mode: {self.mode}")

        if self.mode == "perceiver" and self.freeze_perceiver:
            self._freeze_perceiver_params()
            print("  Perceiver adapter parameters frozen")
            
    def _apply_freezing(self):
        """Freeze parts of the backbone based on config."""
        maia = self.backbone.maia
        
        # Recursively unfreeze everything first to be safe
        for p in self.backbone.parameters():
            p.requires_grad = True
            
        if self.freeze_cnn:
            self._freeze_cnn_params()
            
        if self.freeze_transformer:
            self._freeze_transformer_params()

    def _freeze_cnn_params(self):
        """Freeze CNN backbone parameters."""
        maia = self.backbone.maia
        for p in maia.chess_cnn.parameters():
            p.requires_grad = False
        for p in maia.to_patch_embedding.parameters():
            p.requires_grad = False
        maia.pos_embedding.requires_grad = False

    def _unfreeze_cnn_params(self):
        """Unfreeze CNN backbone parameters."""
        maia = self.backbone.maia
        for p in maia.chess_cnn.parameters():
            p.requires_grad = True
        for p in maia.to_patch_embedding.parameters():
            p.requires_grad = True
        maia.pos_embedding.requires_grad = True

    def _freeze_transformer_params(self):
        """Freeze Transformer backbone parameters."""
        maia = self.backbone.maia
        for p in maia.transformer.parameters():
            p.requires_grad = False
        for p in maia.elo_embedding.parameters():
            p.requires_grad = False
        for p in maia.last_ln.parameters():
            p.requires_grad = False

    def _unfreeze_transformer_params(self):
        """Unfreeze Transformer backbone parameters."""
        maia = self.backbone.maia
        for p in maia.transformer.parameters():
            p.requires_grad = True
        for p in maia.elo_embedding.parameters():
            p.requires_grad = True
        for p in maia.last_ln.parameters():
            p.requires_grad = True

    def _freeze_perceiver_params(self):
        """Freeze Perceiver adapter parameters (maia mode perceiver branch)."""
        if self.mode != "perceiver":
            return
        perceiver_modules = [
            self.side_embedding,
            self.self_attn_layers,
            self.self_attn_norms,
            self.cross_attn_layers,
            self.cross_attn_norms,
            self.ffns,
            self.ffn_norms,
            self.perceiver_ln_f,
            self.projector,
        ]
        for module in perceiver_modules:
            for p in module.parameters():
                p.requires_grad = False
        self.latents.requires_grad = False

    def _unfreeze_perceiver_params(self):
        """Unfreeze Perceiver adapter parameters (maia mode perceiver branch)."""
        if self.mode != "perceiver":
            return
        perceiver_modules = [
            self.side_embedding,
            self.self_attn_layers,
            self.self_attn_norms,
            self.cross_attn_layers,
            self.cross_attn_norms,
            self.ffns,
            self.ffn_norms,
            self.perceiver_ln_f,
            self.projector,
        ]
        for module in perceiver_modules:
            for p in module.parameters():
                p.requires_grad = True
        self.latents.requires_grad = True

            
    def get_num_prefix_tokens(self) -> int:
        """Return the number of tokens prepended to the LLM input."""
        if self.mode == "minimal":
            return 1
        elif self.mode == "perceiver":
            return self.num_latents
        else:
            return 0

    def forward(
        self,
        boards: torch.Tensor,
        elo_self: torch.Tensor = None,
        elo_oppo: torch.Tensor = None,
        **kwargs,
    ):
        """
        Forward pass to get LLM embeddings.
        
        Args:
            boards: (B, 18, 8, 8)
            elo_self: (B,)
            elo_oppo: (B,)
            **kwargs: Can include 'side_to_move' (Tensor or bool/int list)
            
        Returns:
            (B, num_tokens, llm_dim)
                - minimal: num_tokens=1
                - perceiver: num_tokens=k
        """
        batch_size = boards.size(0)
        device = boards.device
        
        # DEBUG: Verify device placement
        if not hasattr(self, "_device_checked"):
            print(f"[DEBUG] MaiaLLMAdapter input device: {boards.device}")
            if self.mode == "perceiver":
                print(f"[DEBUG] Latents device: {self.latents.device}")
            self._device_checked = True
            
        # Handle Side to Move Embedding
        # Default to White (1) if not provided
        side_to_move = kwargs.get('side_to_move', None)
        if side_to_move is None:
             side_indices = torch.ones(batch_size, dtype=torch.long, device=device)
        else:
            # Handle list/bool/tensor
            if isinstance(side_to_move, torch.Tensor):
                side_indices = side_to_move.long().to(device)
            else:
                side_indices = torch.tensor([1 if s else 0 for s in side_to_move], dtype=torch.long, device=device)
        
        # Get side embeddings: (B, 1024)
        side_embeds = self.side_embedding(side_indices)

        if self.mode == "minimal":
            # Get pooled output: (B, 1024)
            features = self.backbone.get_transformer_output(
                boards, elo_self, elo_oppo, return_sequence=False
            ) # (B, 1024)
            
            # Add side embedding
            features = features + side_embeds
            
            # Project: (B, 1024) -> (B, llm_dim)
            projected = self.projector(features)
            
            # Add sequence dimension: (B, 1, llm_dim)
            return projected.unsqueeze(1)
            
        elif self.mode == "perceiver":
            # Get Maia sequence output (context for cross-attn): (B, 256, 1024)
            context = self.backbone.get_transformer_output(
                boards, elo_self, elo_oppo, return_sequence=True
            )

            # Expand latents: (1, k, latent_base_dim) -> (B, k, latent_base_dim)
            x = self.latents.expand(batch_size, -1, -1)

            # Concatenate engineered features to latents (not to context)
            if self.use_main_engineered_concat:
                engineered_features = kwargs.get('engineered_features', None)
                if engineered_features is None:
                    raise ValueError("engineered_features must be provided when Maia perceiver uses main engineered concat")
                engineered_features = engineered_features.to(device)
                # (B, 64, 1843) cat (B, 64, 205) -> (B, 64, 2048)
                x = torch.cat([x, engineered_features], dim=-1)
            
            # Add side embedding to latents (broadcast across k)
            # side_embeds: (B, perceiver_dim) -> (B, 1, perceiver_dim)
            x = x + side_embeds.unsqueeze(1)
            
            # Perceiver Layers (Pre-LN): Self-Attn -> Cross-Attn -> FFN
            for i in range(self.num_cross_attn_layers):
                # 1. Self-Attention (latents attend to each other)
                x_norm = self.self_attn_norms[i](x)
                if self.use_mlp_projections:
                    sa_q = self.self_q_mlps[i](x_norm)
                    sa_kv = self.self_kv_mlps[i](x_norm)
                else:
                    sa_q = x_norm
                    sa_kv = x_norm
                sa_out, _ = self.self_attn_layers[i](query=sa_q, key=sa_kv, value=sa_kv)
                x = x + sa_out  # Residual

                # 2. Cross-Attention (latents attend to Maia context)
                x_norm = self.cross_attn_norms[i](x)
                if self.use_mlp_projections:
                    ca_q = self.cross_q_mlps[i](x_norm)
                    ca_kv = self.cross_kv_mlps[i](context)
                else:
                    ca_q = x_norm
                    ca_kv = context
                ca_out, _ = self.cross_attn_layers[i](query=ca_q, key=ca_kv, value=ca_kv)
                x = x + ca_out  # Residual
                
                # 3. FFN
                x_norm_ffn = self.ffn_norms[i](x)
                ffn_out = self.ffns[i](x_norm_ffn)
                x = x + ffn_out  # Residual
                
            # Final LayerNorm
            x = self.perceiver_ln_f(x)
                
            # Final Projection: (B, k, perceiver_dim) -> (B, k, llm_dim)
            output = self.projector(x)
            
            return output


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Maia Model Components")
    print("=" * 60)
    
    # Test feature extraction
    print("\n1. Testing feature extraction...")
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    try:
        features = extract_maia_features(fen)
        print(f"   Feature shape: {features.shape}")
        assert features.shape == (18, 8, 8), f"Expected (18, 8, 8), got {features.shape}"
        print("   [PASS] Feature extraction")
    except Exception as e:
        print(f"   [FAIL] Feature extraction: {e}")
    
    # Test policy model
    print("\n2. Testing MaiaPolicyModel...")
    try:
        model = MaiaPolicyModel()
        batch_size = 2
        dummy_boards = torch.randn(batch_size, 18, 8, 8)
        
        # Test forward (policy)
        logits = model(dummy_boards)
        print(f"   Logits shape: {logits.shape}")
        # Maia logits should be [B, 1880] approximately (Maia2 vocabs might differ)
        # Checking against shape of model.maia.fc_1
        print("   [PASS] MaiaPolicyModel forward pass")
        
        # Test unpooled sequence
        seq = model.get_transformer_output(dummy_boards, None, None, return_sequence=True)
        print(f"   Unpooled sequence shape: {seq.shape}")
        assert seq.shape == (batch_size, 256, 1024), f"Expected (B, 256, 1024), got {seq.shape}"
        print("   [PASS] Unpooled sequence extraction")
        
    except Exception as e:
        print(f"   [FAIL] MaiaPolicyModel: {e}")
        import traceback
        traceback.print_exc()

    # Test MaiaLLMAdapter - Minimal
    print("\n3. Testing MaiaLLMAdapter (Minimal)...")
    class ConfigMock:
        class maia:
            adapter_mode = "minimal"
            llm_projection_dim = 2048
            freeze_cnn = True
    
    try:
        adapter_min = MaiaLLMAdapter(config=ConfigMock())
        out_min = adapter_min(dummy_boards)
        print(f"   Output shape: {out_min.shape}")
        assert out_min.shape == (batch_size, 1, 2048)
        print("   [PASS] Minimal mode adapter")
    except Exception as e:
        print(f"   [FAIL] Minimal adapter: {e}")
        import traceback
        traceback.print_exc()

    # Test MaiaLLMAdapter - Perceiver
    print("\n4. Testing MaiaLLMAdapter (Perceiver)...")
    class ConfigMockPerc:
        class maia:
            adapter_mode = "perceiver"
            llm_projection_dim = 2048
            num_latents = 4
            perceiver_depth = 1
    
    try:
        adapter_perc = MaiaLLMAdapter(config=ConfigMockPerc())
        out_perc = adapter_perc(dummy_boards)
        print(f"   Output shape: {out_perc.shape}")
        assert out_perc.shape == (batch_size, 4, 2048)
        print("   [PASS] Perceiver mode adapter")
    except Exception as e:
        print(f"   [FAIL] Perceiver adapter: {e}")
        import traceback
        traceback.print_exc()
        
    print("\n" + "=" * 60)
    print("Tests finished.")
    print("=" * 60)
