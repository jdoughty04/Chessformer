"""
Chess Position Adapter Module

Projects LC0 transformer hidden states into LLM embedding space.
Also supports "Engineered" features (manual feature engineering) and "Hybrid" modes.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, List, Dict
import chess
import numpy as np

# ============================================================================
# Legacy / Pure LC0 Adapter (from original code)
# ============================================================================

class ChessPositionAdapter(nn.Module):
    """
    Projects LC0 hidden states to LLM embedding space.
    
    Takes hidden states from multiple LC0 transformer layers and produces
    65 embeddings (1 side token + 64 square embeddings) that can be used as
    prefix tokens for the language model.
    """
    
    def __init__(
        self,
        lc0_dim: int = 768,           # LC0 BT3/BT4 embedding dimension
        projection_dim: int = 128,    # Per-layer projection dimension
        llm_dim: int = 2048,          # TinyLlama hidden_size
        num_layers: int = 4,          # Layers: 4, 8, 12, 15
        num_squares: int = 64,
        dropout: float = 0.1,
        use_simple_projection: bool = False,
    ):
        super().__init__()
        
        # Support initializing from a config object if passed
        if hasattr(lc0_dim, 'lc0_dim'):
             config = lc0_dim
             lc0_dim = getattr(config, 'lc0_dim', 768)
             # Defaults if config is passed but missing fields
             projection_dim = getattr(config, 'projection_dim', 128)
             llm_dim = 2048 # Usually fixed for model
             num_layers = getattr(config, 'num_layers', 4)

        self.lc0_dim = lc0_dim
        self.projection_dim = projection_dim
        self.llm_dim = llm_dim
        self.num_layers = num_layers
        self.num_squares = num_squares
        self.use_simple_projection = use_simple_projection
        
        # Layer keys for LC0 hidden states
        self.layer_keys = ["layer_4", "layer_8", "layer_12", "layer_15"]
        
        # Pre-compute mirror indices for Black-to-move flipping
        mirror_indices = torch.tensor([chess.square_mirror(sq) for sq in range(64)], dtype=torch.long)
        self.register_buffer('mirror_indices', mirror_indices)
        
        # Learnable side-to-move embeddings
        self.white_to_move_embed = nn.Parameter(torch.randn(1, llm_dim) * 0.02)
        self.black_to_move_embed = nn.Parameter(torch.randn(1, llm_dim) * 0.02)
        
        if use_simple_projection:
            # Simple mode: concatenate all layers and project directly
            simple_input_dim = lc0_dim * num_layers
            self.simple_projection = nn.Linear(simple_input_dim, llm_dim)
            self.layer_projections = None
            self.mlp = None
        else:
            # Full mode: per-layer projections + MLP
            self.layer_projections = nn.ModuleList([
                nn.Linear(lc0_dim, projection_dim)
                for _ in range(num_layers)
            ])
            
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(lc0_dim)
                for _ in range(num_layers)
            ])
            
            self.pos_embed_dim = 16
            self.piece_embed_dim = 12
            self.piece_to_idx = {
                'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
                'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
            }
            
            mlp_input_dim = num_layers * projection_dim + self.pos_embed_dim + self.piece_embed_dim
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_dim, llm_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(llm_dim, llm_dim),
            )
            
            pos_embeddings = self._create_positional_embeddings()
            self.register_buffer('pos_embeddings', pos_embeddings)
            
    def _create_positional_embeddings(self) -> Tensor:
        positions = torch.zeros(64, 16, dtype=torch.float32)
        for rank in range(8):
            for file in range(8):
                sq_idx = rank * 8 + file
                positions[sq_idx, file] = 1.0 # File
                positions[sq_idx, 8 + rank] = 1.0 # Rank
        return positions
    
    def _create_piece_encodings(self, fen_list: Optional[List[str]], batch_size: int, device: torch.device) -> Tensor:
        piece_encodings = torch.zeros(batch_size, 64, self.piece_embed_dim, dtype=torch.float32, device=device)
        if not fen_list: return piece_encodings
        
        for batch_idx, fen in enumerate(fen_list):
            if not fen: continue
            piece_placement = fen.split()[0]
            rank, file = 7, 0
            for char in piece_placement:
                if char == '/':
                    rank -= 1; file = 0
                elif char.isdigit():
                    file += int(char)
                else:
                    sq_idx = rank * 8 + file
                    if char in self.piece_to_idx:
                        piece_encodings[batch_idx, sq_idx, self.piece_to_idx[char]] = 1.0
                    file += 1
        return piece_encodings
    
    def get_num_prefix_tokens(self) -> int:
        return self.num_squares + 1

    def forward(self, hidden_states, side_to_move=None, fen=None, return_square_embeddings=True, **kwargs) -> Tensor:
        # Note: Accepts **kwargs to safely ignore 'engineered_features' if passed by generic caller
        
        # Validation for dict input
        if isinstance(hidden_states, dict):
             first_states = list(hidden_states.values())[0]
        else:
             # Fallback if somehow passed as list (though type hint says dict)
             return None 
             
        batch_size = first_states.shape[0] if first_states.dim() == 3 else 1
        device = first_states.device
        
        if first_states.dim() == 2:
             # Handle unbatched
             hidden_states = {k: v.unsqueeze(0) for k, v in hidden_states.items()}
             batch_size = 1
             was_unbatched = True
             if fen and isinstance(fen, str): fen = [fen]
        else:
             was_unbatched = False

        if self.use_simple_projection:
            layer_states = [hidden_states[key] for key in self.layer_keys if key in hidden_states]
            concat = torch.cat(layer_states, dim=-1)
            
            if side_to_move is not None:
                mask = side_to_move.view(batch_size, 1, 1).to(device) if isinstance(side_to_move, Tensor) else torch.tensor(side_to_move, device=device).view(batch_size, 1, 1)
                # If Black to move (False in mask), flip. Wait, mask True=White.
                # If White (True), use Normal. If Black (False), use Flipped.
                concat = torch.where(mask, concat, concat[:, self.mirror_indices, :])
            
            square_embeds = self.simple_projection(concat)
        else:
            projected = []
            for i, key in enumerate(self.layer_keys):
                if key in hidden_states:
                    h = self.layer_norms[i](hidden_states[key])
                    projected.append(self.layer_projections[i](h))
            
            concat = torch.cat(projected, dim=-1)
            
            if side_to_move is not None:
                # Handle mirroring
                mask = side_to_move.view(batch_size, 1, 1).to(device) if isinstance(side_to_move, Tensor) else torch.tensor(side_to_move, device=device).view(batch_size, 1, 1)
                concat = torch.where(mask, concat, concat[:, self.mirror_indices, :])
            
            pos = self.pos_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
            piece_enc = self._create_piece_encodings(fen, batch_size, device)
            
            full_concat = torch.cat([concat, pos, piece_enc], dim=-1)
            square_embeds = self.mlp(full_concat)
            
        # Side embeddings
        if side_to_move is None:
             side_to_move_t = torch.ones(batch_size, dtype=torch.bool, device=device)
        elif not isinstance(side_to_move, Tensor):
             side_to_move_t = torch.tensor(side_to_move, dtype=torch.bool, device=device)
        else:
             side_to_move_t = side_to_move.to(device)
             
        side_embeds = torch.where(
            side_to_move_t.view(batch_size, 1, 1),
            self.white_to_move_embed.unsqueeze(0).expand(batch_size, -1, -1),
            self.black_to_move_embed.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        output = torch.cat([side_embeds, square_embeds], dim=1)
        
        if not return_square_embeddings:
            output = output.mean(dim=1)
            
        if was_unbatched:
            output = output.squeeze(0)
            
        return output


# ============================================================================
# New Engineered & Hybrid Adapters
# ============================================================================

ENGINEERED_SQUARE_ID_DIM = 64
ENGINEERED_PIECE_STATE_DIM = 13  # 12 pieces + explicit empty-square state
ENGINEERED_ATTACK_MASK_DIM = 64
ENGINEERED_DEFENSE_MASK_DIM = 64
ENGINEERED_EMPTY_PIECE_INDEX = 12
ENGINEERED_PIECE_OFFSET = ENGINEERED_SQUARE_ID_DIM
ENGINEERED_ATTACK_OFFSET = ENGINEERED_PIECE_OFFSET + ENGINEERED_PIECE_STATE_DIM
ENGINEERED_DEFENSE_OFFSET = ENGINEERED_ATTACK_OFFSET + ENGINEERED_ATTACK_MASK_DIM
ENGINEERED_FEATURE_DIM = (
    ENGINEERED_SQUARE_ID_DIM
    + ENGINEERED_PIECE_STATE_DIM
    + ENGINEERED_ATTACK_MASK_DIM
    + ENGINEERED_DEFENSE_MASK_DIM
)

_PIECE_TO_INDEX = {
    chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2, 
    chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
}

def extract_engineered_features(fen: str, mode: str = "simplified") -> torch.Tensor:
    """
    Extract engineered feature vector per square.
    
    Args:
        fen: FEN string
        mode: "simplified" (default) or "main".
              Simplified: Populates only explicit Piece/Pos/AttackCount/Flags (~19 dims)
              Main: Populates all 205 dims including full attack/defense bitmasks
    """
    board = chess.Board(fen)
    features = torch.zeros(64, ENGINEERED_FEATURE_DIM, dtype=torch.float32)
    piece_map = board.piece_map()
    
    # Common features (used by both modes, though sparse logic overwrites/ignores some)
    # Actually, legacy dense logic is quite specific. Let's separate cleanly.
    
    if mode == "main":
        # Main mode layout:
        #   0..63    = square identity one-hot
        #   64..76   = 13-way piece state (12 pieces + empty)
        #   77..140  = attacked-target bitmask
        #   141..204 = defended-friendly-target bitmask
        for sq in range(64):
            features[sq, sq] = 1.0
            if sq not in piece_map:
                features[sq, ENGINEERED_PIECE_OFFSET + ENGINEERED_EMPTY_PIECE_INDEX] = 1.0
            
        # 2. Piece encoding (13 dims including explicit empty state)
        for sq, piece in piece_map.items():
            piece_idx = _PIECE_TO_INDEX[piece.piece_type]
            if piece.color == chess.WHITE:
                features[sq, ENGINEERED_PIECE_OFFSET + piece_idx] = 1.0
            else:
                features[sq, ENGINEERED_PIECE_OFFSET + piece_idx + 6] = 1.0
        
        # 3. Attack vector (64 dims) - "Is square X attacked by piece on square SQ?"
        # Wait, legacy logic was:
        # for attacked_sq in board.attacks(sq): features[sq, ENGINEERED_ATTACK_OFFSET + attacked_sq] = 1.0
        # This means: "The piece on SQ attacks square attacked_sq"
        
        # 4. Defense vector (64 dims) - "Does piece on SQ defend friendly piece on X?"
        
        for sq in range(64):
            if sq in piece_map:
                piece = piece_map[sq]
                # Attack vector (64 dims)
                for target_sq in board.attacks(sq):
                    features[sq, ENGINEERED_ATTACK_OFFSET + target_sq] = 1.0
                    
                    # Defense vector (64 dims)
                    target_piece = board.piece_at(target_sq)
                    if target_piece is not None and target_piece.color == piece.color:
                        features[sq, ENGINEERED_DEFENSE_OFFSET + target_sq] = 1.0
                        
    else: # mode == "simplified"
        # 1. Piece & Pos
        for sq, piece in piece_map.items():
            idx = piece.piece_type - 1 # 0-5
            if piece.color == chess.BLACK: idx += 6
            features[sq, idx] = 1.0
            features[sq, 19] = chess.square_rank(sq) / 7.0
            features[sq, 20] = chess.square_file(sq) / 7.0
            
        for sq in range(64):
            if sq not in piece_map:
                features[sq, 18] = 1.0
                features[sq, 19] = chess.square_rank(sq) / 7.0
                features[sq, 20] = chess.square_file(sq) / 7.0
                
            # Attacks (Common to both)
            if board.is_attacked_by(chess.WHITE, sq): features[sq, 12] = 1.0
            if board.is_attacked_by(chess.BLACK, sq): features[sq, 13] = 1.0
            
            w_att = len(board.attackers(chess.WHITE, sq))
            b_att = len(board.attackers(chess.BLACK, sq))
            features[sq, 14] = min(w_att, 5.0) / 5.0
            features[sq, 15] = min(b_att, 5.0) / 5.0

    return features

class EngineeredPositionAdapter(nn.Module):
    def __init__(self, llm_dim: int = 2048, dropout: float = 0.1, **kwargs):
        super().__init__()
        # Handle config object passed as first arg
        if hasattr(llm_dim, 'llm_dim') or hasattr(llm_dim, 'model'): 
             # It's a config object
             llm_dim = 2048 
             
        self.mlp = nn.Sequential(
            nn.Linear(ENGINEERED_FEATURE_DIM, llm_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(llm_dim, llm_dim),
        )
        self.white_to_move_embed = nn.Parameter(torch.randn(1, llm_dim) * 0.02)
        self.black_to_move_embed = nn.Parameter(torch.randn(1, llm_dim) * 0.02)
        
    def get_num_prefix_tokens(self) -> int:
        return 65

    def forward(self, engineered_features, side_to_move=None, **kwargs):
        batch_size = engineered_features.shape[0]
        device = engineered_features.device
        
        square_embeds = self.mlp(engineered_features)
        
        if side_to_move is None:
             side_to_move_t = torch.ones(batch_size, dtype=torch.bool, device=device)
        elif not isinstance(side_to_move, Tensor):
             side_to_move_t = torch.tensor(side_to_move, dtype=torch.bool, device=device)
        else:
             side_to_move_t = side_to_move.to(device)
             
        side_embeds = torch.where(
            side_to_move_t.view(batch_size, 1, 1),
            self.white_to_move_embed.unsqueeze(0).expand(batch_size, -1, -1),
            self.black_to_move_embed.unsqueeze(0).expand(batch_size, -1, -1)
        )
        return torch.cat([side_embeds, square_embeds], dim=1)

class HybridPositionAdapter(nn.Module):
    """
    Combines LC0 (Projected from ChessPositionAdapter logic) with Engineered Features.
    """
    def __init__(self, config=None, lc0_dim: int = 768, lc0_proj_dim: int = 128, llm_dim: int = 2048, num_layers: int = 4, **kwargs):
        super().__init__()
        if config is not None and not isinstance(config, int):
            lc0_dim = getattr(config, 'lc0_dim', lc0_dim)
            lc0_proj_dim = getattr(getattr(config, 'hybrid', config), 'lc0_proj_dim', lc0_proj_dim)
            llm_dim = getattr(config, 'llm_dim', llm_dim)
            num_layers = getattr(config, 'num_layers', num_layers)
        self.lc0_dim = lc0_dim
        self.lc0_proj_dim = lc0_proj_dim
        self.llm_dim = llm_dim
        self.num_layers = num_layers
        
        # LC0 Components
        self.layer_keys = ["layer_4", "layer_8", "layer_12", "layer_15"]
        mirror_indices = torch.tensor([chess.square_mirror(sq) for sq in range(64)], dtype=torch.long)
        self.register_buffer('mirror_indices', mirror_indices)
        
        self.layer_projections = nn.ModuleList([nn.Linear(self.lc0_dim, self.lc0_proj_dim) for _ in range(self.num_layers)])
        self.layer_norms = nn.ModuleList([nn.LayerNorm(self.lc0_dim) for _ in range(self.num_layers)])
        
        # Hybrid MLP
        # Input: Engineered(205) + LC0(4*128)
        mlp_input = ENGINEERED_FEATURE_DIM + (self.num_layers * self.lc0_proj_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input, self.llm_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.llm_dim, self.llm_dim)
        )
        
        self.white_to_move_embed = nn.Parameter(torch.randn(1, self.llm_dim) * 0.02)
        self.black_to_move_embed = nn.Parameter(torch.randn(1, self.llm_dim) * 0.02)
        
    def get_num_prefix_tokens(self) -> int:
        return 65
        
    def forward(self, lc0_hidden_states, engineered_features, side_to_move=None, **kwargs):
        # We assume 3D batch input for safety
        batch_size = engineered_features.shape[0]
        device = engineered_features.device
        
        # 1. Process LC0
        projected = []
        for i, key in enumerate(self.layer_keys):
            if key in lc0_hidden_states:
                # Ensure input is float32 and on correct device
                h = lc0_hidden_states[key].to(device).float() 
                h = self.layer_norms[i](h)
                projected.append(self.layer_projections[i](h))
        
        lc0_concat = torch.cat(projected, dim=-1) # (B, 64, 512)
        
        # Handle LC0 Mirroring
        if side_to_move is not None:
             mask = side_to_move.view(batch_size, 1, 1).to(device) if isinstance(side_to_move, Tensor) else torch.tensor(side_to_move, device=device).view(batch_size, 1, 1)
             lc0_concat = torch.where(mask, lc0_concat, lc0_concat[:, self.mirror_indices, :])
             
        # 2. Concat with Engineered
        # Engineered features are already orientation-agnostic or standard? 
        # Standard: A1 is 0. 
        # If we flipped LC0 to be A1..H8 (White persp), we should ensure Engineered is too.
        # extract_engineered_features returns standard A1..H8.
        # So everything is White Perspective.
        
        combined = torch.cat([engineered_features, lc0_concat], dim=-1) # (B, 64, 717)
        square_embeds = self.mlp(combined)
        
        # 3. Side Embeds
        if side_to_move is None:
             side_to_move_t = torch.ones(batch_size, dtype=torch.bool, device=device)
        elif not isinstance(side_to_move, Tensor):
             side_to_move_t = torch.tensor(side_to_move, dtype=torch.bool, device=device)
        else:
             side_to_move_t = side_to_move.to(device)
             
        side_embeds = torch.where(
            side_to_move_t.view(batch_size, 1, 1),
            self.white_to_move_embed.unsqueeze(0).expand(batch_size, -1, -1),
            self.black_to_move_embed.unsqueeze(0).expand(batch_size, -1, -1)
        )
        
        return torch.cat([side_embeds, square_embeds], dim=1)
