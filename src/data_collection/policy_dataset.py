
import torch
from torch.utils.data import IterableDataset
import chess
import chess.pgn
import numpy as np
from pathlib import Path
from typing import Iterator, Tuple, List, Optional
import random

try:
    from training.perceiver_adapter import extract_perceiver_features
except ImportError:
    extract_perceiver_features = None

try:
    from training.cnn_model import extract_cnn_features
except ImportError:
    extract_cnn_features = None

try:
    from training.maia_model import extract_maia_features, get_maia_mapping
    from maia2 import utils as maia_utils
except ImportError:
    extract_maia_features = None
    maia_utils = None

class ChessPolicyDataset(IterableDataset):
    def __init__(self, pgn_file: str, start_index: int = 0, end_index: Optional[int] = None, infinite: bool = False, feature_mode: str = 'perceiver'):
        """
        Args:
            pgn_file: Path to PGN file.
            start_index: Game index to start at (inclusive).
            end_index: Game index to stop at (exclusive).
            infinite: If True, loops over the dataset indefinitely (useful for training).
            feature_mode: 'perceiver', 'cnn', or 'maia'.
        """
        self.pgn_file = Path(pgn_file)
        self.start_index = start_index
        self.end_index = end_index
        self.infinite = infinite
        self.feature_mode = feature_mode
        
    def __iter__(self) -> Iterator[Tuple[Tuple[torch.Tensor, torch.Tensor], int]]:
        """
        Yields:
            features: 
                - If mode='perceiver': (sq_features, global_features)
                - If mode='cnn': cnn_features (19, 8, 8)
            label: integer index of move (0-4095).
        """
        worker_info = torch.utils.data.get_worker_info()
        
        while True:
            with open(self.pgn_file, "r", encoding="utf-8") as f:
                game_count = -1
                
                while True:
                    # Robust game reading
                    try:
                        offset = f.tell()
                        line = f.readline()
                        if not line: break
                        if not line.startswith("[Event"):
                             continue
                        
                        f.seek(offset)
                        game = chess.pgn.read_game(f)
                    except Exception:
                        break
                        
                    if game is None:
                        break
                        
                    game_count += 1
                    
                    # Range Filtering
                    if game_count < self.start_index:
                        continue
                    if self.end_index is not None and game_count >= self.end_index:
                        break
                        
                    # Worker filtering (if needed, but usually we split ranges per worker or file)
                    if worker_info is not None:
                        if game_count % worker_info.num_workers != worker_info.id:
                            continue
                            
                    # Process Game
                    board = game.board()
                    for move in game.mainline_moves():
                        # We predict the move FROM the current board state
                        
                        # 1. Extract Features from current board
                        # We can simply use the fen, or pass board?
                        # extract_perceiver_features takes FEN string.
                        fen = board.fen()
                        
                        try:
                            if self.feature_mode == 'perceiver':
                                if extract_perceiver_features is None:
                                    raise ImportError("Perceiver features required but module not found.")
                                features = extract_perceiver_features(fen)
                            elif self.feature_mode == 'cnn':
                                if extract_cnn_features is None:
                                    raise ImportError("CNN features required but module not found.")
                                features = extract_cnn_features(fen)
                            elif self.feature_mode == 'maia':
                                if extract_maia_features is None:
                                    raise ImportError("Maia features required but module not found.")
                                features = extract_maia_features(fen)
                            else:
                                raise ValueError(f"Unknown feature mode: {self.feature_mode}")
                        except Exception as e:
                            print(f"Error extracting features for FEN {fen}: {e}")
                            board.push(move)
                            continue
                            
                        # 2. Encode Move (Label)
                        if self.feature_mode == 'maia':
                            # Handle Maia specific encoding (mirroring + vocabulary)
                            move_uci = move.uci()
                            
                            # If Black to move, we mirrored the board, so we must mirror the move
                            if board.turn == chess.BLACK:
                                if maia_utils is not None:
                                    move_uci = maia_utils.mirror_move(move_uci)
                                else:
                                    # Fallback manual mirroring if maia2 utils not available (should not happen)
                                    # But since imports are robust, let's just log error
                                    print("Error: maia2 utils not loaded for mirroring")
                                    
                            # Encode to index
                            label = get_maia_mapping().encode(move_uci)
                            
                            # If label is -1 (unknown move), skip
                            if label == -1:
                                board.push(move)
                                continue
                        else:
                            # Standard encoding: src * 64 + dst
                            # src and dst are 0-63
                            label = move.from_square * 64 + move.to_square
                        
                        yield features, label
                        
                        # Advance board
                        board.push(move)
                        
            if not self.infinite:
                break

def encode_move(move: chess.Move) -> int:
    return move.from_square * 64 + move.to_square

def decode_move(label: int) -> chess.Move:
    from_sq = label // 64
    to_sq = label % 64
    return chess.Move(from_sq, to_sq)
