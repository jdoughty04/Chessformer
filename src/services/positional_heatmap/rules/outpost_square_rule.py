"""Rule for evaluating outpost squares."""

from typing import Dict
import chess

from ..base_rule import PositionalRule


class OutpostSquareRule(PositionalRule):
    """Rule that evaluates outpost squares.
    
    An outpost square is a square that:
    - Must be protected by friendly pawns (REQUIRED)
    - Cannot be attacked by enemy pawns (current or potential)
    - Is on an advanced rank (opponent's side of the board)
    - Is in a strategically important area (usually central, not on edge files)
    - Is occupied by a piece (especially knights)
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.knight_bonus = config.get('knight_bonus', 12.0)
        self.bishop_bonus = config.get('bishop_bonus', 8.0)
        self.central_bonus = config.get('central_bonus', 3.0)
        self.protected_bonus = config.get('protected_bonus', 2.0)

    def evaluate(self, board: chess.Board, perspective: chess.Color) -> Dict[chess.Square, float]:
        scores: Dict[chess.Square, float] = {}
        opponent = not perspective
        
        central_squares = [
            chess.D4, chess.D5, chess.E4, chess.E5,
            chess.C4, chess.C5, chess.F4, chess.F5
        ]
        
        for piece_type in [chess.KNIGHT, chess.BISHOP]:
            pieces = board.pieces(piece_type, perspective)
            
            for piece_square in pieces:
                file = chess.square_file(piece_square)
                rank = chess.square_rank(piece_square)
                
                is_outpost = self._is_outpost_square(board, file, rank, perspective)
                
                if is_outpost:
                    outpost_score = 0.0
                    
                    if piece_type == chess.KNIGHT:
                        outpost_score = self.knight_bonus
                    elif piece_type == chess.BISHOP:
                        outpost_score = self.bishop_bonus
                    
                    if piece_square in central_squares:
                        outpost_score += self.central_bonus
                    
                    scores[piece_square] = outpost_score
        
        return scores
    
    def _is_outpost_square(self, board: chess.Board, file: int, rank: int,
                           color: chess.Color) -> bool:
        opponent = not color
        
        if file == 0 or file == 7:
            return False
        
        if color == chess.WHITE:
            if rank < 3:
                return False
        else:
            if rank > 3:
                return False
        
        if not self._is_protected_by_pawns(board, file, rank, color):
            return False
        
        if self._can_be_attacked_by_pawns(board, file, rank, opponent):
            return False
        
        if self._can_be_attacked_by_advancing_pawns(board, file, rank, opponent):
            return False
        
        return True
    
    def _can_be_attacked_by_pawns(self, board: chess.Board, file: int, rank: int,
                                 color: chess.Color) -> bool:
        enemy_pawns = board.pieces(chess.PAWN, color)
        
        if color == chess.WHITE:
            if rank > 0:
                if file > 0:
                    left_diag_square = chess.square(file - 1, rank - 1)
                    if left_diag_square in enemy_pawns:
                        return True
                if file < 7:
                    right_diag_square = chess.square(file + 1, rank - 1)
                    if right_diag_square in enemy_pawns:
                        return True
        else:
            if rank < 7:
                if file > 0:
                    left_diag_square = chess.square(file - 1, rank + 1)
                    if left_diag_square in enemy_pawns:
                        return True
                if file < 7:
                    right_diag_square = chess.square(file + 1, rank + 1)
                    if right_diag_square in enemy_pawns:
                        return True
        
        return False
    
    def _can_be_attacked_by_advancing_pawns(self, board: chess.Board, file: int, rank: int,
                                            color: chess.Color) -> bool:
        enemy_pawns = board.pieces(chess.PAWN, color)
        
        if color == chess.WHITE:
            if rank < 7:
                for pawn_rank in range(rank + 1, 8):
                    pawn_square = chess.square(file, pawn_rank)
                    if pawn_square in enemy_pawns:
                        return True
                
                if file > 0:
                    for pawn_rank in range(rank + 1, 8):
                        pawn_square = chess.square(file - 1, pawn_rank)
                        if pawn_square in enemy_pawns:
                            return True
                
                if file < 7:
                    for pawn_rank in range(rank + 1, 8):
                        pawn_square = chess.square(file + 1, pawn_rank)
                        if pawn_square in enemy_pawns:
                            return True
        else:
            if rank > 0:
                for pawn_rank in range(rank - 1, -1, -1):
                    pawn_square = chess.square(file, pawn_rank)
                    if pawn_square in enemy_pawns:
                        return True
                
                if file > 0:
                    for pawn_rank in range(rank - 1, -1, -1):
                        pawn_square = chess.square(file - 1, pawn_rank)
                        if pawn_square in enemy_pawns:
                            return True
                
                if file < 7:
                    for pawn_rank in range(rank - 1, -1, -1):
                        pawn_square = chess.square(file + 1, pawn_rank)
                        if pawn_square in enemy_pawns:
                            return True
        
        return False
    
    def _is_protected_by_pawns(self, board: chess.Board, file: int, rank: int,
                               color: chess.Color) -> bool:
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        if color == chess.WHITE:
            if rank > 0:
                if file > 0:
                    left_diag_square = chess.square(file - 1, rank - 1)
                    if left_diag_square in friendly_pawns:
                        return True
                if file < 7:
                    right_diag_square = chess.square(file + 1, rank - 1)
                    if right_diag_square in friendly_pawns:
                        return True
        else:
            if rank < 7:
                if file > 0:
                    left_diag_square = chess.square(file - 1, rank + 1)
                    if left_diag_square in friendly_pawns:
                        return True
                if file < 7:
                    right_diag_square = chess.square(file + 1, rank + 1)
                    if right_diag_square in friendly_pawns:
                        return True
        
        return False
