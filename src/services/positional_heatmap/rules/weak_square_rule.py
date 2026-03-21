"""Rule for evaluating weak squares."""

from typing import Dict
import chess

from ..base_rule import PositionalRule


class WeakSquareRule(PositionalRule):
    """Rule that evaluates weak squares.
    
    A weak square is a square that:
    - Cannot be defended by pawns
    - Is attacked by enemy pieces
    - Is in a strategically important area
    """
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.weak_square_penalty = config.get('score', -8.0)
        self.undefended_penalty = config.get('undefended_penalty', -2.0)
    
    def evaluate(self, board: chess.Board, perspective: chess.Color) -> Dict[chess.Square, float]:
        scores: Dict[chess.Square, float] = {}
        opponent = not perspective
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            if piece.color != perspective:
                continue
            
            file = chess.square_file(square)
            rank = chess.square_rank(square)
            weakness_score = 0.0
            
            if board.is_attacked_by(opponent, square):
                if not board.is_attacked_by(perspective, square):
                    piece_type = piece.piece_type
                    if piece_type == chess.PAWN:
                        base_penalty = -6.0
                    elif piece_type in [chess.KNIGHT, chess.BISHOP]:
                        base_penalty = -8.0
                    elif piece_type == chess.ROOK:
                        base_penalty = -10.0
                    elif piece_type == chess.QUEEN:
                        base_penalty = -12.0
                    elif piece_type == chess.KING:
                        base_penalty = -15.0
                    else:
                        base_penalty = self.weak_square_penalty
                    
                    weakness_score += base_penalty
                    
                    if not self._can_be_defended_by_pawns(board, file, rank, perspective):
                        weakness_score += self.undefended_penalty
            
            if weakness_score != 0.0:
                scores[square] = weakness_score
        
        return scores
    
    def _can_be_defended_by_pawns(self, board: chess.Board, file: int, rank: int,
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
