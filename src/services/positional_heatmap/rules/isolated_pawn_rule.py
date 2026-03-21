"""Rule for evaluating isolated pawns."""

from typing import Dict
import chess

from ..base_rule import PositionalRule


class IsolatedPawnRule(PositionalRule):
    """Rule that evaluates isolated pawns.
    
    An isolated pawn has no friendly pawns on adjacent files.
    Isolated pawns are generally weak because they cannot be defended by other pawns.
    """
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.isolated_pawn_penalty = config.get('score', -10.0)
    
    def evaluate(self, board: chess.Board, perspective: chess.Color) -> Dict[chess.Square, float]:
        scores: Dict[chess.Square, float] = {}
        pawns = board.pieces(chess.PAWN, perspective)
        
        for pawn_square in pawns:
            file = chess.square_file(pawn_square)
            
            if self._is_isolated_pawn(board, file, perspective):
                scores[pawn_square] = self.isolated_pawn_penalty
        
        return scores
    
    def _is_isolated_pawn(self, board: chess.Board, file: int, color: chess.Color) -> bool:
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        if file > 0:
            left_file_has_pawn = any(chess.square_file(sq) == file - 1 for sq in friendly_pawns)
            if left_file_has_pawn:
                return False
        
        if file < 7:
            right_file_has_pawn = any(chess.square_file(sq) == file + 1 for sq in friendly_pawns)
            if right_file_has_pawn:
                return False
        
        return True
