"""Rule for evaluating undeveloped pieces."""

from typing import Dict
import chess

from ..base_rule import PositionalRule


class UndevelopedPieceRule(PositionalRule):
    """Rule that evaluates undeveloped pieces.
    
    An undeveloped piece is a piece that is still on its starting square
    and is blocked by pawns (has no legal moves). Undeveloped pieces are
    generally a positional weakness.
    """
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.undeveloped_penalty = config.get('penalty', -8.0)
    
    def evaluate(self, board: chess.Board, perspective: chess.Color) -> Dict[chess.Square, float]:
        scores: Dict[chess.Square, float] = {}
        starting_squares = self._get_starting_squares(perspective)
        
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK]:
            pieces = board.pieces(piece_type, perspective)
            
            for piece_square in pieces:
                if piece_square not in starting_squares.get(piece_type, []):
                    continue
                
                if perspective == board.turn:
                    legal_moves = [move for move in board.legal_moves 
                                 if move.from_square == piece_square]
                    num_moves = len(legal_moves)
                else:
                    temp_board = board.copy()
                    temp_board.turn = perspective
                    legal_moves = [move for move in temp_board.generate_legal_moves() 
                                 if move.from_square == piece_square]
                    num_moves = len(legal_moves)
                
                if num_moves == 0:
                    scores[piece_square] = self.undeveloped_penalty
        
        return scores
    
    def _get_starting_squares(self, color: chess.Color) -> Dict[int, list]:
        if color == chess.WHITE:
            return {
                chess.KNIGHT: [chess.B1, chess.G1],
                chess.BISHOP: [chess.C1, chess.F1],
                chess.ROOK: [chess.A1, chess.H1],
            }
        else:
            return {
                chess.KNIGHT: [chess.B8, chess.G8],
                chess.BISHOP: [chess.C8, chess.F8],
                chess.ROOK: [chess.A8, chess.H8],
            }
