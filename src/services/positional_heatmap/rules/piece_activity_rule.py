"""Rule for evaluating piece activity."""

from typing import Dict
import chess

from ..base_rule import PositionalRule


class PieceActivityRule(PositionalRule):
    """Rule that evaluates piece activity.
    
    Evaluates how active pieces are based on:
    - Number of legal moves
    - Control of central squares
    - Piece mobility
    """
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.activity_bonus_per_move = config.get('activity_bonus_per_move', 1.0)
        self.central_square_bonus = config.get('central_square_bonus', 3.0)
        self.doubled_rooks_bonus = config.get('doubled_rooks_bonus', 20.0)
    
    def evaluate(self, board: chess.Board, perspective: chess.Color) -> Dict[chess.Square, float]:
        scores: Dict[chess.Square, float] = {}
        central_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
        
        for piece_type in [chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            pieces = board.pieces(piece_type, perspective)
            
            for piece_square in pieces:
                activity_score = 0.0
                
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
                
                attacks = board.attacks(piece_square)
                central_attacks = sum(1 for sq in central_squares if sq in attacks)
                
                if num_moves > 0:
                    if piece_type == chess.KNIGHT:
                        bonus_per_move = 2.0
                        max_bonus = 12.0
                    elif piece_type == chess.BISHOP:
                        bonus_per_move = 2.5
                        max_bonus = 15.0
                    elif piece_type == chess.ROOK:
                        bonus_per_move = 3.0
                        max_bonus = 18.0
                    elif piece_type == chess.QUEEN:
                        bonus_per_move = 4.0
                        max_bonus = 24.0
                    else:
                        bonus_per_move = self.activity_bonus_per_move
                        max_bonus = float('inf')
                    
                    mobility_bonus = min(num_moves * bonus_per_move, max_bonus)
                    activity_score += mobility_bonus
                    activity_score += central_attacks * self.central_square_bonus
                    
                    if piece_type == chess.ROOK:
                        piece_file = chess.square_file(piece_square)
                        if self._is_doubled_rooks_on_open_file(board, piece_file, perspective):
                            activity_score += self.doubled_rooks_bonus
                else:
                    if piece_type in [chess.ROOK, chess.BISHOP, chess.QUEEN]:
                        if piece_type == chess.ROOK:
                            piece_rank = chess.square_rank(piece_square)
                            if (perspective == chess.WHITE and piece_rank in [0, 1]) or \
                               (perspective == chess.BLACK and piece_rank in [6, 7]):
                                activity_score = 2.0
                            elif central_attacks > 0:
                                activity_score = -5.0
                            else:
                                activity_score = -10.0
                        elif piece_type == chess.BISHOP:
                            if self._is_development_square_for_bishop(piece_square, perspective):
                                activity_score = 2.0
                            elif central_attacks > 0:
                                activity_score = -5.0
                            else:
                                activity_score = -10.0
                        elif piece_type == chess.QUEEN:
                            piece_rank = chess.square_rank(piece_square)
                            piece_file = chess.square_file(piece_square)
                            is_on_starting_square = (perspective == chess.WHITE and piece_rank == 0 and piece_file == 3) or \
                                                     (perspective == chess.BLACK and piece_rank == 7 and piece_file == 3)
                            is_in_center_ranks = 2 <= piece_rank <= 5
                            if is_on_starting_square or is_in_center_ranks:
                                activity_score = 2.0
                            elif central_attacks > 0:
                                activity_score = 2.0
                            else:
                                activity_score = -10.0
                    elif piece_type == chess.KNIGHT:
                        piece_file = chess.square_file(piece_square)
                        piece_rank = chess.square_rank(piece_square)
                        is_on_edge = (piece_file == 0 or piece_file == 7) or (piece_rank == 0 or piece_rank == 7)
                        if is_on_edge:
                            activity_score = -5.0
                        elif central_attacks > 0:
                            activity_score = 2.0
                        else:
                            activity_score = 0.0
                
                if activity_score != 0.0:
                    scores[piece_square] = activity_score
        
        return scores
    
    def _is_development_square_for_bishop(self, square: chess.Square, color: chess.Color) -> bool:
        file = chess.square_file(square)
        rank = chess.square_rank(square)
        
        if color == chess.WHITE:
            if rank == 0:
                return file in [2, 5]
            elif rank == 1:
                return file in [3, 4, 6]
            elif rank == 2:
                return file in [2, 5]
        else:
            if rank == 7:
                return file in [2, 5]
            elif rank == 6:
                return file in [3, 4, 6]
            elif rank == 5:
                return file in [2, 5]
        
        return False
    
    def _is_doubled_rooks_on_open_file(self, board: chess.Board, file: int, color: chess.Color) -> bool:
        rooks = board.pieces(chess.ROOK, color)
        rooks_on_file = [sq for sq in rooks if chess.square_file(sq) == file]
        
        if len(rooks_on_file) < 2:
            return False
        
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        has_white_pawns = any(chess.square_file(sq) == file for sq in white_pawns)
        has_black_pawns = any(chess.square_file(sq) == file for sq in black_pawns)
        
        is_open_file = not has_white_pawns and not has_black_pawns
        return is_open_file
