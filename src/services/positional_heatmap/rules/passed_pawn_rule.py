"""Rule for evaluating passed pawns."""

from typing import Dict
import chess

from ..base_rule import PositionalRule


class PassedPawnRule(PositionalRule):
    """Rule that evaluates passed pawns.
    
    A passed pawn is a pawn with no enemy pawns in front of it on the same file
    or adjacent files. Passed pawns are generally advantageous.
    """
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.passed_pawn_score = config.get('score', 20.0)
    
    def evaluate(self, board: chess.Board, perspective: chess.Color) -> Dict[chess.Square, float]:
        scores: Dict[chess.Square, float] = {}
        opponent = not perspective
        pawns = board.pieces(chess.PAWN, perspective)
        central_files = [2, 3, 4, 5]  # c, d, e, f
        
        for pawn_square in pawns:
            file = chess.square_file(pawn_square)
            rank = chess.square_rank(pawn_square)
            piece = board.piece_at(pawn_square)
            if piece is None:
                continue
            pawn_color = piece.color
            pawn_score = 0.0
            
            if self._is_on_starting_rank(rank, pawn_color):
                continue
            
            is_blocked = self._is_blocked_pawn(board, file, rank, pawn_color)
            if is_blocked:
                continue
            
            if file in central_files:
                pawn_score = 10.0
            
            is_attacked = board.is_attacked_by(opponent, pawn_square)
            is_defended = board.is_attacked_by(pawn_color, pawn_square)
            
            if self._is_on_starting_rank(rank, pawn_color):
                if pawn_score != 0.0:
                    scores[pawn_square] = pawn_score
                continue
            
            is_passed = self._is_passed_pawn(board, file, rank, pawn_color)
            
            if is_passed:
                base_bonus = self._passed_pawn_bonus(rank, pawn_color)
                passed_bonus = base_bonus
                
                if is_attacked and not is_defended:
                    if rank < 4:
                        pawn_score = 0.0
                        passed_bonus = -base_bonus * 0.5
                    else:
                        passed_bonus = 0.0
                elif is_attacked:
                    passed_bonus *= 0.4
                
                pawn_score += passed_bonus
            else:
                pawn_score = 0.0
            
            if pawn_score != 0.0:
                scores[pawn_square] = pawn_score
        
        return scores
    
    def _is_on_starting_rank(self, rank: int, color: chess.Color) -> bool:
        if color == chess.WHITE:
            return rank <= 1
        else:
            return rank >= 6
    
    def _passed_pawn_bonus(self, rank: int, color: chess.Color) -> float:
        if color == chess.WHITE:
            if rank <= 1:
                return 0.0
            return (rank - 1) * 5.0
        else:
            if rank >= 6:
                return 0.0
            return (6 - rank) * 5.0
    
    def _is_passed_pawn(self, board: chess.Board, file: int, rank: int, color: chess.Color) -> bool:
        if self._is_on_starting_rank(rank, color):
            return False
        
        opponent = not color
        opponent_pawns = board.pieces(chess.PAWN, opponent)
        
        if color == chess.WHITE:
            ranks_to_check = range(rank + 1, 8)
        else:
            ranks_to_check = range(rank - 1, -1, -1)
        
        files_to_check = [file]
        if file > 0:
            files_to_check.append(file - 1)
        if file < 7:
            files_to_check.append(file + 1)
        
        for check_rank in ranks_to_check:
            for check_file in files_to_check:
                check_square = chess.square(check_file, check_rank)
                if check_square in opponent_pawns:
                    return False
        
        return True
    
    def _is_blocked_pawn(self, board: chess.Board, file: int, rank: int, color: chess.Color) -> bool:
        if color == chess.WHITE:
            if rank >= 7:
                return False
            front_square = chess.square(file, rank + 1)
        else:
            if rank <= 0:
                return False
            front_square = chess.square(file, rank - 1)
        
        return board.piece_at(front_square) is not None
