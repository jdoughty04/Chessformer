"""Rule for evaluating king safety."""

from typing import Dict
import chess

from ..base_rule import PositionalRule


class KingSafetyRule(PositionalRule):
    """Rule that evaluates king safety.
    
    Evaluates factors like:
    - Open files near the king (weakness)
    - Pawn shield around king (strength)
    - Piece proximity to king
    """
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.open_file_penalty = config.get('open_file_penalty', -10.0)
        self.pawn_shield_bonus = config.get('pawn_shield_bonus', 5.0)
        self.exposed_king_penalty = config.get('exposed_king_penalty', -15.0)
    
    def evaluate(self, board: chess.Board, perspective: chess.Color) -> Dict[chess.Square, float]:
        scores: Dict[chess.Square, float] = {}
        
        king_square = board.king(perspective)
        if king_square is None:
            return scores
        
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        
        opponent = not perspective
        is_in_check = board.is_attacked_by(opponent, king_square)
        
        if is_in_check:
            king_score = -30.0
        else:
            king_score = 0.0
            
            open_files = self._get_open_files_near_king(board, king_file, perspective)
            if open_files:
                king_score += len(open_files) * self.open_file_penalty
            
            semi_open_files = self._get_semi_open_files_near_king(board, king_file, perspective)
            if semi_open_files:
                for file in semi_open_files:
                    if file == king_file:
                        king_score += self.open_file_penalty * 2.0
                    else:
                        king_score += self.open_file_penalty * 0.9
            
            pawn_shield_score = self._evaluate_pawn_shield(board, king_file, king_rank, perspective)
            if semi_open_files and king_file in semi_open_files:
                pawn_shield_score *= 0.3
            king_score += pawn_shield_score
            
            if self._is_king_exposed(board, king_file, king_rank, perspective):
                king_score += self.exposed_king_penalty
        
        if king_score != 0.0:
            scores[king_square] = king_score
        
        return scores
    
    def _get_open_files_near_king(self, board: chess.Board, king_file: int, color: chess.Color) -> list:
        open_files = []
        files_to_check = [king_file]
        if king_file > 0:
            files_to_check.append(king_file - 1)
        if king_file < 7:
            files_to_check.append(king_file + 1)
        
        all_pawns = board.pieces(chess.PAWN, chess.WHITE) | board.pieces(chess.PAWN, chess.BLACK)
        
        for file in files_to_check:
            has_pawns = any(chess.square_file(sq) == file for sq in all_pawns)
            if not has_pawns:
                open_files.append(file)
        
        return open_files
    
    def _get_semi_open_files_near_king(self, board: chess.Board, king_file: int, color: chess.Color) -> list:
        semi_open_files = []
        opponent = not color
        
        files_to_check = [king_file]
        if king_file > 0:
            files_to_check.append(king_file - 1)
        if king_file < 7:
            files_to_check.append(king_file + 1)
        
        friendly_pawns = board.pieces(chess.PAWN, color)
        opponent_pawns = board.pieces(chess.PAWN, opponent)
        
        for file in files_to_check:
            has_friendly_pawns = any(chess.square_file(sq) == file for sq in friendly_pawns)
            has_opponent_pawns = any(chess.square_file(sq) == file for sq in opponent_pawns)
            
            if has_friendly_pawns and not has_opponent_pawns:
                semi_open_files.append(file)
        
        return semi_open_files
    
    def _evaluate_pawn_shield(self, board: chess.Board, king_file: int, king_rank: int,
                              color: chess.Color) -> float:
        friendly_pawns = board.pieces(chess.PAWN, color)
        score = 0.0
        
        files_to_check = [king_file]
        if king_file > 0:
            files_to_check.append(king_file - 1)
        if king_file < 7:
            files_to_check.append(king_file + 1)
        
        if color == chess.WHITE:
            ranks_to_check = range(king_rank + 1, 8) if king_rank < 7 else []
        else:
            ranks_to_check = range(king_rank - 1, -1, -1) if king_rank > 0 else []
        
        pawn_count = 0
        for file in files_to_check:
            for rank in list(ranks_to_check)[:2]:
                check_square = chess.square(file, rank)
                if check_square in friendly_pawns:
                    pawn_count += 1
        
        score = pawn_count * self.pawn_shield_bonus
        return score
    
    def _is_king_exposed(self, board: chess.Board, king_file: int, king_rank: int,
                         color: chess.Color) -> bool:
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        files_to_check = [king_file]
        if king_file > 0:
            files_to_check.append(king_file - 1)
        if king_file < 7:
            files_to_check.append(king_file + 1)
        
        if color == chess.WHITE:
            ranks_to_check = range(king_rank + 1, 8) if king_rank < 7 else []
        else:
            ranks_to_check = range(king_rank - 1, -1, -1) if king_rank > 0 else []
        
        for file in files_to_check:
            for rank in ranks_to_check:
                check_square = chess.square(file, rank)
                if check_square in friendly_pawns:
                    return False
        
        return True
