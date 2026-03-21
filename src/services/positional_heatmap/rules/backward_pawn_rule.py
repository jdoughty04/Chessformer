"""Rule for evaluating backward pawns."""

from typing import Dict
import chess

from ..base_rule import PositionalRule


class BackwardPawnRule(PositionalRule):
    """Rule that evaluates backward pawns.
    
    A backward pawn is a pawn that is behind friendly pawns on adjacent files
    and cannot advance safely. Backward pawns are generally weak.
    """
    
    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        self.backward_pawn_penalty = config.get('score', -15.0)
        self.defended_pawn_penalty = config.get('defended_score', -8.0)
    
    def evaluate(self, board: chess.Board, perspective: chess.Color) -> Dict[chess.Square, float]:
        scores: Dict[chess.Square, float] = {}
        pawns = board.pieces(chess.PAWN, perspective)
        
        for pawn_square in pawns:
            file = chess.square_file(pawn_square)
            rank = chess.square_rank(pawn_square)
            
            if self._is_backward_pawn(board, file, rank, perspective):
                is_defended = self._is_pawn_defended_by_adjacent_pawn(board, file, rank, perspective)
                if is_defended:
                    scores[pawn_square] = self.defended_pawn_penalty
                else:
                    scores[pawn_square] = self.backward_pawn_penalty
        
        return scores
    
    def _is_backward_pawn(self, board: chess.Board, file: int, rank: int, color: chess.Color) -> bool:
        friendly_pawns = board.pieces(chess.PAWN, color)
        opponent_pawns = board.pieces(chess.PAWN, not color)
        
        adjacent_files = []
        if file > 0:
            adjacent_files.append(file - 1)
        if file < 7:
            adjacent_files.append(file + 1)
        
        has_friendly_pawns_ahead = False
        
        if color == chess.WHITE:
            for adj_file in adjacent_files:
                for check_rank in range(rank + 1, 8):
                    check_square = chess.square(adj_file, check_rank)
                    if check_square in friendly_pawns:
                        has_friendly_pawns_ahead = True
                        break
                if has_friendly_pawns_ahead:
                    break
        else:
            for adj_file in adjacent_files:
                for check_rank in range(rank - 1, -1, -1):
                    check_square = chess.square(adj_file, check_rank)
                    if check_square in friendly_pawns:
                        has_friendly_pawns_ahead = True
                        break
                if has_friendly_pawns_ahead:
                    break
        
        if not has_friendly_pawns_ahead:
            return False
        
        if color == chess.WHITE:
            if rank >= 7:
                return True
            front_square = chess.square(file, rank + 1)
        else:
            if rank <= 0:
                return True
            front_square = chess.square(file, rank - 1)
        
        attack_files = []
        if file > 0:
            attack_files.append(file - 1)
        if file < 7:
            attack_files.append(file + 1)
        
        can_be_attacked = False
        
        for attack_file in attack_files:
            if attack_file < 0 or attack_file > 7:
                continue
            
            for attack_square in opponent_pawns:
                attack_file_check = chess.square_file(attack_square)
                attack_rank_check = chess.square_rank(attack_square)
                
                if attack_file_check != attack_file:
                    continue
                
                if color == chess.WHITE:
                    front_rank = rank + 1
                    if attack_rank_check >= front_rank:
                        can_be_attacked = True
                else:
                    front_rank = rank - 1
                    if attack_rank_check <= front_rank:
                        can_be_attacked = True
        
        if not can_be_attacked:
            return False
        
        if color == chess.WHITE:
            front_rank = rank + 1
            for defense_file in attack_files:
                if defense_file < 0 or defense_file > 7:
                    continue
                for defense_square in friendly_pawns:
                    defense_file_check = chess.square_file(defense_square)
                    defense_rank_check = chess.square_rank(defense_square)
                    
                    if defense_file_check != defense_file:
                        continue
                    
                    if defense_rank_check == front_rank - 1:
                        return False
        else:
            front_rank = rank - 1
            for defense_file in attack_files:
                if defense_file < 0 or defense_file > 7:
                    continue
                for defense_square in friendly_pawns:
                    defense_file_check = chess.square_file(defense_square)
                    defense_rank_check = chess.square_rank(defense_square)
                    
                    if defense_file_check != defense_file:
                        continue
                    
                    if defense_rank_check == front_rank + 1:
                        return False
        
        return True
    
    def _is_pawn_defended_by_adjacent_pawn(self, board: chess.Board, file: int, rank: int, color: chess.Color) -> bool:
        friendly_pawns = board.pieces(chess.PAWN, color)
        
        adjacent_files = []
        if file > 0:
            adjacent_files.append(file - 1)
        if file < 7:
            adjacent_files.append(file + 1)
        
        for adj_file in adjacent_files:
            for defense_square in friendly_pawns:
                defense_file = chess.square_file(defense_square)
                defense_rank = chess.square_rank(defense_square)
                
                if defense_file != adj_file:
                    continue
                
                if color == chess.WHITE:
                    if defense_rank == rank - 1:
                        return True
                else:
                    if defense_rank == rank + 1:
                        return True
        
        return False
