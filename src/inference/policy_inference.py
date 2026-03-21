
import sys
import argparse
import torch
import chess
import chess.pgn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import yaml

# Add src to path to allow imports
sys.path.append(str(Path.cwd() / "src"))

from training.policy_model import PerceiverPolicyModel
from training.perceiver_adapter import extract_perceiver_features
from training.config import ModelConfig, PerceiverConfig

def load_model(checkpoint_path, device):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Reconstruct config
    # Checkpoint might have 'config' dict
    config_dict = checkpoint.get('config', {})
    model_cfg_dict = config_dict.get('model', {})
    
    # Create ModelConfig
    config = ModelConfig(mode="perceiver")
    if 'perceiver' in model_cfg_dict:
        config.perceiver = PerceiverConfig(**model_cfg_dict['perceiver'])
    
    # Initialize model
    model = PerceiverPolicyModel(config)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        # Maybe it's a direct state dict save?
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()
    return model

def decode_move_from_label(label: int) -> chess.Move:
    from_sq = label // 64
    to_sq = label % 64
    return chess.Move(from_sq, to_sq)

def analyze_pgn(pgn_path, model, device, output_path=None, limit=None):
    print(f"Analyzing PGN: {pgn_path}")
    
    total_moves = 0
    top1_correct = 0
    top5_correct = 0
    
    annotated_game_count = 0
    
    output_handle = None
    if output_path:
        output_handle = open(output_path, "w", encoding="utf-8")
    
    with open(pgn_path, "r", encoding="utf-8") as f:
        while True:
            if limit and annotated_game_count >= limit:
                break
                
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            annotated_game_count += 1
            board = game.board()
            
            node = game
            
            # Iterate through moves
            for move in game.mainline_moves():
                # Extract features for current board
                try:
                    features = extract_perceiver_features(board.fen())
                except Exception as e:
                    print(f"Skipping position due to extraction error: {e}")
                    board.push(move)
                    node = node.variations[0]
                    continue
                
                # Prepare batch
                sq_feats, glob_feats = features
                sq_feats = sq_feats.unsqueeze(0).to(device) # (1, 64, 83)
                glob_feats = glob_feats.unsqueeze(0).to(device) # (1, 4, 16)
                
                # Inference
                with torch.no_grad():
                    # side_to_move: White=True
                    side_to_move = (board.turn == chess.WHITE)
                    side_tensor = torch.tensor([side_to_move], dtype=torch.bool, device=device)
                    
                    from_logits, to_logits = model((sq_feats, glob_feats), side_to_move=side_tensor)
                    
                    from_probs = torch.softmax(from_logits, dim=1) # (1, 64)
                    to_probs = torch.softmax(to_logits, dim=1)     # (1, 64)
                    
                    # Compute joint probabilities: P(from) * P(to)
                    # Outer product: (B, 64, 1) * (B, 1, 64) -> (B, 64, 64)
                    joint_probs = torch.matmul(from_probs.unsqueeze(2), to_probs.unsqueeze(1))
                    
                    # Flatten back to 4096 for easy top-k compatibility with existing code structure
                    probs = joint_probs.view(1, 4096)
                
                # Get Top-K
                topk_probs, topk_indices = torch.topk(probs, k=5, dim=1)
                
                # Ground truth
                gt_label = move.from_square * 64 + move.to_square
                gt_move_uci = move.uci()
                
                # Check accuracy
                top1_pred = topk_indices[0, 0].item()
                if top1_pred == gt_label:
                    top1_correct += 1
                
                if gt_label in topk_indices[0].tolist():
                    top5_correct += 1
                
                total_moves += 1
                
                # Annotation
                if output_handle:
                    # Construct comment
                    # "Model: e4 (0.45), d4 (0.22)..."
                    preds_str = []
                    for i in range(5):
                        idx = topk_indices[0, i].item()
                        prob = topk_probs[0, i].item()
                        pred_move = decode_move_from_label(idx)
                        # Sanity check if move is legal?
                        # Using board.san(pred_move) might fail if illegal
                        try:
                            san = board.san(pred_move)
                        except:
                            san = pred_move.uci() # Fallback
                        
                        preds_str.append(f"{san} ({prob:.2f})")
                    
                    comment = f"Eval: {', '.join(preds_str)}"
                    
                    # Add to PGN node
                    # We are iterating mainline_moves which doesn't give us the node directly easily unless we walk
                    # We need to walk the node tree
                    if node.variations:
                        node = node.variations[0]
                        node.comment = comment
                
                board.push(move)
                
            if output_handle:
                print(game, file=output_handle, end="\n\n")
                
            if annotated_game_count % 10 == 0:
                print(f"Processed {annotated_game_count} games. Current Acc: {top1_correct/total_moves:.4f}")

    if output_handle:
        output_handle.close()
        print(f"Annotated PGN saved to {output_path}")
        
    print("="*40)
    print(f"Total Moves: {total_moves}")
    print(f"Top-1 Accuracy: {top1_correct/total_moves:.4f}")
    if total_moves > 0:
        print(f"Top-5 Accuracy: {top5_correct/total_moves:.4f}")
    print("="*40)

def main():
    parser = argparse.ArgumentParser(description="Run policy inference on PGN")
    parser.add_argument("--pgn", type=str, required=True, help="Input PGN file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--output", type=str, help="Output PGN file for annotations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, help="Limit number of games")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    model = load_model(args.checkpoint, device)
    
    analyze_pgn(args.pgn, model, device, args.output, args.limit)

if __name__ == "__main__":
    main()
