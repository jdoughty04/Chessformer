"""
LC0 Hidden State Extractor

Extracts intermediate hidden states from LC0 BT4 transformer network.
Uses ONNX conversion to access internal transformer layer activations.

The BT4 network uses a transformer architecture where each of the 64 chess squares
is treated as a sequence position (like tokens in a language model). The transformer
processes the input planes and produces embeddings for each square.

We extract hidden states from layers 4, 8, 12, and 15 (0-indexed).
"""

import os
import gzip
import urllib.request
from pathlib import Path
from typing import Optional
import numpy as np

try:
    import chess
except ImportError:
    raise ImportError("python-chess is required. Install with: pip install python-chess")

try:
    import onnx
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnx and onnxruntime are required. Install with: pip install onnx onnxruntime-gpu")


# BT4 network download URL (transformer-based, from lczero.org big-transformers)
BT4_NETWORK_URL = "https://storage.lczero.org/files/networks-contrib/big-transformers/BT4-1740.pb.gz"
BT4_NETWORK_NAME = "BT4-1740.pb.gz"

# Layer indices to extract (1-indexed, maps to encoder{layer-1} in ONNX)
# For BT3/BT4 with 15 encoders (0-14), layer 4 = encoder3, layer 15 = encoder14
DEFAULT_LAYERS = [4, 8, 12, 15]

# LC0 input encoding constants
NUM_HISTORY_POSITIONS = 8  # LC0 uses 8 past positions
PLANES_PER_POSITION = 13   # 12 piece planes + 1 repetition plane
TOTAL_PLANES = 112         # Full input size


class LC0HiddenStateExtractor:
    """
    Extract hidden states from LC0 BT4 transformer network.
    
    The BT4 architecture:
    - Input: 112 planes (8x8 each) encoding position + history
    - Transformer layers with embeddings per square
    - Each square (64 total) gets an embedding per layer
    
    We extract outputs from specified transformer layers and return
    them as (64, embedding_dim) arrays per layer.
    """
    
    def __init__(
        self,
        network_path: Optional[str] = None,
        layers: list[int] = None,
        cache_dir: str = None,
        device: str = "cuda"  # "cuda" or "cpu"
    ):
        """
        Initialize the LC0 hidden state extractor.
        
        Args:
            network_path: Path to LC0 .pb.gz or .onnx network file.
                         If None, downloads BT4 automatically.
            layers: List of transformer layer indices to extract (0-indexed).
                   Default: [4, 8, 12, 15]
            cache_dir: Directory to cache downloaded/converted networks.
            device: "cuda" or "cpu" for ONNX runtime.
        """
        self.layers = layers or DEFAULT_LAYERS
        self.device = device
        
        # Setup cache directory
        if cache_dir is None:
            cache_dir = Path(__file__).parent / "lc0_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Get network path (download if needed)
        if network_path is None:
            network_path = self._ensure_network_downloaded()
        
        self.network_path = Path(network_path)
        
        # Convert to ONNX if needed and load
        self.onnx_path = self._ensure_onnx_converted()
        self.session = self._load_onnx_session()
        
        # Pre-compute positional encoding for input
        self._init_input_encoding()
    
    def _ensure_network_downloaded(self) -> Path:
        """Download BT4 network if not already cached."""
        network_path = self.cache_dir / BT4_NETWORK_NAME
        
        if network_path.exists():
            print(f"Using cached network: {network_path}")
            return network_path
        
        print(f"Downloading BT4 network from {BT4_NETWORK_URL}...")
        print("This may take a few minutes...")
        
        try:
            urllib.request.urlretrieve(BT4_NETWORK_URL, network_path)
            print(f"Downloaded to: {network_path}")
            return network_path
        except Exception as e:
            raise RuntimeError(f"Failed to download BT4 network: {e}")
    
    def _ensure_onnx_converted(self) -> Path:
        """
        Convert LC0 network to ONNX format with intermediate outputs exposed.
        
        This requires the lc0 CLI tool to be installed for leela2onnx conversion.
        If lc0 is not available, we provide instructions.
        """
        onnx_path = self.cache_dir / f"{self.network_path.stem}_hidden.onnx"
        
        if onnx_path.exists():
            print(f"Using cached ONNX model: {onnx_path}")
            return onnx_path
        
        # First, try to convert using lc0 leela2onnx
        base_onnx_path = self.cache_dir / f"{self.network_path.stem}.onnx"
        
        if not base_onnx_path.exists():
            # Try to run lc0 leela2onnx
            import subprocess
            try:
                print("Converting network to ONNX format using lc0...")
                result = subprocess.run(
                    ["lc0", "leela2onnx", 
                     f"--input={self.network_path}",
                     f"--output={base_onnx_path}"],
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                if result.returncode != 0:
                    raise RuntimeError(f"lc0 leela2onnx failed: {result.stderr}")
                print(f"Converted to: {base_onnx_path}")
            except FileNotFoundError:
                raise RuntimeError(
                    "lc0 CLI tool not found. To convert networks to ONNX, you need to:\n"
                    "1. Download lc0 from https://github.com/LeelaChessZero/lc0/releases\n"
                    "2. Add lc0 to your PATH\n"
                    "3. Or manually convert: lc0 leela2onnx --input=<network.pb.gz> --output=<network.onnx>"
                )
        
        # Now modify the ONNX model to expose intermediate layer outputs
        print("Modifying ONNX model to expose hidden states...")
        self._add_intermediate_outputs(base_onnx_path, onnx_path)
        
        return onnx_path
    
    def _add_intermediate_outputs(self, input_path: Path, output_path: Path):
        """
        Modify ONNX model to expose intermediate transformer layer outputs.
        
        We identify the output tensors of each transformer layer and add them
        as additional model outputs.
        
        LC0 transformer naming pattern: /encoderN/ffn/skip
        where N is 0-indexed encoder layer number.
        """
        model = onnx.load(str(input_path))
        
        # Find transformer layer output nodes
        # LC0 BT3/BT4 uses naming like "/encoder{N}/ffn/skip" for the
        # residual connection after the FFN layer of each encoder
        
        layer_outputs = {}
        
        # Build list of patterns to search for
        # Note: self.layers contains 1-indexed layer numbers
        # but ONNX uses 0-indexed encoder numbers
        patterns_to_find = {}
        for layer_idx in self.layers:
            encoder_idx = layer_idx - 1  # Convert to 0-indexed
            # The FFN skip connection is the final output of each encoder block
            patterns_to_find[layer_idx] = f"/encoder{encoder_idx}/ffn/skip"
        
        # Search for matching nodes
        for node in model.graph.node:
            node_name = node.name
            
            for layer_idx, pattern in patterns_to_find.items():
                if node_name == pattern and layer_idx not in layer_outputs:
                    if node.output:
                        layer_outputs[layer_idx] = node.output[0]
                        print(f"  Found layer {layer_idx}: {node_name} -> {node.output[0]}")
                        break
        
        # Check if we found all requested layers
        missing = set(self.layers) - set(layer_outputs.keys())
        if missing:
            print(f"Warning: Could not find outputs for layers: {missing}")
            print("Available encoder nodes:")
            for node in model.graph.node:
                if '/encoder' in node.name and '/ffn/skip' in node.name:
                    print(f"  {node.name} -> {node.output}")
        
        if not layer_outputs:
            raise RuntimeError(
                "Could not identify any transformer layer outputs. "
                "The ONNX graph structure may differ from expected. "
                "Please inspect the model using Netron (https://netron.app)."
            )
        
        print(f"Found layer outputs: {layer_outputs}")
        
        # Add intermediate outputs to the model
        # Don't specify shape - let ONNX runtime infer it dynamically
        for layer_idx, output_name in layer_outputs.items():
            # Create output info with no shape constraint
            output_info = onnx.helper.make_tensor_value_info(
                output_name,
                onnx.TensorProto.FLOAT,
                None  # Dynamic shape determination
            )
            model.graph.output.append(output_info)
        
        onnx.save(model, str(output_path))
        print(f"Saved modified ONNX model to: {output_path}")
    
    def _load_onnx_session(self) -> ort.InferenceSession:
        """Load ONNX model into inference session."""
        providers = []
        if self.device == "cuda":
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        
        print(f"Loading ONNX model with providers: {providers}")
        session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        
        # Print input/output info
        print("\nModel inputs:")
        for inp in session.get_inputs():
            print(f"  {inp.name}: {inp.shape} ({inp.type})")
        
        print("\nModel outputs:")
        for out in session.get_outputs():
            print(f"  {out.name}: {out.shape}")
        
        return session
    
    def _init_input_encoding(self):
        """Initialize data structures for input encoding."""
        # Piece type indices (for encoding)
        self.piece_to_plane = {
            chess.PAWN: 0,
            chess.KNIGHT: 1,
            chess.BISHOP: 2,
            chess.ROOK: 3,
            chess.QUEEN: 4,
            chess.KING: 5,
        }
    
    def board_to_input(self, board: chess.Board) -> np.ndarray:
        """
        Convert a chess.Board with history to LC0's 112-plane input format.
        
        ALL encoding is from the perspective of the side to move.
        When black is to move, the board is vertically flipped.
        
        The 112 planes are organized as:
        - Planes 0-12: Current position (6 our pieces + 6 opponent pieces + 1 repetition)
        - Planes 13-25: Position 1 ply ago
        - ... (8 positions total, 13 planes each = 104 planes)
        - Plane 104: Our queenside castling
        - Plane 105: Our kingside castling
        - Plane 106: Their queenside castling
        - Plane 107: Their kingside castling
        - Plane 108: Side to move (always 1 - we're always "us")
        - Plane 109: 50-move rule counter (normalized)
        - Plane 110: Unused
        - Plane 111: En passant square
        
        Args:
            board: chess.Board with move_stack containing game history
            
        Returns:
            numpy array of shape (112, 8, 8)
        """
        planes = np.zeros((TOTAL_PLANES, 8, 8), dtype=np.float32)
        
        # Get history positions (current + 7 previous)
        history_boards = self._get_history_boards(board)
        
        # Encode each historical position
        for hist_idx, hist_board in enumerate(history_boards):
            base_plane = hist_idx * PLANES_PER_POSITION
            self._encode_position(planes, base_plane, hist_board, board.turn)
        
        # Encode auxiliary planes (planes 104-111)
        self._encode_auxiliary(planes, board)
        
        return planes
    
    def _get_history_boards(self, board: chess.Board) -> list[chess.Board]:
        """
        Get the last 8 board positions including current.
        
        If there aren't 8 positions yet, pad with the earliest available.
        """
        # Start from beginning and replay to build history
        history = []
        temp_board = chess.Board()
        
        for move in board.move_stack:
            history.append(temp_board.copy())
            temp_board.push(move)
        
        # Current position
        history.append(board.copy())
        
        # Get last 8 positions (current + 7 previous)
        if len(history) >= NUM_HISTORY_POSITIONS:
            return history[-NUM_HISTORY_POSITIONS:]
        else:
            # Pad with earliest position
            padding = [history[0]] * (NUM_HISTORY_POSITIONS - len(history))
            return padding + history
    
    def _encode_position(
        self, 
        planes: np.ndarray, 
        base_plane: int, 
        board: chess.Board,
        perspective: chess.Color
    ):
        """
        Encode a single position into planes.
        
        LC0 always encodes from the perspective of the current player.
        When black is to move, the board is flipped.
        
        Args:
            planes: Output array to fill
            base_plane: Starting plane index
            board: Position to encode
            perspective: Whose perspective to encode from
        """
        # Determine if we need to flip
        flip = (perspective == chess.BLACK)
        
        # Encode our pieces (planes 0-5)
        for piece_type in chess.PIECE_TYPES:
            plane_idx = base_plane + self.piece_to_plane[piece_type]
            pieces = board.pieces(piece_type, perspective)
            for square in pieces:
                if flip:
                    square = chess.square_mirror(square)
                row = square // 8
                col = square % 8
                planes[plane_idx, row, col] = 1.0
        
        # Encode opponent pieces (planes 6-11)
        opponent = not perspective
        for piece_type in chess.PIECE_TYPES:
            plane_idx = base_plane + 6 + self.piece_to_plane[piece_type]
            pieces = board.pieces(piece_type, opponent)
            for square in pieces:
                if flip:
                    square = chess.square_mirror(square)
                row = square // 8
                col = square % 8
                planes[plane_idx, row, col] = 1.0
        
        # Repetition plane (plane 12)
        # Check if this position has been seen before
        if board.is_repetition(2):
            planes[base_plane + 12, :, :] = 1.0
    
    def _encode_auxiliary(self, planes: np.ndarray, board: chess.Board):
        """
        Encode auxiliary game state planes (104-111).
        
        LC0 encodes these from the perspective of the side to move:
        - Planes 104-107: Castling rights (our queenside, our kingside, their queenside, their kingside)
        - Plane 108: Unused (was side-to-move, but always 1 from our perspective)
        - Plane 109: 50-move rule counter (normalized)
        - Plane 110-111: En passant information
        """
        perspective = board.turn
        opponent = not perspective
        flip = (perspective == chess.BLACK)
        
        # Castling rights from the perspective of side to move
        # Plane 104: Our queenside castling
        if board.has_queenside_castling_rights(perspective):
            planes[104, :, :] = 1.0
        # Plane 105: Our kingside castling
        if board.has_kingside_castling_rights(perspective):
            planes[105, :, :] = 1.0
        # Plane 106: Their queenside castling
        if board.has_queenside_castling_rights(opponent):
            planes[106, :, :] = 1.0
        # Plane 107: Their kingside castling
        if board.has_kingside_castling_rights(opponent):
            planes[107, :, :] = 1.0
        
        # Plane 108: Side to move - always 1 since we're encoding from our perspective
        # (LC0 always encodes from the player-to-move's perspective)
        planes[108, :, :] = 1.0
        
        # Plane 109: 50-move rule counter (normalized)
        planes[109, :, :] = board.halfmove_clock / 100.0
        
        # Plane 110: Unused / legacy
        # (previously used for various experimental features)
        
        # Plane 111: En passant square
        if board.ep_square is not None:
            ep_sq = board.ep_square
            if flip:
                ep_sq = chess.square_mirror(ep_sq)
            row = ep_sq // 8
            col = ep_sq % 8
            planes[111, row, col] = 1.0
    
    def extract(self, board: chess.Board) -> dict[str, np.ndarray]:
        """
        Extract hidden states for a position WITH history.
        
        Args:
            board: chess.Board with move_stack containing game history
            
        Returns:
            Dictionary with keys like "layer_4", "layer_8", etc.
            Each value has shape (64, 768) - 64 squares, 768 embedding dim
        """
        # Convert board to input
        input_planes = self.board_to_input(board)
        
        # Add batch dimension and run inference
        input_batch = input_planes[np.newaxis, ...]  # (1, 112, 8, 8)
        
        # Get input name
        input_name = self.session.get_inputs()[0].name
        
        # Run inference - get all outputs
        output_names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.run(output_names, {input_name: input_batch})
        
        # Extract hidden states for requested layers
        result = {}
        for layer_idx in self.layers:
            layer_key = f"layer_{layer_idx}"
            encoder_idx = layer_idx - 1  # Convert to 0-indexed
            
            # Look for the encoder output by name pattern
            target_pattern = f"/encoder{encoder_idx}/ffn/skip"
            
            for out_name, out_value in zip(output_names, outputs):
                if out_name == target_pattern:
                    # Get the hidden state - shape should be (64, 768)
                    # Intermediate outputs don't have a batch dimension
                    result[layer_key] = out_value
                    break
            else:
                # Fallback: look for any output containing this encoder number
                for out_name, out_value in zip(output_names, outputs):
                    if f"encoder{encoder_idx}" in out_name:
                        result[layer_key] = out_value
                        break
        
        return result
    
    def extract_batch(self, boards: list[chess.Board]) -> dict[str, np.ndarray]:
        """
        Extract hidden states for a batch of positions.
        
        Args:
            boards: List of chess.Board objects with history
            
        Returns:
            Dictionary with keys like "layer_4", "layer_8", etc.
            Each value has shape (batch_size, 64, 768)
        """
        batch_size = len(boards)
        
        # Convert all boards to inputs
        inputs = np.stack([self.board_to_input(b) for b in boards])
        
        input_name = self.session.get_inputs()[0].name
        output_names = [o.name for o in self.session.get_outputs()]
        outputs = self.session.run(output_names, {input_name: inputs})
        
        # Extract hidden states using name-based matching
        # NOTE: ONNX outputs are FLATTENED: (batch*64, 768) instead of (batch, 64, 768)
        # We need to reshape them back to (batch, 64, 768)
        result = {}
        for layer_idx in self.layers:
            layer_key = f"layer_{layer_idx}"
            encoder_idx = layer_idx - 1  # Convert to 0-indexed
            
            # Look for the encoder output by name pattern
            target_pattern = f"/encoder{encoder_idx}/ffn/skip"
            
            for out_name, out_value in zip(output_names, outputs):
                if out_name == target_pattern:
                    # ONNX returns flattened shape: (batch*64, 768)
                    # Reshape to (batch, 64, 768)
                    if out_value.shape[0] == batch_size * 64:
                        out_value = out_value.reshape(batch_size, 64, -1)
                    result[layer_key] = out_value
                    break
            else:
                # Fallback: look for any output containing this encoder number
                for out_name, out_value in zip(output_names, outputs):
                    if f"encoder{encoder_idx}" in out_name:
                        if out_value.shape[0] == batch_size * 64:
                            out_value = out_value.reshape(batch_size, 64, -1)
                        result[layer_key] = out_value
                        break
        
        return result


def test_extractor():
    """Test the LC0 hidden state extractor with a sample position."""
    print("Testing LC0 Hidden State Extractor...")
    
    # Create a test position
    board = chess.Board()
    
    # Play some moves to have history
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"]
    for move in moves:
        board.push_uci(move)
    
    print(f"\nTest position after {len(moves)} moves:")
    print(board)
    print(f"FEN: {board.fen()}")
    
    # Create extractor (will download network if needed)
    try:
        extractor = LC0HiddenStateExtractor()
        
        # Extract hidden states
        hidden_states = extractor.extract(board)
        
        print("\nExtracted hidden states:")
        for layer_name, states in hidden_states.items():
            print(f"  {layer_name}: shape={states.shape}, dtype={states.dtype}")
            print(f"    mean={states.mean():.4f}, std={states.std():.4f}")
    
    except Exception as e:
        print(f"\nError during extraction: {e}")
        print("\nThis is expected if lc0 CLI is not installed.")
        print("The module will guide you through the setup process.")


if __name__ == "__main__":
    test_extractor()
