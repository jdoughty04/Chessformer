from pathlib import Path
import sys

import chess
import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.chess_adapter import ENGINEERED_FEATURE_DIM, ENGINEERED_PIECE_OFFSET, ENGINEERED_EMPTY_PIECE_INDEX, extract_engineered_features
from training import train as train_module
from training.train import ChessCommentaryTrainingDataset


class DummyTokenizer:
    pad_token_id = 0

    def __call__(self, text, truncation=False, padding=False, return_tensors=None):
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def encode(self, text):
        return [1, 2, 3]


def test_dataset_extracts_main_engineered_features_for_chess_fusion_source_without_maia(monkeypatch):
    sample = {
        "fen": "8/8/8/8/8/8/8/K6k w - - 0 1",
        "commentary": "White keeps control.",
    }
    feature_modes = []

    monkeypatch.setattr(
        ChessCommentaryTrainingDataset,
        "_print_supervision_probe",
        lambda self, max_scan=512: None,
    )
    monkeypatch.setattr(
        train_module.Path,
        "glob",
        lambda self, pattern: [self / "sample0.pt"],
        raising=False,
    )
    monkeypatch.setattr(
        train_module,
        "load_training_sample",
        lambda path: sample,
    )
    monkeypatch.setattr(
        train_module,
        "_safe_apply_chat_template",
        lambda tokenizer, messages, add_generation_prompt=True: "Prompt: ",
    )

    def fake_extract_engineered_features(fen: str, mode: str = "simplified") -> torch.Tensor:
        feature_modes.append(mode)
        fill_value = 7.0 if mode == "main" else 1.0
        return torch.full((64, ENGINEERED_FEATURE_DIM), fill_value, dtype=torch.float32)

    monkeypatch.setattr(
        train_module,
        "extract_engineered_features",
        fake_extract_engineered_features,
    )

    dataset = ChessCommentaryTrainingDataset(
        samples_dir="chess_fusion_training",
        tokenizer=DummyTokenizer(),
        use_maia_features=False,
        use_chess_fusion_main_engineered_source=True,
        source_tag="test",
    )

    item = dataset[0]

    assert feature_modes == ["main"]
    assert "engineered_features" in item
    assert item["engineered_features"].shape == (64, ENGINEERED_FEATURE_DIM)
    assert torch.all(item["engineered_features"] == 7.0)


def test_extract_engineered_features_main_marks_empty_squares_with_explicit_piece_channel():
    features = extract_engineered_features("8/8/8/8/8/8/8/K6k w - - 0 1", mode="main")

    assert features.shape == (64, ENGINEERED_FEATURE_DIM)
    assert features[chess.A1, ENGINEERED_PIECE_OFFSET + ENGINEERED_EMPTY_PIECE_INDEX].item() == 0.0
    assert features[chess.B1, ENGINEERED_PIECE_OFFSET + ENGINEERED_EMPTY_PIECE_INDEX].item() == 1.0
