from pathlib import Path
import sys

import chess
import torch


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.chess_adapter import ENGINEERED_FEATURE_DIM, ENGINEERED_PIECE_OFFSET, ENGINEERED_EMPTY_PIECE_INDEX, extract_engineered_features
from training import train as train_module
from training.config import TrainingConfig
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


class BudgetTokenizer:
    pad_token_id = 0

    def __call__(self, text, truncation=False, padding=False, return_tensors=None):
        token_count = max(1, len(self.encode(text)))
        input_ids = torch.arange(1, token_count + 1, dtype=torch.long).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    def encode(self, text):
        return [idx + 1 for idx, token in enumerate(text.split()) if token]


def _make_budget_dataset(monkeypatch, training_config: TrainingConfig | None = None) -> ChessCommentaryTrainingDataset:
    monkeypatch.setattr(
        ChessCommentaryTrainingDataset,
        "_print_supervision_probe",
        lambda self, max_scan=512: None,
    )
    monkeypatch.setattr(
        train_module.Path,
        "glob",
        lambda self, pattern: [],
        raising=False,
    )
    monkeypatch.setattr(
        train_module,
        "_safe_apply_chat_template",
        lambda tokenizer, messages, add_generation_prompt=True: "Prompt:",
    )
    return ChessCommentaryTrainingDataset(
        samples_dir="chess_fusion_training",
        tokenizer=BudgetTokenizer(),
        source_tag="test",
        training_config=training_config,
    )


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


def test_structured_commentary_trimming_preserves_protected_engine_section(monkeypatch):
    dataset = _make_budget_dataset(monkeypatch)

    sections = [
        {
            "kind": "overview",
            "priority": 100,
            "protected": False,
            "text": "Overview intro sentence. Extra overview sentence.",
        },
        {
            "kind": "engine_eval_top_moves",
            "priority": 10,
            "protected": True,
            "text": (
                "Evaluation and top continuations: The position is winning for Black. "
                "The best line for Black is Rh4 Ng2. The close second is Rd4."
            ),
        },
        {
            "kind": "recent_context",
            "priority": 200,
            "protected": False,
            "text": "Recent game context: 1. e4 e5 2. Nf3 Nc6.",
        },
    ]
    commentary = dataset._join_commentary_sections(sections)
    engine_only = dataset._join_commentary_sections([sections[1]])
    budget = len(dataset.tokenizer.encode("Prompt:" + engine_only)) + 1

    fitted = dataset._fit_commentary_to_budget(
        prompt_text="Prompt:",
        commentary=commentary,
        effective_text_budget=budget,
        idx=0,
        commentary_sections=sections,
    )

    assert "Evaluation and top continuations:" in fitted
    assert "The best line for Black is Rh4 Ng2." in fitted
    assert "Overview intro sentence." not in fitted
    assert "Recent game context:" not in fitted


def test_legacy_commentary_trimming_keeps_existing_sentence_front_trim(monkeypatch):
    dataset = _make_budget_dataset(monkeypatch)

    commentary = "First sentence. Second important sentence. Third sentence."
    budget = len(dataset.tokenizer.encode("Prompt:Second important sentence. Third sentence.")) + 1

    fitted = dataset._fit_commentary_to_budget(
        prompt_text="Prompt:",
        commentary=commentary,
        effective_text_budget=budget,
        idx=0,
        commentary_sections=None,
    )

    assert fitted == "Second important sentence. Third sentence."


def test_engine_only_training_flag_discards_factoid_sections(monkeypatch):
    dataset = _make_budget_dataset(
        monkeypatch,
        training_config=TrainingConfig(pretrain_engine_outlook_only=True),
    )

    sections = [
        {
            "kind": "overview",
            "priority": 100,
            "protected": False,
            "text": "Overview intro sentence. Extra overview sentence.",
        },
        {
            "kind": "engine_eval_top_moves",
            "priority": 10,
            "protected": True,
            "text": "Engine outlook: The position is winning for Black. The best line for Black is Rh4 Ng2.",
        },
        {
            "kind": "recent_context",
            "priority": 200,
            "protected": False,
            "text": "Recent game context: 1. e4 e5 2. Nf3 Nc6.",
        },
    ]

    commentary = dataset._join_commentary_sections(sections)
    selected_commentary, selected_sections = dataset._select_commentary_for_training(
        commentary,
        sections,
    )

    assert selected_commentary == sections[1]["text"]
    assert selected_sections == [sections[1]]
