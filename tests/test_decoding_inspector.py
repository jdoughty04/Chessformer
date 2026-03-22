from pathlib import Path
import sys
from types import SimpleNamespace

import torch
import torch.nn as nn


SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from inference.decoding_inspector import DecodeSession
from training.train import ChessCommentaryModel


class ScriptedOutput:
    def __init__(self, logits: torch.Tensor, past_key_values):
        self.logits = logits
        self.past_key_values = past_key_values


class StubTokenizer:
    def __init__(self, vocab_size: int = 6, eos_token_id: int = 4):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = 0

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return f"tok{int(token_id)}"

    def decode(self, token_ids, skip_special_tokens: bool = False, clean_up_tokenization_spaces: bool = False):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        pieces = []
        for token_id in token_ids:
            token_id = int(token_id)
            if skip_special_tokens and token_id == self.eos_token_id:
                continue
            pieces.append(f"tok{token_id}")
        return " ".join(pieces)


def make_trace(layer_idx: int, slot_index: int):
    raw_slot_weights = torch.zeros(192, dtype=torch.float32)
    raw_slot_weights[slot_index] = 1.0
    source_square_weights = raw_slot_weights.view(3, 64)
    return {
        layer_idx: {
            "raw_slot_weights": raw_slot_weights,
            "source_square_weights": source_square_weights,
            "aggregate_square_weights": source_square_weights.sum(dim=0),
            "global_weights": torch.tensor([0.75, 0.25], dtype=torch.float32),
            "last_token_index": torch.tensor(0, dtype=torch.long),
            "csmp_square_weights": source_square_weights[0].clone(),
            "perceiver_square_weights": source_square_weights[1].clone(),
            "policy_square_weights": source_square_weights[2].clone(),
        }
    }


class StubAdapter:
    def __init__(self, trace_sequence):
        self.xattn_layer_indices = [7]
        self.trace_sequence = list(trace_sequence)
        self.current_trace = self.trace_sequence[0] if self.trace_sequence else {}
        self.clear_calls = 0

    def clear_last_token_traces(self) -> None:
        self.clear_calls += 1

    def get_last_token_structured_traces(self, sample_index: int = 0):
        return self.current_trace


class ScriptedLLM(nn.Module):
    def __init__(self, logits_sequence, adapter: StubAdapter):
        super().__init__()
        self.embedding = nn.Embedding(8, 4)
        self.logits_sequence = [
            torch.tensor(logits, dtype=torch.float32).view(1, 1, -1)
            for logits in logits_sequence
        ]
        self.adapter = adapter
        self.calls = []

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, **kwargs):
        call_index = len(self.calls)
        self.calls.append(kwargs)
        if self.adapter.trace_sequence:
            self.adapter.current_trace = self.adapter.trace_sequence[
                min(call_index, len(self.adapter.trace_sequence) - 1)
            ]
        logits = self.logits_sequence[min(call_index, len(self.logits_sequence) - 1)]
        return ScriptedOutput(logits=logits, past_key_values=f"pkv-{call_index}")


class StubModel:
    def __init__(self, logits_sequence, trace_sequence):
        self.tokenizer = StubTokenizer(vocab_size=len(logits_sequence[0]), eos_token_id=4)
        self.adapter = StubAdapter(trace_sequence)
        self.llm = ScriptedLLM(logits_sequence, self.adapter)
        self.num_prefix_tokens = 0
        self.prepare_calls = []
        self.set_context_calls = []
        self.clear_context_calls = 0

    def prepare_generation_inputs(self, **kwargs):
        self.prepare_calls.append(kwargs)
        return {
            "generation_context": {"kind": "stub"},
            "combined_embeds": torch.zeros(1, 3, 4),
            "combined_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        }

    def set_generation_context(self, generation_context, text_attention_mask=None):
        cloned_mask = None if text_attention_mask is None else text_attention_mask.clone()
        self.set_context_calls.append((generation_context, cloned_mask))

    def clear_generation_context(self):
        self.clear_context_calls += 1

    def _build_position_ids(self, attention_mask: torch.Tensor, prefix_len: int = 0) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        return position_ids


class CountingContext:
    def __init__(self, events):
        self.events = events

    def __enter__(self):
        self.events.append("enter")
        return self

    def __exit__(self, exc_type, exc, tb):
        self.events.append("exit")
        return False


def _build_session(logits_sequence):
    trace_sequence = [make_trace(7, idx) for idx in range(len(logits_sequence))]
    return DecodeSession(StubModel(logits_sequence, trace_sequence))


def test_decode_session_prime_returns_top_tokens_before_any_step():
    session = _build_session(
        [
            [0.1, 0.4, 2.5, 0.3, -0.2, 0.0],
            [0.0, 0.2, 0.1, 1.8, -0.5, -0.1],
        ]
    )

    snapshot = session.start("8/8/8/8/8/8/8/K6k w - - 0 1", "Inspect this")

    assert snapshot["emitted_token_ids"] == []
    assert len(snapshot["top_tokens"]) == 5
    assert snapshot["top_tokens"][0]["token_id"] == 2
    assert snapshot["available_layers"] == [7]
    assert "7" in snapshot["layer_traces"]


def test_decode_session_greedy_step_appends_exactly_one_token():
    session = _build_session(
        [
            [0.1, 0.4, 2.5, 0.3, -0.2, 0.0],
            [0.0, 0.2, 0.1, 1.8, -0.5, -0.1],
        ]
    )
    session.start("8/8/8/8/8/8/8/K6k w - - 0 1", "Inspect this")

    snapshot = session.step()

    assert snapshot["emitted_token_ids"] == [2]
    assert snapshot["emitted_tokens"] == ["tok2"]
    assert snapshot["top_tokens"][0]["token_id"] == 3


def test_decode_session_forced_step_honors_selected_token():
    session = _build_session(
        [
            [0.1, 0.4, 2.5, 0.3, -0.2, 0.0],
            [0.0, 0.2, 0.1, 1.8, -0.5, -0.1],
        ]
    )
    session.start("8/8/8/8/8/8/8/K6k w - - 0 1", "Inspect this")

    snapshot = session.step(token_id=1)

    assert snapshot["emitted_token_ids"] == [1]
    assert snapshot["emitted_tokens"] == ["tok1"]


def test_decode_session_eos_disables_future_steps():
    session = _build_session(
        [
            [0.0, -0.2, 0.1, 0.3, 4.0, -0.5],
            [0.1, 0.2, 0.3, 0.4, -0.1, -0.2],
        ]
    )
    session.start("8/8/8/8/8/8/8/K6k w - - 0 1", "Inspect this")

    after_first = session.step()
    after_second = session.step()

    assert after_first["emitted_token_ids"] == [4]
    assert after_first["eos_reached"] is True
    assert after_second["emitted_token_ids"] == [4]
    assert len(session.model.llm.calls) == 2


def test_decode_session_wraps_llm_calls_in_autocast_context():
    session = _build_session(
        [
            [0.1, 0.4, 2.5, 0.3, -0.2, 0.0],
            [0.0, 0.2, 0.1, 1.8, -0.5, -0.1],
        ]
    )
    events = []
    session._llm_autocast_context = lambda: CountingContext(events)

    session.start("8/8/8/8/8/8/8/K6k w - - 0 1", "Inspect this")
    session.step()

    assert events == ["enter", "exit", "enter", "exit"]


class FakeGenerateLLM:
    def __init__(self):
        self.generate_calls = []

    def generate(self, **kwargs):
        self.generate_calls.append(kwargs)
        return torch.tensor([[1, 2, 4]], dtype=torch.long)


class FakeGenerateSelf:
    def __init__(self):
        self.config = SimpleNamespace(mode="chess_fusion")
        self.lm_enabled = True
        self.tokenizer = StubTokenizer(vocab_size=6, eos_token_id=4)
        self.llm = FakeGenerateLLM()
        self.prepare_calls = []
        self.context_calls = []
        self.clear_calls = 0
        self.eval_called = False

    def eval(self):
        self.eval_called = True

    def prepare_generation_inputs(self, **kwargs):
        self.prepare_calls.append(kwargs)
        return {
            "generation_context": {"kind": "context"},
            "combined_embeds": torch.zeros(1, 3, 4),
            "combined_mask": torch.ones(1, 3, dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
        }

    def set_generation_context(self, generation_context, text_attention_mask=None):
        cloned_mask = None if text_attention_mask is None else text_attention_mask.clone()
        self.context_calls.append((generation_context, cloned_mask))

    def clear_generation_context(self):
        self.clear_calls += 1


def test_generate_uses_shared_generation_prep_and_clears_context():
    fake_self = FakeGenerateSelf()

    commentary, generated_ids = ChessCommentaryModel.generate(
        fake_self,
        lc0_hidden_states=None,
        side_to_move=True,
        prompt="Prompt text",
        fen="8/8/8/8/8/8/8/K6k w - - 0 1",
        return_ids=True,
    )

    assert fake_self.eval_called is True
    assert fake_self.prepare_calls[0]["prompt"] == "Prompt text"
    assert fake_self.context_calls[0][0] == {"kind": "context"}
    assert torch.equal(fake_self.context_calls[0][1], torch.ones(1, 3, dtype=torch.long))
    assert fake_self.clear_calls == 1
    assert fake_self.llm.generate_calls[0]["inputs_embeds"].shape == (1, 3, 4)
    assert commentary == "tok1 tok2"
    assert generated_ids.tolist() == [1, 2, 4]
