"""토크나이저 출력 정규화와 SFT 라벨 마스킹 회귀 테스트.

`apply_chat_template(tokenize=True)`의 반환형이 transformers 버전에 따라 다르다.
스텁 토크나이저가 list[int]를 반환하도록 만들어 두는 바람에 실제 환경에서만
`TypeError: unsupported operand type(s) for +: 'BatchEncoding' and 'list'`로
터졌다. 스텁을 실물 모양에 맞추지 않으면 검증이 통과해도 의미가 없다.

torch/peft 없이 돌도록 train_lora에서 필요한 것만 직접 로드한다.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path

import pytest

IGNORE = -100


def _load_train_lora():
    """train_lora는 torch/peft를 임포트하므로, 없으면 이 테스트를 건너뛴다."""
    spec = importlib.util.spec_from_file_location(
        "train_lora", Path(__file__).resolve().parents[1] / "finetune" / "train_lora.py"
    )
    module = importlib.util.module_from_spec(spec)
    # @dataclass가 정의 시점에 sys.modules[__name__]을 찾는다. 등록하지 않으면
    # Example 데코레이션에서 AttributeError로 죽는다.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as e:
        del sys.modules[spec.name]
        pytest.skip(f"학습 의존성 없음: {e}")
    return module


class FakeBatchEncoding(Mapping):
    """BatchEncoding은 dict가 아니라 UserDict다 — isinstance(x, dict)로는 안 잡힌다."""

    def __init__(self, data: dict):
        self._data = data

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)


@pytest.mark.parametrize(
    "encoded",
    [
        [1, 2, 3],
        FakeBatchEncoding({"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}),
        FakeBatchEncoding({"input_ids": [[1, 2, 3]]}),
        [[1, 2, 3]],
    ],
    ids=["list", "mapping", "mapping-batched", "nested-list"],
)
def test_as_ids_normalizes_tokenizer_output(encoded):
    assert _load_train_lora()._as_ids(encoded) == [1, 2, 3]


def test_batch_encoding_is_not_a_dict():
    """dict 검사로 분기하면 안 되는 이유를 고정해 둔다."""
    assert not isinstance(FakeBatchEncoding({"input_ids": [1]}), dict)
    assert isinstance(FakeBatchEncoding({"input_ids": [1]}), Mapping)


class StubTokenizer:
    """실제 토크나이저처럼 Mapping을 돌려준다."""

    eos_token_id = 2
    pad_token_id = 0

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        text = "".join(f"<{m['role']}>{m['content']}" for m in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return FakeBatchEncoding({"input_ids": [3 + (ord(c) % 900) for c in text]})

    def __call__(self, text, add_special_tokens=False):
        return FakeBatchEncoding({"input_ids": [3 + (ord(c) % 900) for c in text]})


def _write_sft(tmp_path: Path, answer: str) -> Path:
    row = {
        "messages": [
            {"role": "system", "content": "규칙"},
            {"role": "user", "content": "[1] 근거\n\n질문: 연차는?"},
            {"role": "assistant", "content": answer},
        ],
        "kind": "answerable",
    }
    path = tmp_path / "sft.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_only_answer_tokens_are_supervised(tmp_path):
    """프롬프트는 전부 마스킹되고 답변+EOS에만 손실이 걸려야 한다."""
    module = _load_train_lora()
    tok = StubTokenizer()
    answer = "연차는 15일입니다. [1]"
    ds = module.ChatSFTDataset(_write_sft(tmp_path, answer), tok, max_len=10_000)

    expected = [3 + (ord(c) % 900) for c in answer] + [tok.eos_token_id]
    example = ds.examples[0]

    assert len(example.input_ids) == len(example.labels)
    assert example.labels[-len(expected) :] == expected
    assert example.input_ids[-len(expected) :] == expected
    assert all(x == IGNORE for x in example.labels[: -len(expected)])


def test_truncation_preserves_the_answer(tmp_path):
    """max_len을 넘으면 프롬프트 앞을 자른다 — 답변이 잘리면 학습 신호가 사라진다."""
    module = _load_train_lora()
    tok = StubTokenizer()
    answer = "연차는 15일입니다. [1]"
    ds = module.ChatSFTDataset(_write_sft(tmp_path, answer), tok, max_len=20)

    expected = [3 + (ord(c) % 900) for c in answer] + [tok.eos_token_id]
    example = ds.examples[0]

    assert len(example.input_ids) == 20
    assert ds.n_truncated == 1
    assert example.labels[-len(expected) :] == expected


def test_collate_pads_without_leaking_loss(tmp_path):
    """패딩 자리는 손실에서 빠지고 attention도 0이어야 한다."""
    module = _load_train_lora()
    tok = StubTokenizer()
    ds = module.ChatSFTDataset(_write_sft(tmp_path, "짧다. [1]"), tok, max_len=10_000)
    short = ds.examples[0]
    long = module.Example(
        input_ids=short.input_ids + [7, 7, 7], labels=short.labels + [7, 7, 7]
    )

    batch = module.collate([short, long], pad_id=tok.pad_token_id)
    width = batch["input_ids"].shape[1]

    assert width == len(long.input_ids)
    assert batch["labels"][0, len(short.input_ids) :].tolist() == [IGNORE] * 3
    assert batch["attention_mask"][0, len(short.input_ids) :].tolist() == [0] * 3
    assert batch["attention_mask"][0, : len(short.input_ids)].tolist() == [1] * len(short.input_ids)
