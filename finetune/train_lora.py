"""RAG 형식 계약을 작은 모델에 LoRA로 가르친다.

`build_dataset.py`가 만든 chat 형식 JSONL을 그대로 먹는다. 학습은 **답변 토큰에만**
손실을 건다 — 프롬프트(시스템 규칙 + 검색된 청크)까지 학습시키면 모델이 문서 내용을
외우는 쪽으로 새고, 우리가 원하는 건 "주어진 근거로 형식을 지켜 답하는 행동"이다.

T4(Turing)를 가정한다. bf16이 없는 세대라 fp16으로 돌린다.

사용:
    python finetune/train_lora.py --model Qwen/Qwen2.5-0.5B-Instruct --out finetune/out/lora
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

IGNORE = -100


@dataclass
class Example:
    input_ids: list[int]
    labels: list[int]


class ChatSFTDataset(Dataset):
    """chat JSONL -> (input_ids, labels). 답변 구간 밖은 전부 IGNORE로 마스킹한다."""

    def __init__(self, path: Path, tokenizer, max_len: int):
        lines = path.read_text(encoding="utf-8").splitlines()
        self.rows = [json.loads(x) for x in lines if x.strip()]
        self.tok = tokenizer
        self.max_len = max_len
        self.n_truncated = 0
        self.examples = [self._encode(r) for r in self.rows]

    def _encode(self, row: dict) -> Example:
        messages = row["messages"]
        prompt_msgs = [m for m in messages if m["role"] != "assistant"]
        answer = next(m["content"] for m in messages if m["role"] == "assistant")

        # 추론 시와 같은 템플릿을 써야 학습이 실제 사용 형태와 어긋나지 않는다.
        prompt_ids = self.tok.apply_chat_template(
            prompt_msgs, add_generation_prompt=True, tokenize=True
        )
        answer_ids = self.tok(answer, add_special_tokens=False)["input_ids"]
        if self.tok.eos_token_id is not None:
            answer_ids = answer_ids + [self.tok.eos_token_id]

        input_ids = prompt_ids + answer_ids
        labels = [IGNORE] * len(prompt_ids) + answer_ids[:]

        if len(input_ids) > self.max_len:
            # 앞(프롬프트)을 잘라 답변은 온전히 남긴다. 답변이 잘리면 학습 신호가 망가진다.
            cut = len(input_ids) - self.max_len
            input_ids = input_ids[cut:]
            labels = labels[cut:]
            self.n_truncated += 1

        return Example(input_ids=input_ids, labels=labels)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Example:
        return self.examples[i]


def collate(batch: list[Example], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(e.input_ids) for e in batch)
    input_ids, labels, attn = [], [], []
    for e in batch:
        pad = width - len(e.input_ids)
        input_ids.append(e.input_ids + [pad_id] * pad)
        labels.append(e.labels + [IGNORE] * pad)
        attn.append([1] * len(e.input_ids) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG 형식 계약 LoRA 학습")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--train-file", default="finetune/data/sft.jsonl")
    ap.add_argument("--out", default="finetune/out/lora")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    ds = ChatSFTDataset(Path(args.train_file), tok, args.max_len)
    print(f"학습 샘플 {len(ds)}건 / max_len {args.max_len}")
    if ds.n_truncated:
        print(f"  주의: {ds.n_truncated}건이 max_len을 넘어 프롬프트 앞부분이 잘렸다")

    n_answer_tokens = sum(sum(1 for x in e.labels if x != IGNORE) for e in ds.examples)
    print(f"  손실이 걸리는 답변 토큰 {n_answer_tokens:,}개 (프롬프트는 마스킹됨)")

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map=None,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=5,
        save_strategy="no",
        # T4는 Turing이라 bf16이 없다.
        fp16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        report_to=[],
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=lambda b: collate(b, tok.pad_token_id),
    )
    trainer.train()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    print(f"\n어댑터 저장: {out}")


if __name__ == "__main__":
    main()
