"""SNS 대화 학습과 검증. 프로젝트 루트에 두고 실행합니다."""
import json
import math
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "sns_auto"
TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer.json"
MODEL_DIR = ROOT / "artifacts" / "tiny-chat-model-sns-v2"

EPOCHS = 3
CONTEXT_LENGTH = 256
LEARNING_RATE = 0.0003
SEED = 42


def load_examples(path, tokenizer):
    examples = []
    with path.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            roles = [m["role"] for m in messages]
            if not messages or roles != ["user", "assistant"] * (len(messages) // 2):
                raise ValueError(f"{path.name}:{line_number}: 역할 순서를 확인하세요.")
            ids, labels = [], []
            for message in messages:
                role = message["role"]
                content = tokenizer.encode(message["content"]).ids
                turn = [
                    tokenizer.token_to_id(f"<{role}>"),
                    *content,
                    tokenizer.token_to_id("<eos>"),
                ]
                ids.extend(turn)
                labels.extend(
                    [-100, *turn[1:]] if role == "assistant" else [-100] * len(turn)
                )
            if len(ids) > CONTEXT_LENGTH:
                raise ValueError(f"{path.name}:{line_number}: 토큰 한도 초과")
            input_ids = torch.tensor([ids], dtype=torch.long)
            targets = torch.tensor([labels], dtype=torch.long)
            count = int((targets[:, 1:] != -100).sum())
            if not count:
                raise ValueError(f"{path.name}:{line_number}: 학습할 답변이 없습니다.")
            examples.append((input_ids, targets, count))
    if not examples:
        raise ValueError(f"비어 있는 데이터: {path}")
    return examples


def run_epoch(model, examples, optimizer=None):
    training = optimizer is not None
    model.train(training)
    order = list(range(len(examples)))
    if training:
        random.shuffle(order)

    total_loss = 0.0
    total_tokens = 0
    with torch.set_grad_enabled(training):
        for step, index in enumerate(order, 1):
            inputs, labels, count = examples[index]
            if training:
                optimizer.zero_grad()
            loss = model(input_ids=inputs, labels=labels).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("손실이 유한하지 않습니다. 학습을 중단합니다.")
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * count
            total_tokens += count
            if training and step % 200 == 0:
                print(f"  학습 진행 {step}/{len(order)}", flush=True)
    return total_loss / total_tokens


def main():
    if MODEL_DIR.exists():
        raise FileExistsError(
            f"이미 모델 폴더가 있습니다: {MODEL_DIR}\n"
            "기존 결과를 보존하려면 MODEL_DIR을 새 이름으로 바꾸세요."
        )

    random.seed(SEED)
    torch.manual_seed(SEED)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    for token in ("<user>", "<assistant>", "<eos>", "<pad>"):
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"필요한 토큰이 없습니다: {token}")

    train = load_examples(DATA_DIR / "train.jsonl", tokenizer)
    validation = load_examples(DATA_DIR / "validation.jsonl", tokenizer)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=CONTEXT_LENGTH,
        n_embd=64,
        n_layer=2,
        n_head=4,
        bos_token_id=None,
        eos_token_id=tokenizer.token_to_id("<eos>"),
        pad_token_id=tokenizer.token_to_id("<pad>"),
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print(f"CPU 실행 | 학습 {len(train)}개 | 검증 {len(validation)}개", flush=True)

    initial_loss = run_epoch(model, validation)
    print(f"학습 전 검증 손실: {initial_loss:.4f}", flush=True)
    history = []
    best_loss = math.inf
    best_epoch = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = run_epoch(model, train, optimizer)
        validation_loss = run_epoch(model, validation)
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
        })
        print(
            f"{epoch}/{EPOCHS}회 | 학습 {train_loss:.4f} | 검증 {validation_loss:.4f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            model.save_pretrained(str(MODEL_DIR))
            tokenizer.save(str(MODEL_DIR / "tokenizer.json"))
            print("  최저 검증 손실 모델 저장", flush=True)

        report = {
            "initial_validation_loss": initial_loss,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
            "history": history,
            "train_count": len(train),
            "validation_count": len(validation),
            "seed": SEED,
            "learning_rate": LEARNING_RATE,
            "context_length": CONTEXT_LENGTH,
            "vocab_size": tokenizer.get_vocab_size(),
            "device": "cpu",
        }
        (MODEL_DIR / "training_log.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(f"완료 | 채택 회차 {best_epoch} | 저장 위치 {MODEL_DIR}", flush=True)


if __name__ == "__main__":
    main()
