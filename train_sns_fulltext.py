"""전체 발화 다음 토큰 학습 실험. 프로젝트 루트에서 실행하세요."""
import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
BATCH_SIZE = 4
MAX_EPOCHS = 5
LEARNING_RATE = 0.0003
SEED = 42
CONTEXT_LENGTH = 128


def load_examples(path, tokenizer, learn_all=False):
    examples = []
    with path.open(encoding="utf-8-sig") as file:
        for number, line in enumerate(file, 1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"]
            roles = [m["role"] for m in messages]
            if not messages or roles != ["user", "assistant"] * (len(messages) // 2):
                raise ValueError(f"{path.name}:{number}: 역할 순서 오류")
            ids, labels = [], []
            for message in messages:
                role = message["role"]
                turn = [
                    tokenizer.token_to_id(f"<{role}>"),
                    *tokenizer.encode(message["content"]).ids,
                    tokenizer.token_to_id("<eos>"),
                ]
                ids.extend(turn)
                labels.extend([-100, *turn[1:]] if role == "assistant" else [-100] * len(turn))
            if len(ids) > CONTEXT_LENGTH:
                raise ValueError(f"{path.name}:{number}: 128토큰 초과")
            if not any(label != -100 for label in labels[1:]):
                raise ValueError(f"{path.name}:{number}: 답변 없음")
            if learn_all:
                labels = ids.copy()
            examples.append((torch.tensor(ids), torch.tensor(labels)))
    if not examples:
        raise ValueError(f"데이터가 비어 있습니다: {path}")
    return examples


def make_batch(examples, indices, pad_id):
    selected = [examples[i] for i in indices]
    ids = pad_sequence([x[0] for x in selected], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([x[1] for x in selected], batch_first=True, padding_value=-100)
    mask = (ids != pad_id).long()
    count = int((labels[:, 1:] != -100).sum())
    return ids, labels, mask, count


def validation_loss(model, examples, pad_id, deadline):
    model.eval()
    total, tokens = 0.0, 0
    with torch.inference_mode():
        for start in range(0, len(examples), BATCH_SIZE):
            if time.perf_counter() >= deadline:
                return None
            indices = range(start, min(start + BATCH_SIZE, len(examples)))
            ids, labels, mask, count = make_batch(examples, indices, pad_id)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("검증 손실이 유한하지 않습니다.")
            total += loss.item() * count
            tokens += count
    return total / tokens


def save_model(model, tokenizer, folder):
    model.save_pretrained(str(folder))
    tokenizer.save(str(folder / "tokenizer.json"))



def save_generation_examples(model, tokenizer, folder, step):
    questions = [
        "안녕!",
        "오늘 날씨가 좋아서 산책하고 왔어.",
        "오늘 좀 피곤하네.",
        "주말에 영화 보려고 하는데 너는 영화 좋아해?",
        "화분이 많이 자라서 분갈이를 하려고 해.",
    ]
    model.eval()
    results = []
    with torch.inference_mode():
        for question in questions:
            prompt = [
                tokenizer.token_to_id("<user>"),
                *tokenizer.encode(question).ids,
                tokenizer.token_to_id("<eos>"),
                tokenizer.token_to_id("<assistant>"),
            ]
            inputs = torch.tensor([prompt], dtype=torch.long)
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=min(48, CONTEXT_LENGTH - len(prompt)),
                do_sample=False,
            )
            answer = tokenizer.decode(
                output[0, len(prompt):].tolist(), skip_special_tokens=True
            ).strip()
            results.append({"question": question, "answer": answer})
    (folder / "generation_examples.json").write_text(
        json.dumps({"step": step, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/sns_30min")
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/sns-tokenizer-v1/tokenizer.json")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/sns-fulltext-v1")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"이미 폴더가 있습니다. --output으로 다른 이름을 지정하세요: {args.output}")
    started = time.perf_counter()
    training_deadline = started + 20 * 60
    validation_deadline = started + 28 * 60
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    for token in ("<user>", "<assistant>", "<eos>", "<pad>"):
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"필요한 토큰이 없습니다: {token}")
    report_path = args.data / "report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        fingerprint = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
        if fingerprint != report["tokenizer_sha256"]:
            raise ValueError("전처리에 사용한 토크나이저와 다릅니다.")

    train = load_examples(args.data / "train.jsonl", tokenizer, learn_all=True)
    validation = load_examples(args.data / "validation.jsonl", tokenizer)
    full_validation = load_examples(args.data / "validation.jsonl", tokenizer, learn_all=True)
    pad_id = tokenizer.token_to_id("<pad>")
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=CONTEXT_LENGTH,
        n_embd=128,
        n_layer=4,
        n_head=4,
        bos_token_id=None,
        eos_token_id=tokenizer.token_to_id("<eos>"),
        pad_token_id=pad_id,
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    args.output.mkdir(parents=True)
    metadata = {
        "objective": "all_real_tokens",
        "selection_metric": "response_validation_loss",
        "initialization": "random",
        "train_count": len(train),
        "validation_count": len(validation),
        "batch_size": BATCH_SIZE,
        "max_epochs": MAX_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "seed": SEED,
        "device": "cpu",
        "threads": 4,
        "parameters": sum(p.numel() for p in model.parameters()),
        "context_length": CONTEXT_LENGTH,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256((args.data / "train.jsonl").read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256((args.data / "validation.jsonl").read_bytes()).hexdigest(),
        "history": [],
        "best_validation_loss": None,
        "best_step": None,
    }
    print(f"CPU | 학습 {len(train)} | 검증 {len(validation)} | 배치 {BATCH_SIZE}", flush=True)
    print("학습 예산 20분, 전체 마감 목표 30분. 검증·저장 시간을 별도로 확보합니다.", flush=True)
    best = math.inf
    steps = 0
    last_saved_step = 0
    stop_reason = "max_epochs"
    for epoch in range(1, MAX_EPOCHS + 1):
        order = list(range(len(train)))
        random.shuffle(order)
        total, tokens = 0.0, 0
        model.train()
        for start in range(0, len(order), BATCH_SIZE):
            if time.perf_counter() >= training_deadline:
                stop_reason = "training_time_budget"
                break
            ids, labels, mask, count = make_batch(train, order[start:start + BATCH_SIZE], pad_id)
            optimizer.zero_grad()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("학습 손실이 유한하지 않습니다.")
            loss.backward()
            optimizer.step()
            steps += 1
            total += loss.item() * count
            tokens += count
            if steps % 100 == 0:
                print(f"{epoch}회차 | 전체 {steps}스텝 | 경과 {(time.perf_counter()-started)/60:.1f}분", flush=True)
            if steps % 500 == 0:
                save_model(model, tokenizer, args.output / "latest")
                last_saved_step = steps
        if tokens == 0:
            break
        # 기존 방식과 비교할 답변 부분 손실을 우선 측정합니다.
        value = validation_loss(model, validation, pad_id, validation_deadline)
        full_value = validation_loss(model, full_validation, pad_id, validation_deadline)
        row = {
            "epoch": epoch,
            "step": steps,
            "epoch_complete": start + BATCH_SIZE >= len(order) and stop_reason == "max_epochs",
            "train_loss": total / tokens,
            "response_validation_loss": value,
            "fulltext_validation_loss": full_value,
        }
        metadata["history"].append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        if value is not None and value < best:
            best = value
            save_model(model, tokenizer, args.output)
            metadata["best_validation_loss"] = value
            metadata["best_step"] = steps
            save_generation_examples(model, tokenizer, args.output, steps)
            print("최저 답변 검증 손실 모델과 생성 예시 저장", flush=True)
        (args.output / "training_log.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if stop_reason == "training_time_budget" or value is None:
            if value is None:
                stop_reason = "validation_time_budget"
            break

    if steps != last_saved_step and steps:
        save_model(model, tokenizer, args.output / "latest")
    metadata.update({
        "stop_reason": stop_reason,
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
    })
    (args.output / "training_log.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("완료:", args.output, flush=True)
    print("최저 검증 손실 모델 스텝:", metadata["best_step"], flush=True)


if __name__ == "__main__":
    main()
