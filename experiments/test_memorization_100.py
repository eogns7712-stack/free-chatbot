"""SNS 학습 데이터 20개를 재현하는지 검사하는 실험입니다."""

# 프로젝트 루트에서 실행하되, 이동 후에도 데이터·모델 경로와 공통 함수 import를 유지합니다.
import sys as _sys
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(PROJECT_ROOT))

import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = PROJECT_ROOT
DATA_PATH = ROOT / "data" / "sns_auto" / "train.jsonl"
TOKENIZER_PATH = ROOT / "artifacts" / "sns-tokenizer-v1" / "tokenizer.json"
MODEL_DIR = ROOT / "artifacts" / "sns-memorization-100"

MAX_EPOCHS = 100
LEARNING_RATE = 0.0003
SEED = 42


def select_examples(tokenizer):
    selected = []
    seen_questions = set()
    with DATA_PATH.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            messages = json.loads(line)["messages"][:2]
            if [m["role"] for m in messages] != ["user", "assistant"]:
                continue
            question, answer = [m["content"] for m in messages]
            if len(question) < 8 or len(answer) < 8 or question in seen_questions:
                continue
            question_ids = tokenizer.encode(question).ids
            answer_ids = tokenizer.encode(answer).ids
            if len(question_ids) + len(answer_ids) + 4 > 96:
                continue
            selected.append({
                "source_line": line_number,
                "messages": messages,
            })
            seen_questions.add(question)
            if len(selected) == 100:
                break
    if len(selected) != 100:
        raise ValueError(
            f"조건에 맞는 대화가 100개보다 적습니다: {len(selected)}"
        )
    return selected


def make_tensors(record, tokenizer):
    question, answer = record["messages"]
    eos = tokenizer.token_to_id("<eos>")
    prompt = [
        tokenizer.token_to_id("<user>"),
        *tokenizer.encode(question["content"]).ids,
        eos,
        tokenizer.token_to_id("<assistant>"),
    ]
    response = [*tokenizer.encode(answer["content"]).ids, eos]
    ids = torch.tensor([prompt + response], dtype=torch.long)
    labels = torch.tensor(
        [[-100] * len(prompt) + response], dtype=torch.long
    )
    return ids, labels, prompt, response


def evaluate(model, records, tensors, tokenizer):
    model.eval()
    results = []
    with torch.inference_mode():
        for record, (_, _, prompt, expected_ids) in zip(records, tensors):
            inputs = torch.tensor([prompt], dtype=torch.long)
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=len(expected_ids) + 8,
                do_sample=False,
            )
            generated_ids = output[0, len(prompt):].tolist()
            generated = tokenizer.decode(
                generated_ids, skip_special_tokens=True
            ).strip()
            expected = record["messages"][1]["content"].strip()
            results.append({
                "question": record["messages"][0]["content"],
                "reference": expected,
                "generated": generated,
                "text_match": generated == expected,
                "token_match_including_eos": generated_ids == expected_ids,
            })
    return results


def main():
    if MODEL_DIR.exists():
        raise FileExistsError(f"이미 실험 폴더가 있습니다: {MODEL_DIR}")
    random.seed(SEED)
    torch.manual_seed(SEED)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    for token in ("<user>", "<assistant>", "<eos>", "<pad>"):
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"필요한 토큰이 없습니다: {token}")

    records = select_examples(tokenizer)
    tensors = [make_tensors(r, tokenizer) for r in records]
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=256,
        n_embd=64,
        n_layer=2,
        n_head=4,
        bos_token_id=None,
        eos_token_id=tokenizer.token_to_id("<eos>"),
        pad_token_id=tokenizer.token_to_id("<pad>"),
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    MODEL_DIR.mkdir(parents=True)
    (MODEL_DIR / "selected_examples.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8",
    )
    print(
        f"{len(records)}개 대화 | CPU | 최대 {MAX_EPOCHS}회 학습",
        flush=True,
    )
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = list(range(len(tensors)))
        random.shuffle(order)
        total_loss = 0.0
        total_tokens = 0
        for index in order:
            ids, labels, _, response = tensors[index]
            optimizer.zero_grad()
            loss = model(input_ids=ids, labels=labels).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("손실이 유한하지 않습니다.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(response)
            total_tokens += len(response)
        average = total_loss / total_tokens
        history.append({"epoch": epoch, "train_loss": average})
        if epoch == 1 or epoch % 10 == 0:
            print(f"{epoch:03d}회 | 학습 손실 {average:.4f}", flush=True)

        if epoch % 20 == 0:
            results = evaluate(model, records, tensors, tokenizer)
            matched = sum(r["text_match"] for r in results)
            exact = sum(r["token_match_including_eos"] for r in results)
            print(
                f"  답변 일치 {matched}/{len(records)}"
                f" | 종료 토큰까지 일치 {exact}/{len(records)}",
                flush=True,
            )
            for result in results[:3]:
                print(json.dumps(result, ensure_ascii=False), flush=True)

            model.save_pretrained(str(MODEL_DIR))
            tokenizer.save(str(MODEL_DIR / "tokenizer.json"))
            (MODEL_DIR / "results.json").write_text(
                json.dumps({
                    "epoch": epoch,
                    "seed": SEED,
                    "learning_rate": LEARNING_RATE,
                    "text_matches": matched,
                    "token_matches": exact,
                    "history": history,
                    "results": results,
                    "purpose": "학습 예제 재현 검사. 일반 대화 성능 평가와 구분.",
                }, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if exact == len(records):
                break

    print("완료:", MODEL_DIR, flush=True)


if __name__ == "__main__":
    main()
