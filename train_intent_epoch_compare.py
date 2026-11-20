"""매 회차 모델과 생성 답변을 보존하는 6회 비교 학습. 프로젝트 루트에서 실행하세요."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

from train_sns_30min import load_examples, make_batch, validation_loss

ROOT = Path(__file__).resolve().parent


def save_examples(model, tokenizer, records, folder):
    model.eval()
    results = []
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            question = record["messages"][0]["content"]
            reference = record["messages"][1]["content"]
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
                max_new_tokens=min(48, model.config.n_positions - len(prompt)),
                do_sample=False,
            )
            results.append({
                "case": index,
                "question": question,
                "reference": reference,
                "answer": tokenizer.decode(
                    output[0, len(prompt):].tolist(), skip_special_tokens=True
                ).strip(),
            })
    (folder / "validation_answers.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )



def snapshot(model, tokenizer, records, questions, output, epoch, train, pad_id):
    # 진단 전후 난수 상태를 보존해 학습 순서와 드롭아웃에 영향을 주지 않습니다.
    rng = torch.get_rng_state()
    python_rng = random.getstate()
    was_training = model.training
    try:
        folder = output / "epochs" / f"epoch-{epoch:02d}"
        folder.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(str(folder))
        tokenizer.save(str(folder / "tokenizer.json"))
        model.eval()
        train_nll = validation_loss(model, train, pad_id, float("inf"))
        save_examples(model, tokenizer, records, folder)
        answers = []
        with torch.inference_mode():
            for record in questions:
                prompt = [tokenizer.token_to_id("<user>"), *tokenizer.encode(record["question"]).ids,
                          tokenizer.token_to_id("<eos>"), tokenizer.token_to_id("<assistant>")]
                budget = min(48, model.config.n_positions - len(prompt))
                if budget < 1:
                    raise ValueError(f"입력이 너무 깁니다: {record['id']}")
                inputs = torch.tensor([prompt], dtype=torch.long)
                generated = model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                           max_new_tokens=budget, do_sample=False, num_beams=1,
                                           repetition_penalty=1.0)
                answer = tokenizer.decode(generated[0, len(prompt):].tolist(), skip_special_tokens=True).strip()
                answers.append({**record, "answer": answer, "decision": "pending"})
        (folder / "test_answers.json").write_text(json.dumps(answers, ensure_ascii=False, indent=2), encoding="utf-8")
        summary = {"epoch": epoch, "train_eval_nll": train_nll, "question_count": len(answers)}
        (folder / "diagnosis.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"회차 {epoch}: 모델·검증 답변·비교 답변 저장 | 학습 문답 평가 손실 {train_nll:.4f}", flush=True)
    finally:
        torch.set_rng_state(rng)
        random.setstate(python_rng)
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/sns-fulltext-v1")
    parser.add_argument("--data", type=Path, default=ROOT / "data/intent_dialogue")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/intent-pretrained-v1")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--scratch", action="store_true", help="동일 구조의 무작위 초기화 비교 실험")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/intent_expanded/test_questions.jsonl")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("epochs는 1 이상이어야 합니다.")
    if args.output.exists():
        raise FileExistsError(f"이미 결과 폴더가 있습니다: {args.output}")

    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    tokenizer_path = args.source / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    for token in ("<user>", "<assistant>", "<eos>", "<pad>"):
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"특수 토큰이 없습니다: {token}")
    if args.scratch:
        model = GPT2LMHeadModel(GPT2Config.from_pretrained(str(args.source)))
    else:
        model = GPT2LMHeadModel.from_pretrained(str(args.source))

    if model.config.vocab_size != tokenizer.get_vocab_size():
        raise ValueError("모델과 토크나이저 어휘 크기가 다릅니다.")
    if model.config.n_positions < 128:
        raise ValueError("문맥 길이 128 이상인 모델이 필요합니다.")
    train = load_examples(args.data / "train.jsonl", tokenizer)
    validation = load_examples(args.data / "validation.jsonl", tokenizer)
    records = [
        json.loads(line)
        for line in (args.data / "validation.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    questions = [json.loads(line) for line in args.questions.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not questions or len({r["id"] for r in questions}) != len(questions):
        raise ValueError("평가 질문이 없거나 ID가 중복됩니다.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    pad_id = tokenizer.token_to_id("<pad>")
    args.output.mkdir(parents=True)

    baseline = validation_loss(model, validation, pad_id, float("inf"))
    report = {
        "initialization": "random_same_architecture" if args.scratch else "pretrained",
        "source": str(args.source),
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_count": len(train),
        "validation_count": len(validation),
        "batch_size": 1,
        "learning_rate": 0.0003,
        "seed": 42,
        "tokenizer_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "train_sha256": hashlib.sha256((args.data / "train.jsonl").read_bytes()).hexdigest(),
        "validation_sha256": hashlib.sha256((args.data / "validation.jsonl").read_bytes()).hexdigest(),
        "initial_validation_loss": baseline,
        "best_epoch": 0,
        "best_validation_loss": baseline,
        "history": [],
    }
    model.save_pretrained(str(args.output))
    tokenizer.save(str(args.output / "tokenizer.json"))
    print(f"초기화: {report['initialization']} | 학습 {len(train)} | 검증 {len(validation)}", flush=True)
    print(f"학습 전 답변 검증 손실: {baseline:.4f}", flush=True)
    print("시작 모델도 최선 모델 후보로 저장했습니다.", flush=True)

    report["questions_sha256"] = hashlib.sha256(args.questions.read_bytes()).hexdigest()
    report["note"] = "매 회차 생성 비교. 개발용 질문이며 독립 최종 시험이 아님. 자동 통과 판정 없음."
    snapshot(model, tokenizer, records, questions, args.output, 0, train, pad_id)
    (args.output / "training_log.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = list(range(len(train)))
        random.shuffle(order)
        total, tokens = 0.0, 0
        for index in order:
            ids, labels, mask, count = make_batch(train, [index], pad_id)
            optimizer.zero_grad()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("손실이 유한하지 않습니다.")
            loss.backward()
            optimizer.step()
            total += loss.item() * count
            tokens += count
        value = validation_loss(model, validation, pad_id, float("inf"))
        report["history"].append({
            "epoch": epoch, "train_loss": total / tokens, "validation_loss": value,
        })
        print(f"{epoch}/{args.epochs}회 | 학습 {total/tokens:.4f} | 검증 {value:.4f}", flush=True)
        if value < report["best_validation_loss"]:
            model.save_pretrained(str(args.output))
            report["best_epoch"] = epoch
            report["best_validation_loss"] = value
        snapshot(model, tokenizer, records, questions, args.output, epoch, train, pad_id)
        (args.output / "training_log.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    best_model = GPT2LMHeadModel.from_pretrained(str(args.output))
    save_examples(best_model, tokenizer, records, args.output)
    print("완료 | 선택 회차:", report["best_epoch"], flush=True)
    print("생성 결과:", args.output / "validation_answers.json", flush=True)


if __name__ == "__main__":
    main()
