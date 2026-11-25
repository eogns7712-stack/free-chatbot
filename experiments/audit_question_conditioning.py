"""같은 답변의 예측 손실을 올바른 질문과 잘못 짝지은 질문에서 비교."""

# 프로젝트 루트에서 실행하되, 이동 후에도 데이터·모델 경로와 공통 함수 import를 유지합니다.
import sys as _sys
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

ROOT = PROJECT_ROOT
MODEL_PATHS = {
    "pretraining": "artifacts/sns-pretrain-blocks-v1/models/step-00063516",
    "sft50": "artifacts/intent-pretrained-largecorpus-v1",
    "sft600": "artifacts/intent-clean600-v1",
}


def read_pairs(path):
    pairs = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            messages = json.loads(line)["messages"]
            if [m["role"] for m in messages] != ["user", "assistant"]:
                raise ValueError("질문·답변 한 쌍인 데이터가 필요합니다.")
            pairs.append((messages[0]["content"], messages[1]["content"]))
    return pairs


def score(model, tokenizer, question, answer):
    prompt = [
        tokenizer.token_to_id("<user>"),
        *tokenizer.encode(question).ids,
        tokenizer.token_to_id("<eos>"),
        tokenizer.token_to_id("<assistant>"),
    ]
    response = [*tokenizer.encode(answer).ids, tokenizer.token_to_id("<eos>")]
    ids = prompt + response
    if len(ids) > model.config.n_positions:
        raise ValueError("모델 문맥 길이를 초과했습니다.")
    inputs = torch.tensor([ids], dtype=torch.long)
    labels = torch.tensor([[-100] * len(prompt) + response], dtype=torch.long)
    with torch.inference_mode():
        loss = model(
            input_ids=inputs, attention_mask=torch.ones_like(inputs), labels=labels
        ).loss.item()
    return loss, len(response)


def audit_model(folder, datasets):
    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    model = GPT2LMHeadModel.from_pretrained(str(folder))
    model.eval()
    summary = {}
    for name, pairs in datasets.items():
        # 고정 순환 치환으로 질문만 교체합니다.
        offset = max(1, len(pairs) // 2)
        records = []
        total_correct = total_wrong = 0.0
        total_tokens = 0
        for i, (question, answer) in enumerate(pairs):
            wrong = pairs[(i + offset) % len(pairs)][0]
            assert question != wrong
            correct_loss, count = score(model, tokenizer, question, answer)
            wrong_loss, _ = score(model, tokenizer, wrong, answer)
            records.append({
                "case": i + 1, "question": question, "wrong_question": wrong,
                "answer": answer, "correct_nll": correct_loss,
                "wrong_nll": wrong_loss, "gain": wrong_loss - correct_loss,
            })
            total_correct += correct_loss * count
            total_wrong += wrong_loss * count
            total_tokens += count
        summary[name] = {
            "cases": len(pairs),
            "correct_response_nll": total_correct / total_tokens,
            "wrong_response_nll": total_wrong / total_tokens,
            "question_gain": (total_wrong - total_correct) / total_tokens,
            "cases_preferring_correct_question": sum(r["gain"] > 0 for r in records),
            "records": records,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "question_conditioning_audit.json")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"이미 결과가 있습니다: {args.output}")
    torch.set_num_threads(4)
    datasets = {
        "shared_train50": read_pairs(args.project / "data/intent_dialogue/train.jsonl"),
        "validation15": read_pairs(args.project / "data/intent_dialogue/validation.jsonl"),
    }
    result = {
        "method": "응답·종료 토큰의 teacher-forced NLL 비교. 질문은 고정 순환 치환.",
        "interpretation": "gain 양수이면 원래 질문이 같은 정답의 예측에 더 유리함.",
        "limitations": [
            "생성 문장의 품질이나 자유주제 성능 점수가 아닙니다.",
            "질문 교체 시 길이와 절대 위치도 바뀔 수 있어 의미 이해만의 인과 효과로 해석하지 않습니다.",
            "검증 15개는 여러 차례 모델 선택에 사용한 개발용 자료입니다.",
        ],
        "models": {},
    }
    for name, path in MODEL_PATHS.items():
        result["models"][name] = audit_model(args.project / path, datasets)
        for dataset, row in result["models"][name].items():
            print(name, dataset, json.dumps({k:v for k,v in row.items() if k!="records"}), flush=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
