"""학습 완료 후 별도 30개 질문 평가. 프로젝트 루트에서 실행하세요."""

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "artifacts/intent-pretrained-500-v1")
    parser.add_argument("--questions", type=Path, default=ROOT / "data/intent_expanded/test_questions.jsonl")
    args = parser.parse_args()
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
    model = GPT2LMHeadModel.from_pretrained(str(args.model))
    model.eval()
    records = [
        json.loads(line)
        for line in args.questions.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    results = []
    with torch.inference_mode():
        for record in records:
            prompt = [
                tokenizer.token_to_id("<user>"),
                *tokenizer.encode(record["question"]).ids,
                tokenizer.token_to_id("<eos>"),
                tokenizer.token_to_id("<assistant>"),
            ]
            if len(prompt) >= model.config.n_positions:
                raise ValueError(f"입력이 너무 깁니다: {record['id']}")
            inputs = torch.tensor([prompt], dtype=torch.long)
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=min(48, model.config.n_positions - len(prompt)),
                do_sample=False,
            )
            answer = tokenizer.decode(
                output[0, len(prompt):].tolist(), skip_special_tokens=True
            ).strip()
            result = {**record, "answer": answer, "decision": "pending"}
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    path = args.model / "test_answers.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("저장:", path)


if __name__ == "__main__":
    main()
