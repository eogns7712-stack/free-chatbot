import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
MODEL_DIR = (
    ROOT / "artifacts" / "sns-continuous-v1"
    / "models" / "step-00007500"
)
DATA_PATH = ROOT / "data" / "sns_30min" / "train.jsonl"


def generate(model, tokenizer, question):
    assistant_id = tokenizer.token_to_id("<assistant>")

    if question is None:
        prompt_ids = [assistant_id]
    else:
        prompt_ids = [
            tokenizer.token_to_id("<user>"),
            *tokenizer.encode(question).ids,
            tokenizer.token_to_id("<eos>"),
            assistant_id,
        ]

    inputs = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.inference_mode():
        output = model.generate(
            input_ids=inputs,
            attention_mask=torch.ones_like(inputs),
            max_new_tokens=48,
            do_sample=False,
        )

    return tokenizer.decode(
        output[0, len(prompt_ids):].tolist(),
        skip_special_tokens=True,
    ).strip()


def main():
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))
    model.eval()

    records = []
    with DATA_PATH.open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
            if len(records) == 3:
                break

    results = []

    for index, record in enumerate(records, start=1):
        question = record["messages"][0]["content"]
        reference = record["messages"][1]["content"]
        answer = generate(model, tokenizer, question)

        print(f"\n[학습 질문 {index}]")
        print("질문:", question)
        print("학습 답변:", reference)
        print("생성 답변:", answer or "(빈 답변)")

        results.append({
            "type": "training_question",
            "question": question,
            "reference": reference,
            "generated": answer,
        })

    for label, question in [
        ("다른 질문", "오늘은 어떤 영화를 볼까?"),
        ("질문 없이 답변 시작", None),
    ]:
        answer = generate(model, tokenizer, question)

        print(f"\n[{label}]")
        print("답변:", answer or "(빈 답변)")

        results.append({
            "type": label,
            "question": question,
            "generated": answer,
        })

    output_path = ROOT / "question_usage_results.json"
    output_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("\n결과 저장:", output_path)


if __name__ == "__main__":
    main()