from pathlib import Path

import json
import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "sns-pretrain-blocks-v1"
    / "models"
    / "step-00063516"
)

def main():
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))
    model.eval()

    print("모델 경로:", MODEL_DIR)
    print("모델 어휘 크기:", model.config.vocab_size)
    print("토크나이저 어휘 크기:", tokenizer.get_vocab_size())

    for token in ("<user>", "<assistant>", "<eos>", "<pad>"):
        print(token, tokenizer.token_to_id(token))

        validation_path = (
        Path(__file__).resolve().parent
        / "data"
        / "intent_dialogue"
        / "validation.jsonl"
    )

    with validation_path.open(encoding="utf-8-sig") as file:
        validation_records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    cases = [
        ("인사", "안녕!"),
        ("산책", "오늘 날씨가 좋아서 산책하고 왔어."),
        ("피곤함", "오늘 좀 피곤하네."),
        ("영화", "주말에 영화 보려고 하는데 너는 영화 좋아해?"),
        ("분갈이", "화분이 많이 자라서 분갈이를 하려고 해."),
    ]

    for label, question in cases:
        # 학습 때와 같은 형식으로 답변 시작 지점까지 입력합니다.
        prompt_ids = [
            tokenizer.token_to_id("<user>"),
            *tokenizer.encode(question).ids,
            tokenizer.token_to_id("<eos>"),
            tokenizer.token_to_id("<assistant>"),
        ]
        inputs = torch.tensor([prompt_ids], dtype=torch.long)

        with torch.inference_mode():
            output = model.generate(
                input_ids=inputs,
                attention_mask=torch.ones_like(inputs),
                max_new_tokens=48,
                do_sample=False,
            )

        reply_ids = output[0, len(prompt_ids):].tolist()
        reply = tokenizer.decode(reply_ids, skip_special_tokens=True).strip()

        print(f"\n[{label}]")
        print(f"질문: {question}")
        print(f"답변: {reply or '(빈 답변)'}")


if __name__ == "__main__":
    main()