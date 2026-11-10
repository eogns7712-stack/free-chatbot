from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

MODEL_DIR = Path(__file__).resolve().parent / "artifacts" / "tiny-chat-model-v2"


def main():
    tokenizer = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
    model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))
    model.eval()

    cases = [
        ("학습에 사용한 질문", "안녕!"),
        ("처음 보는 질문", "안녕하세요!"),
        ("학습에 사용한 계산", "1시간 20분은 몇 분이야? 계산도 보여줘."),
        ("처음 보는 계산", "1시간 30분은 몇 분이야? 계산도 보여줘."),
        ("새로운 분 값", "1시간 25분은 몇 분이야? 계산도 보여줘."),
        ("새로운 시간·분 조합", "2시간 30분은 몇 분이야? 계산도 보여줘."),
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