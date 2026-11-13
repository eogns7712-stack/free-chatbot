from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "tiny-chat-model-sns-v2"
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

    cases = [
    ("인사", "안녕!"),
    ("일상 대화", "오늘 날씨가 좋아서 산책하고 왔어."),
    ("감정 반응", "오늘 좀 피곤하네."),
    ("취향 대화", "주말에 영화 보려고 하는데 너는 영화 좋아해?"),
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