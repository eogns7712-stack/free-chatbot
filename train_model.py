import json
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.jsonl"
TOKENIZER_PATH = ROOT / "artifacts" / "tokenizer.json"
MODEL_DIR = ROOT / "artifacts" / "tiny-chat-model-sns-v2"
LEARNING_RATE = 0.0003

EPOCHS = 60
CONTEXT_LENGTH = 256


def load_examples(tokenizer):
    examples = []

    with DATA_PATH.open(encoding="utf-8-sig") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            input_ids = []
            labels = []

            for message in json.loads(line)["messages"]:
                role = message["role"]
                if role not in ("user", "assistant"):
                    raise ValueError(f"{line_number}행: 지원하지 않는 역할 {role!r}")

                role_id = tokenizer.token_to_id(f"<{role}>")
                content_ids = tokenizer.encode(message["content"]).ids
                turn_ids = [role_id, *content_ids, tokenizer.token_to_id("<eos>")]

                input_ids.extend(turn_ids)
                if role == "assistant":
                    labels.extend([-100, *turn_ids[1:]])
                else:
                    labels.extend([-100] * len(turn_ids))

            if len(input_ids) > CONTEXT_LENGTH:
                raise ValueError(f"{line_number}행: 대화가 {CONTEXT_LENGTH}토큰을 넘습니다.")

            examples.append(
                (
                    torch.tensor([input_ids], dtype=torch.long),
                    torch.tensor([labels], dtype=torch.long),
                )
            )

    return examples


def main():
    torch.manual_seed(42)
    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    examples = load_examples(tokenizer)
    if not examples:
        raise ValueError("학습할 대화가 없습니다.")

    eos_id = tokenizer.token_to_id("<eos>")
    pad_id = tokenizer.token_to_id("<pad>")

    config = GPT2Config(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=CONTEXT_LENGTH,
        n_embd=64,
        n_layer=2,
        n_head=4,
        bos_token_id=None,
        eos_token_id=eos_id,
        pad_token_id=pad_id,
    )
    model = GPT2LMHeadModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0

        for input_ids, labels in examples:
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, labels=labels).loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print(f"학습 {epoch:02d}/{EPOCHS}회 | 평균 손실 {total_loss / len(examples):.4f}")

    model.save_pretrained(str(MODEL_DIR))
    tokenizer.save(str(MODEL_DIR / "tokenizer.json"))
    print(f"모델과 토크나이저 저장: {MODEL_DIR}")

    saved_model = GPT2LMHeadModel.from_pretrained(str(MODEL_DIR))
    saved_model.eval()

    question = "안녕!"
    prompt_ids = [
        tokenizer.token_to_id("<user>"),
        *tokenizer.encode(question).ids,
        eos_id,
        tokenizer.token_to_id("<assistant>"),
    ]
    prompt = torch.tensor([prompt_ids], dtype=torch.long)

    with torch.inference_mode():
        output = saved_model.generate(
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            max_new_tokens=48,
            do_sample=False,
        )

    reply_ids = output[0, len(prompt_ids):].tolist()
    print(f"질문: {question}")
    print(f"답변: {tokenizer.decode(reply_ids, skip_special_tokens=True)}")


if __name__ == "__main__":
    main()