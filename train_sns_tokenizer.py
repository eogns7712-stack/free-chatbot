import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sns_auto" / "train.jsonl"
OUTPUT_PATH = ROOT / "artifacts" / "sns-tokenizer-v1" / "tokenizer.json"


def read_texts():
    with DATA_PATH.open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                for message in json.loads(line)["messages"]:
                    yield message["content"]


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"이미 토크나이저가 있습니다: {OUTPUT_PATH}")

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=4000,
        min_frequency=2,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["<pad>", "<unk>", "<user>", "<assistant>", "<eos>"],
    )
    tokenizer.train_from_iterator(read_texts(), trainer=trainer)

    sample = "오늘 날씨가 좋아서 산책하고 왔어."
    encoded = tokenizer.encode(sample)
    restored = tokenizer.decode(encoded.ids)

    if restored != sample:
        raise ValueError("문장 복원에 실패했습니다.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_PATH))

    print("저장 위치:", OUTPUT_PATH)
    print("실제 어휘 크기:", tokenizer.get_vocab_size())
    print("테스트 문장 토큰 수:", len(encoded.ids))
    print("복원:", restored)


if __name__ == "__main__":
    main()