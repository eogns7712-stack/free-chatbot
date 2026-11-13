import json
from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.jsonl"
OUTPUT_PATH = ROOT / "artifacts" / "tokenizer.json"

SPECIAL_TOKENS = ["<pad>", "<unk>", "<user>", "<assistant>", "<eos>"]


def read_messages():
    with DATA_PATH.open(encoding="utf-8-sig") as file:
        for line in file:
            if line.strip():
                conversation = json.loads(line)
                for message in conversation["messages"]:
                    yield message["content"]


def main():
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=386,
        min_frequency=1,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIAL_TOKENS,
    )
    tokenizer.train_from_iterator(read_messages(), trainer=trainer)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(OUTPUT_PATH))

    saved_tokenizer = Tokenizer.from_file(str(OUTPUT_PATH))
    sample = "오늘은 날씨가 맑아요."
    encoded = saved_tokenizer.encode(sample)
    decoded = saved_tokenizer.decode(encoded.ids)

    print("저장 위치:", OUTPUT_PATH)
    print("실제 어휘 크기:", saved_tokenizer.get_vocab_size())
    print("원문:", sample)
    print("토큰 ID:", encoded.ids)
    print("복원:", decoded)

    if decoded != sample:
        raise ValueError("복원한 문장이 원문과 다릅니다.")
    print("복원 테스트 통과")


if __name__ == "__main__":
    main()