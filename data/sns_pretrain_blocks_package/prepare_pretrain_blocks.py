"""AI Hub SNS의 긴 대화도 보존하는 블록 사전학습 말뭉치 생성."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import ijson
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
DEFAULT_ZIP = (
    "C:/Users/510/Downloads/024.한국어 SNS 데이터/01.데이터/"
    "1.Training/1.라벨링데이터_0509_add/"
    "(비식별화완료)[라벨]한국어SNS_train.zip"
)
SPECIAL = ("<pad>", "<unk>", "<user>", "<assistant>", "<eos>")
BLOCKED = ("씨발", "시발", "병신", "개새끼", "좆", "ㅅㅂ", "ㅂㅅ")


def dialogue_tokens(conversation, tokenizer, counts):
    if conversation["header"]["dialogueInfo"]["numberOfParticipants"] != 2:
        counts["not_two_speakers"] += 1
        return None
    turns = []
    for utterance in conversation["body"]:
        text = re.sub(r"([ㅋㅎㅠㅜ])\1{3,}", r"\1\1\1", utterance["utterance"])
        text = re.sub(r"[ \t]+", " ", text).strip()
        if not text:
            continue
        if (
            "#" in text or "\ufffd" in text
            or any(word in text for word in BLOCKED)
            or any(token in text for token in SPECIAL)
        ):
            counts["filtered_text"] += 1
            return None
        speaker = utterance["participantID"]
        if turns and turns[-1][0] == speaker:
            turns[-1][1] += "\n" + text
        else:
            turns.append([speaker, text])
    speakers = list(dict.fromkeys(s for s, _ in turns))
    if len(speakers) != 2 or sum(len(re.findall(r"[가-힣A-Za-z0-9]", t)) for _, t in turns) < 10:
        counts["insufficient_text"] += 1
        return None
    ids = []
    for speaker, text in turns:
        role = "<user>" if speaker == speakers[0] else "<assistant>"
        ids.extend([tokenizer.token_to_id(role), *tokenizer.encode(text).ids,
                    tokenizer.token_to_id("<eos>")])
    return ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=DEFAULT_ZIP)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/sns-tokenizer-v1/tokenizer.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/sns_pretrain_blocks")
    parser.add_argument("--train-dialogues", type=int, default=50000)
    parser.add_argument("--validation-dialogues", type=int, default=5000)
    args = parser.parse_args()
    if args.train_dialogues < 1 or args.validation_dialogues < 1:
        parser.error("대화 수는 양수여야 합니다.")
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    for token in SPECIAL:
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"토큰 없음: {token}")
    if tokenizer.get_vocab_size() > 65536:
        raise ValueError("uint16 저장을 위해 어휘는 65536개 이하여야 합니다.")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "report.json").write_text('{"status":"building"}', encoding="utf-8")
    stats = {"train": Counter(), "validation": Counter()}
    seen_ids, seen_text = set(), set()
    topics = []

    with (
        ZipFile(args.zip) as archive,
        (args.output / "train.bin").open("wb") as train_file,
        (args.output / "validation.bin").open("wb") as validation_file,
        (args.output / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
    ):
        entries = [e for e in archive.infolist() if e.filename.lower().endswith(".json")]
        for index, entry in enumerate(entries):
            quotas = {
                split: total // len(entries) + (index < total % len(entries))
                for split, total in [("train", args.train_dialogues),
                                     ("validation", args.validation_dialogues)]
            }
            counts = Counter()
            with archive.open(entry) as source:
                for conversation in ijson.items(source, "data.item"):
                    counts["scanned"] += 1
                    info = conversation["header"]["dialogueInfo"]
                    key = info["dialogueID"]
                    if key in seen_ids:
                        continue
                    split = (
                        "validation"
                        if int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10 == 0
                        else "train"
                    )
                    if counts[split] >= quotas[split]:
                        continue
                    ids = dialogue_tokens(conversation, tokenizer, counts)
                    if ids is None:
                        continue
                    digest = hashlib.sha256(np.asarray(ids, dtype="<u2").tobytes()).hexdigest()
                    if digest in seen_text:
                        continue
                    seen_ids.add(key)
                    seen_text.add(digest)
                    destination = train_file if split == "train" else validation_file
                    block_start = stats[split]["blocks"]
                    # 블록 경계에서 직전 토큰 1개를 겹쳐 다음 토큰 예측을 보존합니다.
                    for start in range(0, len(ids) - 1, 127):
                        part = ids[start:start + 128]
                        row = np.full(128, tokenizer.token_to_id("<pad>"), dtype="<u2")
                        row[:len(part)] = part
                        destination.write(row.tobytes())
                        stats[split]["blocks"] += 1
                        stats[split]["prediction_targets"] += len(part) - 1
                    stats[split]["dialogues"] += 1
                    stats[split]["unique_serialized_tokens"] += len(ids)
                    counts[split] += 1
                    manifest.write(json.dumps({
                        "dialogue_id": key, "split": split, "topic": info["topic"],
                        "source_json": entry.filename,
                        "tokens": len(ids), "block_start": block_start,
                        "block_count": stats[split]["blocks"] - block_start,
                        "content_hash": digest,
                    }, ensure_ascii=False) + "\n")
                    if all(counts[s] >= quotas[s] for s in quotas):
                        break
            topics.append({"source_json": entry.filename, **dict(counts)})
            print(f"주제 {index+1} | 학습 {counts['train']} | 검증 {counts['validation']}", flush=True)

    expected = {"train": args.train_dialogues, "validation": args.validation_dialogues}
    if any(stats[s]["dialogues"] != expected[s] for s in expected):
        raise ValueError("목표 수량 미달. 다른 출력 폴더로 조건을 확인해 다시 실행하세요.")
    report = {
        "status": "complete", "dtype": "<u2", "block_size": 128, "overlap": 1,
        "pad_token_id": tokenizer.token_to_id("<pad>"),
        "vocab_size": tokenizer.get_vocab_size(),
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "source_zip": Path(args.zip).name,
        "splits": {s: dict(v) for s, v in stats.items()},
        "topics": topics,
        "limitations": [
            "원본 주제별 앞부분의 조건 통과 대화이며 무작위 대표 표본이 아님.",
            "문장과 대화 흐름을 모두 쓰는 언어모델 학습용. 역할이 블록 첫머리에 없을 수도 있음.",
            "패딩을 제외한 모든 토큰을 학습하며, 서로 다른 대화를 같은 블록에 합치지 않음.",
            "자동 필터로 의미 적절성과 사실성은 보장하지 못함.",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report["splits"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
