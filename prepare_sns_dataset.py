"""AI Hub SNS ZIP에서 현재 256토큰 모델용 대화를 자동 선별합니다."""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import ijson
from tokenizers import Tokenizer

DEFAULT_ZIP = (
    "C:/Users/510/Downloads/024.한국어 SNS 데이터/01.데이터/"
    "1.Training/1.라벨링데이터_0509_add/"
    "(비식별화완료)[라벨]한국어SNS_train.zip"
)
BLOCKED_WORDS = ("씨발", "시발", "병신", "개새끼", "좆", "ㅅㅂ", "ㅂㅅ")
SPECIAL_TOKENS = ("<user>", "<assistant>", "<eos>", "<pad>", "<unk>")


def clean_text(text):
    text = re.sub(r"([ㅋㅎㅠㅜ])\1{3,}", r"\1\1\1", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def build_sample(conversation, tokenizer, limit, counts):
    if conversation["header"]["dialogueInfo"]["numberOfParticipants"] != 2:
        counts["not_two_participants"] += 1
        return None

    turns = []
    for utterance in conversation["body"]:
        text = clean_text(utterance["utterance"])
        if not text:
            continue
        if (
            "#" in text
            or "\ufffd" in text
            or any(token in text for token in SPECIAL_TOKENS)
            or any(word in text for word in BLOCKED_WORDS)
        ):
            counts["mask_or_blocked_text"] += 1
            return None
        speaker = utterance["participantID"]
        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += "\n" + text
        else:
            turns.append({"speaker": speaker, "text": text})

    speakers = list(dict.fromkeys(t["speaker"] for t in turns))
    if len(speakers) != 2:
        counts["invalid_speakers"] += 1
        return None

    # 대화 시작부터 온전한 질문·응답 쌍만 추가합니다.
    messages = []
    token_count = 0
    for index in range(0, len(turns) - 1, 2):
        pair = turns[index:index + 2]
        if any(len(re.findall(r"[가-힣A-Za-z0-9]", t["text"])) < 2 for t in pair):
            break
        pair_tokens = sum(2 + len(tokenizer.encode(t["text"]).ids) for t in pair)
        if token_count + pair_tokens > limit:
            break
        messages.extend([
            {"role": "user", "content": pair[0]["text"]},
            {"role": "assistant", "content": pair[1]["text"]},
        ])
        token_count += pair_tokens

    if not messages:
        counts["short_or_over_limit_prefix"] += 1
        return None
    return {"messages": messages}, token_count, len(turns)


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default=DEFAULT_ZIP)
    parser.add_argument("--tokenizer", type=Path, default=root / "artifacts/tokenizer.json")
    parser.add_argument("--output", type=Path, default=root / "data/sns_auto")
    parser.add_argument("--per-topic", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    if args.per_topic < 1 or args.max_tokens < 4:
        parser.error("per-topic은 1 이상, max-tokens는 4 이상이어야 합니다.")

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    for token in SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"토크나이저에 특수 토큰이 없습니다: {token}")
    # 기존 전처리 결과를 덮어쓰지 않습니다.
    args.output.mkdir(parents=True, exist_ok=False)
    seen_ids = set()
    seen_content = set()
    topic_stats = []
    split_counts = Counter()
    lengths = []

    with (
        ZipFile(args.zip) as archive,
        (args.output / "train.jsonl").open("w", encoding="utf-8") as train,
        (args.output / "validation.jsonl").open("w", encoding="utf-8") as validation,
        (args.output / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
    ):
        files = [e for e in archive.infolist() if e.filename.lower().endswith(".json")]
        for entry in files:
            stats = Counter()
            with archive.open(entry) as source:
                for conversation in ijson.items(source, "data.item"):
                    stats["scanned"] += 1
                    info = conversation["header"]["dialogueInfo"]
                    dialogue_id = info["dialogueID"]
                    if dialogue_id in seen_ids:
                        stats["duplicate_id"] += 1
                        continue
                    result = build_sample(conversation, tokenizer, args.max_tokens, stats)
                    if result is None:
                        continue
                    sample, length, original_turns = result
                    serialized = json.dumps(sample, ensure_ascii=False)
                    content_hash = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
                    if content_hash in seen_content:
                        stats["duplicate_content"] += 1
                        continue
                    # 같은 ID는 항상 같은 분할에 배정됩니다.
                    id_hash = hashlib.sha256(dialogue_id.encode("utf-8")).hexdigest()
                    split = "validation" if int(id_hash, 16) % 10 == 0 else "train"
                    target = validation if split == "validation" else train
                    target.write(serialized + "\n")
                    split_counts[split] += 1
                    record = {
                        "split": split,
                        "line": split_counts[split],
                        "dialogue_id": dialogue_id,
                        "source_json": entry.filename,
                        "topic": info["topic"],
                        "tokens": length,
                        "selected_turns": len(sample["messages"]),
                        "original_merged_turns": original_turns,
                        "content_hash": content_hash,
                    }
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    seen_ids.add(dialogue_id)
                    seen_content.add(content_hash)
                    stats["selected"] += 1
                    stats[split] += 1
                    lengths.append(length)
                    if stats["selected"] >= args.per_topic:
                        break
            topic_stats.append({"source_json": entry.filename, **dict(stats)})
            print(f"주제 {len(topic_stats)}: {stats['selected']}개 선정 / {stats['scanned']}개 검사", flush=True)

    report = {
        "source_zip": Path(args.zip).name,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "vocab_size": tokenizer.get_vocab_size(),
        "max_tokens": args.max_tokens,
        "per_topic_target": args.per_topic,
        "split_method": "SHA256(dialogue_id) modulo 10; validation bucket 0",
        "counts": dict(split_counts),
        "total": len(lengths),
        "token_min": min(lengths) if lengths else None,
        "token_max": max(lengths) if lengths else None,
        "topics": topic_stats,
        "limitations": [
            "각 주제 파일 앞부분의 조건 통과 대화를 사용하므로 무작위 대표 표본이 아닙니다.",
            "자동 필터는 문맥·사실성·비속어를 완전히 판별하지 못합니다.",
            "첫 화자는 user, 상대는 assistant로 매핑하며 사람 간 대화를 모사합니다.",
            "기존 소량 토크나이저와 256토큰 한도로 긴 대화는 제외되거나 앞 구간만 사용됩니다.",
        ],
    }
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not split_counts["train"] or not split_counts["validation"]:
        raise ValueError("학습 또는 검증 분할이 비었습니다. report.json을 확인하세요.")
    print(json.dumps(report["counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
