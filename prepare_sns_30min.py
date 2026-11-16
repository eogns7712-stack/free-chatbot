"""AI Hub SNS ZIP에서 현재 128토큰 모델용 대화를 자동 선별합니다."""
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
    parser.add_argument("--tokenizer", type=Path, default=root / "artifacts/sns-tokenizer-v1/tokenizer.json")
    parser.add_argument("--output", type=Path, default=root / "data/sns_30min")
    args = parser.parse_args()
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    for token in SPECIAL_TOKENS:
        if tokenizer.token_to_id(token) is None:
            raise ValueError(f"필요한 토큰이 없습니다: {token}")
    args.output.mkdir(parents=True, exist_ok=False)
    seen_ids, seen_hashes = set(), set()
    totals = Counter()
    topic_stats = []
    with (
        ZipFile(args.zip) as archive,
        (args.output / "train.jsonl").open("w", encoding="utf-8") as train,
        (args.output / "validation.jsonl").open("w", encoding="utf-8") as validation,
        (args.output / "manifest.jsonl").open("w", encoding="utf-8") as manifest,
    ):
        entries = [e for e in archive.infolist() if e.filename.lower().endswith(".json")]
        for index, entry in enumerate(entries):
            train_goal = 5000 // len(entries) + (index < 5000 % len(entries))
            validation_goal = 450 // len(entries) + (index < 450 % len(entries))
            counts = Counter()
            with archive.open(entry) as file:
                for conversation in ijson.items(file, "data.item"):
                    counts["scanned"] += 1
                    info = conversation["header"]["dialogueInfo"]
                    key = info["dialogueID"]
                    if key in seen_ids:
                        continue
                    bucket = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10
                    split = "validation" if bucket == 0 else "train"
                    goal = validation_goal if split == "validation" else train_goal
                    if counts[split] >= goal:
                        continue
                    result = build_sample(conversation, tokenizer, 128, counts)
                    if result is None:
                        continue
                    sample, length, original_turns = result
                    text = json.dumps(sample, ensure_ascii=False)
                    fingerprint = hashlib.sha256(text.encode("utf-8")).hexdigest()
                    if fingerprint in seen_hashes:
                        continue
                    seen_ids.add(key)
                    seen_hashes.add(fingerprint)
                    target = validation if split == "validation" else train
                    target.write(text + "\n")
                    counts[split] += 1
                    totals[split] += 1
                    manifest.write(json.dumps({
                        "split": split,
                        "line": totals[split],
                        "dialogue_id": key,
                        "source_json": entry.filename,
                        "topic": info["topic"],
                        "tokens": length,
                        "content_hash": fingerprint,
                        "original_merged_turns": original_turns,
                    }, ensure_ascii=False) + "\n")
                    if counts["train"] == train_goal and counts["validation"] == validation_goal:
                        break
            topic_stats.append({"file": entry.filename, **dict(counts)})
            print(f"주제 {index + 1}: 학습 {counts['train']} / 검증 {counts['validation']}", flush=True)
    report = {
        "counts": dict(totals),
        "max_tokens": 128,
        "tokenizer_sha256": hashlib.sha256(args.tokenizer.read_bytes()).hexdigest(),
        "topics": topic_stats,
        "selection": "주제별 앞부분에서 조건 통과 대화 추출. 무작위 대표 표본이 아님.",
        "split": "SHA256 원본 대화 ID modulo 10; 검증 bucket 0",
    }
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if totals["train"] != 5000 or totals["validation"] != 450:
        raise ValueError("목표 수량에 미달했습니다. report.json을 확인하세요.")
    print("완료:", dict(totals), flush=True)


if __name__ == "__main__":
    main()
