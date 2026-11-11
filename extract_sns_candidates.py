import json
from pathlib import Path
from zipfile import ZipFile

import ijson

ROOT = Path(__file__).resolve().parent

ZIP_PATH = Path(
    r"C:\Users\510\Downloads\024.한국어 SNS 데이터"
    r"\01.데이터\1.Training\1.라벨링데이터_0509_add"
    r"\(비식별화완료)[라벨]한국어SNS_train.zip"
)

OUTPUT_PATH = ROOT / "data" / "sns_candidates.jsonl"
MAX_CANDIDATES = 100


def merge_utterances(body):
    turns = []

    for utterance in body:
        speaker = utterance["participantID"]
        text = utterance["utterance"].strip()

        if not text:
            continue

        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += "\n" + text
            turns[-1]["utterance_ids"].append(utterance["utteranceID"])
        else:
            turns.append(
                {
                    "speaker": speaker,
                    "text": text,
                    "utterance_ids": [utterance["utteranceID"]],
                }
            )

    return turns


def main():
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {ZIP_PATH}")

    # 재실행으로 검토 결과를 덮어쓰지 않도록 합니다.
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"이미 저장된 파일이 있습니다: {OUTPUT_PATH}\n"
            "기존 파일을 다른 이름으로 보관한 뒤 다시 실행하세요."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_ids = set()

    with ZipFile(ZIP_PATH) as archive:
        json_files = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir()
            and entry.filename.lower().endswith(".json")
        ]

        if not json_files:
            raise ValueError("ZIP 안에 JSON 파일이 없습니다.")

        first_file = json_files[0]
        print("읽는 파일:", first_file.filename)

        with archive.open(first_file) as source:
            with OUTPUT_PATH.open("x", encoding="utf-8") as output:
                for conversation in ijson.items(source, "data.item"):
                    info = conversation["header"]["dialogueInfo"]
                    dialogue_id = info["dialogueID"]

                    if info["numberOfParticipants"] != 2:
                        continue

                    if dialogue_id in seen_ids:
                        continue

                    turns = merge_utterances(conversation["body"])
                    speakers = {turn["speaker"] for turn in turns}

                    if len(speakers) != 2:
                        continue

                    if any("#" in turn["text"] for turn in turns):
                        continue

                    candidate = {
                        "source": {
                            "dataset": "AI Hub 한국어 SNS",
                            "zip_file": ZIP_PATH.name,
                            "json_file": first_file.filename,
                            "dialogue_id": dialogue_id,
                            "topic": info["topic"],
                        },
                        "review": {
                            "decision": "pending",
                            "note": "",
                        },
                        "turns": turns,
                    }

                    output.write(json.dumps(candidate, ensure_ascii=False) + "\n")
                    seen_ids.add(dialogue_id)
                    count += 1

                    if count >= MAX_CANDIDATES:
                        break

    print(f"저장한 후보: {count}개")
    print("저장 위치:", OUTPUT_PATH)


if __name__ == "__main__":
    main()