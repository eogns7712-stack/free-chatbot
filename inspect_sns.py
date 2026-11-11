from pathlib import Path
from zipfile import ZipFile

import ijson

ZIP_PATH = Path(
    r"C:\Users\510\Downloads\024.한국어 SNS 데이터"
    r"\01.데이터\1.Training\1.라벨링데이터_0509_add"
    r"\(비식별화완료)[라벨]한국어SNS_train.zip"
)

MAX_CANDIDATES = 3


def merge_utterances(body):
    turns = []

    for utterance in body:
        speaker = utterance["participantID"]
        text = utterance["utterance"].strip()

        if not text:
            continue

        if turns and turns[-1]["speaker"] == speaker:
            turns[-1]["text"] += "\n" + text
        else:
            turns.append({"speaker": speaker, "text": text})

    return turns


def main():
    if not ZIP_PATH.is_file():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {ZIP_PATH}")

    printed = 0

    with ZipFile(ZIP_PATH) as archive:
        json_files = [
            entry
            for entry in archive.infolist()
            if not entry.is_dir()
            and entry.filename.lower().endswith(".json")
        ]

        if not json_files:
            print("ZIP 안에 JSON 파일이 없습니다.")
            return

        first_file = json_files[0]
        print("읽는 파일:", first_file.filename)

        with archive.open(first_file) as file:
            for conversation in ijson.items(file, "data.item"):
                info = conversation["header"]["dialogueInfo"]

                if info["numberOfParticipants"] != 2:
                    continue

                turns = merge_utterances(conversation["body"])
                speakers = {turn["speaker"] for turn in turns}

                if len(speakers) != 2:
                    continue

                if any("#" in turn["text"] for turn in turns):
                    continue

                printed += 1

                print(f"\n--- 후보 {printed} ---")
                print("대화 ID:", info["dialogueID"])
                print("주제:", info["topic"])
                print("합친 뒤 발화 수:", len(turns))

                for turn in turns:
                    print(f"\n{turn['speaker']}: {turn['text']}")

                if printed >= MAX_CANDIDATES:
                    break

    print(f"\n출력한 후보: {printed}개")


if __name__ == "__main__":
    main()