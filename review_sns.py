import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "sns_candidates_reviewed.jsonl"

CANDIDATE_NUMBERS = [4, 13, 14]


def main():
    with DATA_PATH.open(encoding="utf-8-sig") as file:
        candidates = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    for number in CANDIDATE_NUMBERS:
        candidate = candidates[number - 1]

        print(f"\n{'=' * 50}")
        print(f"후보 {number}")
        print("대화 ID:", candidate["source"]["dialogue_id"])

        for turn_number, turn in enumerate(candidate["turns"], start=1):
            print(f"\n발화 {turn_number} / {turn['speaker']}")
            print(turn["text"])


if __name__ == "__main__":
    main()