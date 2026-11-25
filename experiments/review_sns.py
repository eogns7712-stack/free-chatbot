
# 프로젝트 루트에서 실행하되, 이동 후에도 데이터·모델 경로와 공통 함수 import를 유지합니다.
import sys as _sys
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(PROJECT_ROOT))

import json
from pathlib import Path

ROOT = PROJECT_ROOT
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