# 프로젝트 폴더 안내

정리일: 2026-11-25. 목적은 현재 앱과 과거 실험을 구분하는 것이다. 가중치 삭제를 통한 용량 회수 작업이 아니다.

## 매일 사용하는 파일

| 위치 | 역할 |
| --- | --- |
| web_app_three_turn.py | 현재 로컬 채팅 화면 |
| three_turn_runtime.py | 최근 두 문답을 역할 토큰으로 직렬화하고 모델 추론 |
| train_three_turn_v4.py | 현재3턴 모델 학습·회차별 평가 |
| train_sns_30min.py | 현재 학습기와 이전 학습기의 배치·검증 공통 함수. 삭제하지 않기 |
| README.md / PLAN.md / PROGRESS.md | 실행 안내·계획 변경·실험 이력 |

프로젝트 루트에서 실행한다.

```powershell
.\.venv\Scripts\python.exe web_app_three_turn.py --model artifacts/three-turn-flow-v4/epochs/epoch-06
```

## 과거 실험 실행

이전 Python 파일은 experiments/에 모았다. 이동 뒤에도 ROOT 계산과 공통 함수 import가 프로젝트 루트를 사용하도록 수정했다.

```powershell
.\.venv\Scripts\python.exe experiments/train_capacity_target.py --help
.\.venv\Scripts\python.exe experiments/evaluate_dialogue_30.py --help
.\.venv\Scripts\python.exe experiments/web_app_two_turn.py --model artifacts/two-turn-switch-v2/epochs/epoch-03
```

오래된 앱·학습 코드는 당시 용도와 기본 모델 경로를 보존한다. 모든 과거 실험을 재학습해 재현한 것은 아니다. 기존 docs/archive 문서 속 옛 명령은 이동표를 적용해 읽는다. 프로젝트 루트 밖의 터미널 작업 위치에서는 상대 --data/--model/--output 경로를 직접 확인한다.

## 데이터와 모델

data/와 artifacts/는 학습 로그·실행 명령에서 참조하므로 위치를 유지했다. 폴더 이름은 품질 순위가 아니다.

- 현재3턴 시험 모델: artifacts/three-turn-flow-v4/epochs/epoch-06
- 기존2턴 기준: artifacts/two-turn-switch-v2/epochs/epoch-03
- 단문 기준: artifacts/short-dialogue-mixed-v3/epochs/epoch-06
- 현재3턴 데이터: data/three_turn_flow_v4
- 사전학습 비교 데이터: data/sns_pretrain_blocks, data/sns_pretrain_blocks_100k
- .venv/와 .git/는 실행 환경과 Git 이력이므로 그대로 둔다.
- __pycache__/는 Python 실행 시 자동 생성되는 캐시다.

## 자료 보관 위치

- docs/archive/: 과거 평가 문서·계획 PDF·실험 안내 원본
- reports/: capacity_benchmark.json 등 루트 JSON 분석 결과
- archive/imported_packages/intent_clean600/: data/intent_clean600과 동일했던 루트 압축해제 사본
- data/sns_pretrain_blocks_package/: 기존 경로를 보존한 과거 패키지 폴더. 자동 삭제하지 않음

## 이동표

| 이전 위치 | 현재 위치 |
| --- | --- |
| `app.py` | `experiments/app.py` |
| `audit_dialogue_500.py` | `experiments/audit_dialogue_500.py` |
| `audit_question_conditioning.py` | `experiments/audit_question_conditioning.py` |
| `benchmark_cpu_training.py` | `experiments/benchmark_cpu_training.py` |
| `benchmark_model_capacity.py` | `experiments/benchmark_model_capacity.py` |
| `build_short_dialogue_control.py` | `experiments/build_short_dialogue_control.py` |
| `check_question_usage.py` | `experiments/check_question_usage.py` |
| `evaluate_dialogue_30.py` | `experiments/evaluate_dialogue_30.py` |
| `extract_sns_candidates.py` | `experiments/extract_sns_candidates.py` |
| `inspect_sns.py` | `experiments/inspect_sns.py` |
| `prepare_pretrain_blocks.py` | `experiments/prepare_pretrain_blocks.py` |
| `prepare_sns_30min.py` | `experiments/prepare_sns_30min.py` |
| `prepare_sns_dataset.py` | `experiments/prepare_sns_dataset.py` |
| `review_sns.py` | `experiments/review_sns.py` |
| `test_memorization_100.py` | `experiments/test_memorization_100.py` |
| `test_model.py` | `experiments/test_model.py` |
| `train_capacity_target.py` | `experiments/train_capacity_target.py` |
| `train_capacity_trial.py` | `experiments/train_capacity_trial.py` |
| `train_intent_epoch_compare.py` | `experiments/train_intent_epoch_compare.py` |
| `train_intent_from_pretrained.py` | `experiments/train_intent_from_pretrained.py` |
| `train_intent_lr_compare.py` | `experiments/train_intent_lr_compare.py` |
| `train_model.py` | `experiments/train_model.py` |
| `train_pretrain_blocks.py` | `experiments/train_pretrain_blocks.py` |
| `train_sns_fulltext.py` | `experiments/train_sns_fulltext.py` |
| `train_sns_model.py` | `experiments/train_sns_model.py` |
| `train_sns_resume.py` | `experiments/train_sns_resume.py` |
| `train_sns_tokenizer.py` | `experiments/train_sns_tokenizer.py` |
| `train_tokenizer.py` | `experiments/train_tokenizer.py` |
| `train_two_turn_v1.py` | `experiments/train_two_turn_v1.py` |
| `train_two_turn_varied_v3.py` | `experiments/train_two_turn_varied_v3.py` |
| `two_turn_runtime.py` | `experiments/two_turn_runtime.py` |
| `web_app.py` | `experiments/web_app.py` |
| `web_app_two_turn.py` | `experiments/web_app_two_turn.py` |
| `web_app_v3.py` | `experiments/web_app_v3.py` |
| `BASELINE_EVALUATION.md` | `docs/archive/BASELINE_EVALUATION.md` |
| `NEXT_CAPACITY_EXPERIMENT.md` | `docs/archive/NEXT_CAPACITY_EXPERIMENT.md` |
| `PRETRAINING_README.md` | `docs/archive/PRETRAINING_README.md` |
| `SNS_30MIN_README.md` | `docs/archive/SNS_30MIN_README.md` |
| `free-chatbot-plan.pdf` | `docs/archive/free-chatbot-plan.pdf` |
| `capacity_benchmark.json` | `reports/capacity_benchmark.json` |
| `existing_token_audit.json` | `reports/existing_token_audit.json` |
| `question_conditioning_audit.json` | `reports/question_conditioning_audit.json` |
| `question_usage_results.json` | `reports/question_usage_results.json` |
| `intent_clean600` | `archive/imported_packages/intent_clean600` |
