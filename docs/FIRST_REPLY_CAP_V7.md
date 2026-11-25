# v7: 반복된 첫 답변 학습 노출 제한

초기 v6는 모든 assistant 목표를 레코드마다 학습했다. 데이터 검사에서 같은 첫 문답이 다수 레코드에 반복됨을 확인했다. 편중이 퇴보 원인인지는 확정하지 않았으며 이번에 그 영향을 비교한다.

데이터435개와 순서·문장·시작 모델·시드42·CPU4스레드·배치1·학습률0.0001·3회·검증·probes는 v6와 동일하다. 동일한 첫 사용자+assistant 토큰열마다 한 레코드에서만 첫 응답을 학습한다. 같은 첫 사용자라도 assistant 답변이 다르면 별도 그룹이다. 한 번이란 전체 학습에서 한 번이 아니라 매 회차 한 레코드 노출이다. 단문을 우선 보존하고, 나머지 다중 턴 레코드에서 첫 assistant만 마스킹한다. 입력 문맥은 삭제하지 않는다.

기존 last_assistant70개 설정과 첫 문답 이후 모든 목표를 보존한다. 데이터·공통 runtime·기존 학습기·가중치는 수정하지 않는다. 별도 실험 학습기만 사용한다. 목표 토큰 수는 회차당8310→5380으로 감소한다. 손실의 토큰 정규화와 dropout 등으로 연산 경로도 달라지므로 단순히 예시 수만 같은 비교임을 명시한다.

## 실행

프로젝트 루트에서:

```powershell
.\.venv\Scripts\python.exe experiments/train_first_reply_cap_v7.py --source artifacts/three-turn-flow-v4/epochs/epoch-06 --data data/followup_switch_v6 --output artifacts/first-reply-cap-v7 --epochs 3 --learning-rate 0.0001
```

기존 v6 모델에서 이어가지 않는다. 기존 출력이 있으면 중단한다. resume은 지원하지 않는다. 3회 이후 자동 연장하지 않는다. 아직 본 학습은 실행하지 않았다.

검사만 실행하려면 같은 명령 끝에 --check-only를 추가한다. 학습·저장을 하지 않는다. --first-reply-cap 0은 기존 마스킹을 그대로 쓰는 비교 모드다. 이전 v6 결과와 비교할 수 있으므로 대조군 재학습은 기본적으로 필요하지 않다.

## 평가

기존203개 probes에서 회차별 실제 전체 흐름과 첫/후속 응답을 구분한다. 기존 단문·전환에서 새로운 실패가 있는지 확인한다. 마지막 답변 손실만으로 교체하지 않는다. 새로운 첫 표현10개는 evaluation/first_expression_v7.json에 고정했고 학습에 넣지 않는다. 기존 데이터·시나리오 사용자 문장과 NFKC·공백 제거 정확 중복0개이며 의미적 독립성은 보장하지 않는다.

v6 1회차의 새 표현 생성 원문은 reports/first-expression-v7-v6-epoch01/answers.json에 있다. pending은 판정 전 표시다. 새 모델 회차는 기존 검증·probes를 보고 선택한 뒤 같은 새 표현 평가로 비교한다. 첫 표현10개 점수로 회차를 반복 선택하지 않는다.

예: 선택 회차가1이라면 학습 후:

```powershell
.\.venv\Scripts\python.exe evaluation/evaluate_scenarios.py --model artifacts/first-reply-cap-v7/epochs/epoch-01 --scenarios evaluation/first_expression_v7.json --output reports/first-expression-v7-cap-epoch01
```

이해 가능한 관련 응답을 통과로 평가한다. 정확 일치는 보조 지표다. 새 첫 표현 반응이 개선되고 후속·전환을 유지할 때만 시험 모델 후보로 삼는다. 개선이 없으면 같은 설정의 추가 학습을 멈춘다. 자유 대화·장기 기억 달성 실험은 아니다.
