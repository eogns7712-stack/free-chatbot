# 후속 정보 v5: 학습 범위 비교

가설: 신규 문맥의 첫 답변도 학습하면 첫 응답 오류를 줄여 실제 후속 대화가 개선되는지 확인한다. 개선은 아직 확인하지 않았다.

비교 대상은 artifacts/followup-information-v5다. 같은 시작 모델 three-turn-flow-v4/epoch-06, 같은387개 순서·문장, CPU4스레드·배치1·시드42·학습률0.0001·3회를 유지한다. 신규48개만 loss_scope를 제거해 첫 assistant 본문과EOS까지 학습한다. 기존339개의 손실 범위는 유지한다. 검증·test·probes 파일은 바이트 단위로 동일하다. 학습 토큰 수는 늘어나므로 같은 연산량 비교는 아니다.

이번 비교는 이전 실험의 선택된2회차에서 이어가지 않고, 이전 실험과 같은 원본 모델에서 시작한다.

프로젝트 루트에서 실행:

```powershell
.\.venv\Scripts\python.exe train_three_turn_v4.py --source artifacts/three-turn-flow-v4/epochs/epoch-06 --data data/followup_information_v5_all_assistant --output artifacts/followup-information-v5-all-assistant --epochs 3 --learning-rate 0.0001
```

출력이 있으면 덮어쓰지 않는다. resume은 지원하지 않는다. CPU3회로 제한하며 시간은 실측 전이다. 기존 가중치·데이터·결과는 보존한다.

평가: 마지막 답변의 정답 문맥 손실뿐 아니라 실제 rollout의 첫 응답, 마지막 응답, 전체 흐름을 각각 확인한다. 같은 회차끼리 비교하고 최저 검증 손실 선택 결과도 따로 비교한다. 정답 문장 정확 일치는 보조 지표다. 기존 단문·주제 전환·3턴 회귀에서 새 실패가 생기면 기준 모델로 교체하지 않는다. 검증용 신규6개로 회차를 선택하고 이미 살펴본 비교6개는 개발 비교로만 해석한다.

종료 기준:3회 이후 자동 연장하지 않는다. 첫 응답·후속 응답 개선과 기존 기능 보존이 함께 확인될 때만 화면 시험 후보로 삼는다. 종료 의사·미학습 주제·임의 정보 회상은 이번 범위에 포함하지 않는다.

재현 생성기: experiments/build_followup_scope_compare_v5.py. build_report.json에 변경48개와 평가 파일 해시를 기록한다.
