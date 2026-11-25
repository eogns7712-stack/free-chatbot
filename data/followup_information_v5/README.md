# 후속 정보 반영 실험 v5

목표: 작품명·식물명·산책 장소를 받은 뒤 같은 정보를 다시 묻는 오류를 줄인다. 자유 대화 완성이나 종료 처리 실험은 아니다.

기존 three_turn_flow_v4 학습339개를 그대로 앞부분에 보존하고48개를 추가했다. 검증69개(기존63+신규6), 비교69개(기존63+신규6), probes155개다. 신규 예시는 마지막 assistant만 학습한다. 신규 세 주제별16개 학습,2개 검증,2개 비교다. 검증·비교는 학습과 다른 이름과 후속 문형을 사용하지만 첫 문맥은 공유한다. 의미적으로 독립적인 자유 대화 시험이 아니다.

## 가설과 고정 조건

후속 답변의 문형·이름 변형을 보강하면 이미 받은 정보를 재질문하는 일이 줄어드는지 확인한다. 구조·토크나이저·직렬화·생성 방식은 유지한다. 종료·미학습 주제·회상 예시는 추가하지 않는다. 원본 모델과 결과는 덮어쓰지 않는다.

프로젝트 루트에서 실행:

```powershell
.\.venv\Scripts\python.exe train_three_turn_v4.py --source artifacts/three-turn-flow-v4/epochs/epoch-06 --data data/followup_information_v5 --output artifacts/followup-information-v5 --epochs 3 --learning-rate 0.0001
```

CPU4스레드·배치1·3회로 제한한다. 시간은 실측 전이며30분을 보장하지 않는다. 회차별 모델과 결과를 보존한다. 이 학습기는 resume을 지원하지 않으며 기존 출력 폴더가 있으면 중단한다.

## 평가와 종료 기준

각 회차 test_answers.json에서 정답 문맥의 answer와 실제 생성 문맥의 rollout_turns를 구분한다. 첫 응답부터 잘못되면 후속 정보 반영 오류로 단정하지 않는다. 첫 응답이 적절한 경우의 후속 성공과 전체 흐름 성공을 따로 센다. 학습 예시의 마지막 답변만 손실을 주므로 첫 응답 오류 개선은 보장되지 않는다.

통과: 받은 정보를 인정하는 이해 가능한 관련 응답. 부분: 이미 제공한 정보를 재질문. 실패: 명확한 주제 이탈·깨진 말·심한 반복. 정답 문장과 정확히 일치할 필요는 없다.

검증6개로 회차를 비교하고 비교6개는 선택 후 확인한다. 같은 파일에 두 묶음이 들어 있으므로 group이 아니라 ID의 validation/test를 구분한다. 기존 단문·후속·전환 probes의 새로운 실패가 없는지도 확인한다. 개선이 없으면3회에서 종료하고 같은 조건으로 더 학습하지 않는다. 기존 모델 교체는 평가 후 결정한다.

신규12개 실제 흐름의 학습 전 결과: reports/followup-information-v5-baseline-corrected/answers.json. 자동 pending은 점수가 아니다. 기존 fresh_dialogues_v1도 재사용 시 개발 평가로 해석한다.

학습 후 선택한 회차 평가 예(회차는 검증 후 지정):

```powershell
.\.venv\Scripts\python.exe evaluation/evaluate_scenarios.py --model artifacts/followup-information-v5/epochs/epoch-03 --scenarios evaluation/followup_information_v5.json --output reports/followup-information-v5-epoch03
```

생성기 experiments/build_followup_information_v5.py는 기존 출력이 있으면 덮어쓰지 않는다. build_report.json에 원본 해시와 검사 요약이 있다.

최초 baseline 폴더는 조사 교정 전 진단이며 비교에는 baseline-corrected를 사용한다. 기존 결과는 보존했다.
