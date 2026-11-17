# 문답500개 확장 실험

## 구성과 출처

- train.jsonl은 기존 50개와 songys/Chatbot_data의 일상(label=0) 문답 450개를 합친 500개입니다.
- validation.jsonl은 기존 15개를 바이트 그대로 유지했습니다.
- test_questions.jsonl은 이 실험용 새 질문 30개와 수동 채점 기준입니다. 학습·모델 선택에 사용하지 않습니다.
- 기존 다섯 주제의 검증 질문 15개와 더 넓은 일상 주제까지 담은 새 30개는 범위가 달라 점수를 구분해서 기록합니다.
- 외부 450개는 주제별 90개처럼 균등 배분한 데이터가 아닙니다.
- 공개 데이터 출처는 https://github.com/songys/Chatbot_data 입니다.
- Youngsook Song.(2018). Chatbot_data_for_Korean v1.0.
- 원본 저장소의 MIT 라이선스를 Chatbot_data_LICENSE.txt에 동봉했습니다. 공개 데이터의 사본과 함께 유지하세요.
- manifest.json에는 학습 파일 행별 출처와 원본 CSV 행 번호가 있습니다. report.json에는 원본 해시와 선별 조건을 기록했습니다.
- 추가 450개는 길이·중복·질문 유사도 규칙에 따라 선별했습니다. 전체 의미 적절성을 사람이 검수한 데이터는 아닙니다. 원문 답변은 수정하지 않았습니다.
- 검증·평가 질문과 동일하거나 높은 문자열 유사도(.85 이상)를 보이는 외부 질문을 제외했습니다. 의미상 모든 중복을 배제했다고 보장하지 않습니다.
- 새 질문은 추가 문답 학습에서 제외합니다. 과거 사전학습 원문 전체와의 의미 중복까지 보장하는 독립 테스트셋은 아닙니다.
- 학습량뿐 아니라 출처·주제·말투 구성도 달라졌으므로 이번 결과를 데이터 수만의 효과로 해석하지 않습니다.

## 학습

기존 train_intent_from_pretrained.py와 train_sns_30min.py가 프로젝트 루트에 있어야 합니다.

~~~powershell
.\.venv\Scripts\python.exe train_intent_from_pretrained.py --data data/intent_expanded --output artifacts/intent-pretrained-500-v1
~~~

기본 시작 모델은 artifacts/sns-fulltext-v1, 학습 횟수는 30회입니다. 기존 50개 실험의 결과와 별도 경로에 저장합니다. 검증 손실로 선택한 모델의 검증 15개 답변은 validation_answers.json에 저장됩니다.

## 별도 평가

모델 선택과 학습이 완료된 다음 실행합니다.

~~~powershell
.\.venv\Scripts\python.exe evaluate_dialogue_30.py
~~~

결과는 artifacts/intent-pretrained-500-v1/test_answers.json에 저장됩니다. 질문과 관련성, 이해 가능한 문장, 반복이나 주제 혼합을 수동 채점합니다. 모델 답변이 참고 문장과 문자 단위로 일치해야 하는 평가는 아닙니다. 새 평가 결과를 보고 다시 설정을 조정하면 이후 이 30개는 개발용 평가로 관리합니다.

## 검사 결과

현재 4000개 어휘 토크나이저와 기존 데이터 로더로 학습 500개와 검증 15개가 로딩되는지 확인했습니다. 학습·검증 입력은 128토큰 이하이며, 새 평가 30개도 생성 길이 여유를 확보했습니다. 이번에는 데이터 준비와 실행 검증까지만 진행했습니다.
