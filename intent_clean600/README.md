# 문답 600개 데이터 구성 실험

## 변경 내용

- 기존 intent_expanded/train.jsonl의 500개를 기반으로 한다.
- revision_proposals.jsonl의 원문과 일치하는 답변 18개를 수정했다.
- 인사, 피로, 산책, 영화, 식물의 서로 다른 상황 문답 20개씩, 총 100개를 추가했다.
- 전체 600개를 고정 시드 42로 섞었다.
- manifest.json에 원본 행 번호, 공개 데이터 CSV 행 번호, 수정 이유와 원래 답변, 새 문답 주제를 남겼다.
- validation.jsonl 15개와 test_questions.jsonl 30개는 기존 파일의 바이트를 그대로 유지했다.
- 600개 질문은 정규화 기준으로 중복이 없다. 새 질문 100개는 평가·검증 질문과 .85 이상의 문자열 유사도가 없는지 확인했다. 의미상 모든 중복을 배제한 것은 아니다.
- 현재 4000개 어휘 토크나이저 기준 최대 47토큰이며 기존 로더로 600개를 읽었다.

## 실험 해석

1,000개는 후속 준비 목표이며 이번 묶음은 600개까지 진행한 것이다. 숫자를 채우기 위한 반복 문장을 추가하지 않았다.

기존 미검토 401개는 그대로 남아 있다. 600개 전체가 의미 품질 검수를 마쳤다는 뜻은 아니다. 18개 수정과 100개 보강이 함께 바뀌므로 결과를 각각의 단독 효과로 해석하지 않는다. 자유로운 주제의 완성 모델보다는 질문·응답 연결 개선 여부를 보는 데이터 구성 실험이다.

현재 가장 나았던 확대 사전학습 모델을 같은 출발점으로 사용한다. 기존 500개 실험은 시작 모델도 달랐으므로 그 점수와 단독 데이터 효과 비교를 하지 않는다.

## 학습

프로젝트 루트에 압축을 풀고 기존 학습 스크립트를 사용한다.

~~~powershell
.\.venv\Scripts\python.exe train_intent_from_pretrained.py --source artifacts/sns-pretrain-blocks-v1/models/step-00063516 --data data/intent_clean600 --output artifacts/intent-clean600-v1
~~~

기본 30회 상한과 학습률 등은 유지한다. 시작 모델도 선택 후보에 포함하고 검증 손실이 가장 낮은 모델을 저장한다.

## 공통 질문 평가

~~~powershell
.\.venv\Scripts\python.exe evaluate_dialogue_30.py --model artifacts/intent-clean600-v1 --questions data/intent_clean600/test_questions.jsonl
~~~

모범 답변과 문자 단위 일치가 아니라 질문 관련성, 이해 가능한 문장, 주제 혼합 및 반복을 같은 기준으로 수동 채점한다. 기존 30개는 반복 사용 중인 개발용 비교 질문이며 독립적인 최종 블라인드 시험이 아니다.

## 출처

기존 450개는 songys/Chatbot_data에서 왔다.
https://github.com/songys/Chatbot_data

Youngsook Song.(2018). Chatbot_data_for_Korean v1.0.
원본 MIT 라이선스를 Chatbot_data_LICENSE.txt로 유지했다. 기존 50개, 추가 100개 및 답변 수정안은 이 프로젝트를 위해 작성했다.
