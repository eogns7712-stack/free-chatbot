# 자유주제 한국어 챗봇과 직접 학습 실험

Python과 Hugging Face 도구로 한국어 대화 모델을 직접 학습하고, 자유로운 주제로 대화하는 챗봇을 만드는 프로젝트입니다.

**현재는 모델 학습·평가 단계입니다.** 학습·저장·재개와 비교 실험은 실행되지만, 직접 학습 모델이 처음 보는 질문에 안정적으로 답하는 수준은 아직 확보하지 못했습니다. 기존 터미널·Gradio 앱은 현재 Qwen 모델을 사용합니다.

문서 갱신일은 **2026년 11월 18일**입니다.

## 처음 계획과 달라진 점

초기 계획은 공개 모델 Qwen2.5-0.5B-Instruct를 Python에 연결해 로컬 챗봇을 만들고, 2~4주 동안 답변 품질과 실행 문서를 정리하는 것이었습니다. 이후 사용자가 모델도 직접 학습하기를 원해 토크나이저·무작위 초기화 GPT-2·데이터 전처리·사전학습·문답 학습까지 범위가 확장됐습니다.

학습 문장 재현과 손실 감소가 실제 대화 개선으로 이어지지 않아, 계획을 데이터와 모델 규모의 통제 비교 및 실제 생성 평가 중심으로 변경했습니다. 기존 11/09~12/06 기간은 실험·성과 정리의 초기 일정이며 자유 대화 품질을 보장하는 마감일이 아닙니다.

상세 변경 이유와 예정 일정은 [PLAN.md](PLAN.md), 완료한 실험과 결과는 [PROGRESS.md](PROGRESS.md)에서 확인할 수 있습니다.

## 현재 기능과 실험 범위

| 구분 | 상태 |
| --- | --- |
| Qwen 터미널·브라우저 채팅 | 실행 가능 |
| 직접 토크나이저·GPT-2 학습 및 재로딩 | 구현 |
| CPU 시간 예산·체크포인트 재개 | 구현 및 소량 재현 검사 |
| SNS 50,000대화의 블록 사전학습 | 데이터 준비 및 모델 규모 비교 완료 |
| 새 질문의 자연스러운 응답 | 미달성. 반복·주제 혼합·문장 오류가 남음 |
| 직접 학습 모델의 다중 턴 품질 | 정식 검수 미완료 |
| 직접 모델의 Gradio 연결 | 미완료 |

## 환경과 설치

확인한 환경은 Windows, Python 3.14.5, Ryzen 7 1700, RAM 약 8GB입니다. GTX 1060 3GB가 있지만 설치된 PyTorch는 CPU 빌드이며 현재 학습은 CPU에서 진행합니다.

| 패키지 | 확인한 버전 | 용도 |
| --- | --- | --- |
| torch | 2.15.0 | 텐서, 손실, AdamW, 학습과 추론 |
| transformers | 5.22.2 | GPT-2 구조, 체크포인트와 생성 |
| tokenizers | 0.23.3 | ByteLevel BPE 학습과 토큰화 |
| gradio | 6.31.0 | 기존 Qwen 웹 채팅 |
| ijson | 3.6.0 | ZIP 내부 대형 JSON 순차 처리 |
| numpy | 2.5.4 | uint16 토큰 블록과 메모리 매핑 |

프로젝트 폴더에서 가상환경을 만들고 설치합니다.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install "tokenizers==0.23.3" "ijson==3.6.0" "numpy==2.5.4"
~~~

현재 requirements.txt는 torch, transformers, gradio만 고정합니다. 추가 명령은 직접 학습 스크립트의 의존성을 보완합니다. 위 버전은 기존 환경의 기록이며 새 환경 전체 재현 검수는 남아 있습니다. VS Code의 Python 인터프리터도 .venv를 선택합니다.

## 기존 Qwen 앱 실행

~~~powershell
.\.venv\Scripts\python.exe app.py
~~~

터미널에서 /reset으로 기록을 지우고 /exit로 종료합니다.

~~~powershell
.\.venv\Scripts\python.exe web_app.py
~~~

출력된 로컬 주소를 브라우저에서 엽니다. 웹 앱의 휴지통으로 대화를 지우며 서버는 Ctrl+C로 종료합니다. 모델 첫 다운로드에는 인터넷 연결이 필요합니다. 이 명령은 직접 학습 모델이 아닌 app.py에 설정된 Qwen을 사용합니다.

## 직접 학습 코드와 데이터

| 파일 | 역할 |
| --- | --- |
| train_tokenizer.py, train_model.py | 최초 소량 토크나이저·모델 실험 |
| train_sns_tokenizer.py | SNS 학습 텍스트로 어휘 4,000개 토크나이저 학습 |
| prepare_sns_dataset.py, prepare_sns_30min.py | 짧은 대화 JSONL 선별·분할 |
| prepare_pretrain_blocks.py | 긴 대화를 보존해 128토큰 블록 생성 |
| train_sns_30min.py | 기존 배치 학습 및 공통 로더·검증 함수 |
| train_sns_fulltext.py | 전체 발화 다음 토큰 학습 비교 |
| train_sns_resume.py | 짧은 JSONL 실험의 체크포인트 재개 |
| train_pretrain_blocks.py | 메모리 매핑 블록 말뭉치의 누적 사전학습 |
| train_capacity_trial.py | 같은 데이터·스텝의 128·256차원 규모 비교 |
| train_intent_from_pretrained.py | 기존 가중치에 모범 문답 답변 부분 추가 학습 |
| evaluate_dialogue_30.py | 공통 질문의 결정적 생성과 결과 저장 |
| audit_dialogue_500.py, audit_question_conditioning.py | 문답 구조·표현 및 질문 조건 활용 검사 |

직접 모델은 <user>, <assistant>, <eos>, <pad> 등의 토큰으로 학습과 추론 형식을 맞춥니다. 기존 Qwen용 채팅 템플릿을 그대로 적용하지 않습니다.

### 필요한 로컬 파일

~~~text
data/
  sns_pretrain_blocks/
    train.bin
    validation.bin
    report.json
    manifest.jsonl
  intent_dialogue/
    train.jsonl
    validation.jsonl
  intent_expanded/
    test_questions.jsonl
artifacts/
  sns-tokenizer-v1/
    tokenizer.json
~~~

큰 말뭉치·체크포인트가 저장소에 포함되지 않을 수 있습니다. 클론만으로 모든 실험을 실행할 수 있다고 가정하지 않습니다. AI Hub 원본 접근과 전처리 결과, 학습한 토크나이저가 필요합니다. 모델·말뭉치·토크나이저 파일은 해당 실험의 해시와 맞아야 합니다.

[AI Hub 한국어 SNS](https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=114)에서 원본을 준비합니다. 전처리 스크립트의 기본 ZIP 경로는 개발 PC 경로이므로 다른 환경에서는 --zip으로 지정합니다.

~~~powershell
.\.venv\Scripts\python.exe prepare_pretrain_blocks.py --zip "D:\datasets\한국어SNS_train.zip" --output data/sns_pretrain_blocks
~~~

출력 폴더가 이미 있으면 중단합니다. 기존 토크나이저를 다시 학습하거나 교체하면 과거 모델·말뭉치와 호환되지 않을 수 있습니다. 기존 실험을 재현할 때는 같은 tokenizer.json을 사용합니다.

### 블록 말뭉치

학습 50,000대화·5,352,204토큰과 검증 5,000대화·534,070토큰입니다. 학습 63,514블록을 배치 4로 처리하므로 말뭉치 1회는 15,879스텝입니다.

서로 다른 원본 대화를 학습·검증에 나누고 중복을 검사했습니다. 긴 대화는 128토큰 블록으로 나누며 직전 토큰 하나를 겹칩니다. 파일 형식은 little-endian uint16이고 메모리 매핑으로 읽습니다.

## 모델 규모 비교 재현 예시

아래는 별도의 새 결과 경로에서 실행하는 예시입니다. 이미 완료한 모델 폴더에 덮어쓰지 않습니다. 기존 train_sns_30min.py는 공통 함수로 사용되므로 프로젝트 루트에 있어야 합니다.

~~~powershell
.\.venv\Scripts\python.exe train_capacity_trial.py --width 128 --output artifacts/capacity128-repro --steps 3000
.\.venv\Scripts\python.exe train_capacity_trial.py --width 256 --output artifacts/capacity256-repro --steps 3000
~~~

재개할 때는 같은 차원과 결과 폴더에 --resume을 지정합니다.

~~~powershell
.\.venv\Scripts\python.exe train_capacity_trial.py --width 256 --output artifacts/capacity256-repro --resume --steps 3000
~~~

**--steps는 이번 실행에서 추가할 스텝 상한**입니다. 재개 학습의 평생 누적 스텝 상한은 없으며 시간을 기준으로 종료될 수도 있습니다. 이미 12,000스텝이면 3,879스텝을 추가해 15,879에 도달합니다.

모델·옵티마이저·현재 순서와 위치·난수 상태를 저장합니다. 100스텝마다 체크포인트를 교체하므로 Ctrl+C 이후에는 마지막 완전 저장 지점부터 재개합니다. 전체 실행 약 30분 중 학습 약 20분, 나머지는 검증·저장에 배정합니다.

처음 128·256차원 비교는 양쪽 모두 무작위 초기화합니다. 차원이 다른 기존 가중치를 그대로 가져오는 방식은 사용하지 않습니다.

## 문답 추가 학습과 생성 평가

다음은 실제 수행한 큰 모델의 문답 학습 명령입니다. 결과 폴더가 이미 있으면 재실행 전에 --output을 새 이름으로 지정합니다.

~~~powershell
.\.venv\Scripts\python.exe train_intent_from_pretrained.py --source artifacts/capacity-256-v1/models/step-00015879 --data data/intent_dialogue --output artifacts/intent-capacity256-v1
~~~

원래 모델 가중치에서 시작하고 이 단계의 옵티마이저는 새로 초기화합니다. 기본 상한은 30회입니다. 시작 모델도 후보로 포함하고 답변 부분 검증 손실이 가장 낮은 회차를 저장합니다.

~~~powershell
.\.venv\Scripts\python.exe evaluate_dialogue_30.py --model artifacts/intent-capacity256-v1 --questions data/intent_expanded/test_questions.jsonl
~~~

training_log.json에 학습·선택 기록, validation_answers.json에 검증 질문 생성, test_answers.json에 30개 공통 질문 답변이 저장됩니다. 자동 pending 표시는 수동 판정 전 상태입니다.

## 최근 결과와 해석

| 같은 조건의 실험 | 128차원 | 256차원 |
| --- | ---: | ---: |
| 파라미터 | 1,321,728 | 4,216,320 |
| 말뭉치 1회 후 전체 발화 검증 손실 | 4.8587 | 4.7258 |
| 50개 문답 추가 학습 최저 검증 손실 | 3.3311 | 3.2126 |
| 공통 생성 평가의 명확한 통과 | 수동 1/30 | 수동 3/30 |

큰 모델의 손실이 더 낮고 일부 생성 이점이 관찰됐습니다. 그러나 두 모델 모두 주제 혼합·반복·문장 오류가 남았고, 자유 대화 능력 확보로 판단하지 않습니다. 이 비교는 단일 시드·작은 개발 평가이며 CPU 비용도 큰 모델이 약 두 배였습니다.

공통 30개 질문과 검증 15개는 이미 여러 번 사용한 개발 자료입니다. 새 최종 평가로 취급하지 않습니다. 학습 문답의 재현율과 자유 생성 점수를 구분하고, 토크나이저·손실 대상이 다른 수치를 직접 비교하지 않습니다.

## 다음 단계

현재 규모 비교를 결과와 함께 문서화하고, 후속 가설·변경 조건·종료 기준을 먼저 정합니다. 같은 조건의 무제한 추가 학습과 1,000개까지의 기계적 문답 확대는 보류했습니다. 직접 모델의 Gradio 통합과 새 질문·다중 턴 최종 검수는 남아 있습니다.

## 데이터 출처와 저장소 관리

- SNS 사전학습 원본은 AI Hub 한국어 SNS입니다. 원본 신청·이용·배포 조건에 맞춰 관리합니다.
- 500·600개 문답 실험에는 [songys/Chatbot_data](https://github.com/songys/Chatbot_data)의 일부가 포함됩니다. 해당 실험 폴더의 출처·원본 행 기록과 MIT 라이선스를 함께 유지합니다.
- .venv, 캐시, 인증 정보와 모델 가중치는 .gitignore 및 커밋 목록에서 확인합니다. 체크포인트의 checkpoint.pt는 모델뿐 아니라 옵티마이저와 상태를 포함하므로 공개 여부를 별도로 정합니다.
- 이 저장소의 실행 안내와 기존 학습 파일만으로 누락된 토크나이저·학습 데이터·모델을 자동 복구하는 기능은 없습니다.
