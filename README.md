# 자유주제 AI 챗봇

Python과 Hugging Face 모델을 사용해 자유주제로 한국어 대화를 나누는 로컬 챗봇입니다. 터미널과 Gradio 웹 화면에서 질문할 수 있고, 최근 질문·답변 최대 3쌍을 후속 답변에 사용합니다.

현재 두 실행 방식이 구현돼 있으며, 다음 단계에서는 고정 평가 사례로 답변 품질을 측정하고 모델과 생성 설정을 비교합니다.

## 프로젝트 구성

| 파일 | 역할 |
| --- | --- |
| app.py | 모델 로딩, 공통 답변 생성 함수, 터미널 채팅 |
| web_app.py | 공통 답변 함수를 사용하는 Gradio 웹 화면 |
| [PLAN.md](PLAN.md) | 2~4주 날짜별 일정, 기술 적용, 모델 평가 기준 |
| [PROGRESS.md](PROGRESS.md) | 구현 현황, 현재 모델 설정, 다음 작업과 실험 기록 |
| README.md | 프로젝트 소개와 설치·실행 안내 |

## 설치

Windows PowerShell을 기준으로 합니다. Python과 VS Code의 Python 확장을 설치하고 app.py와 web_app.py가 들어 있는 프로젝트 폴더를 VS Code에서 엽니다. 아래 버전은 기존 개발 환경에서 확인한 직접 의존성입니다.

~~~powershell
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.15.0" "transformers==5.22.2" "gradio==6.31.0"
~~~

기존 개발 환경의 Python 버전은 **3.14.5**입니다. VS Code의 명령 팔레트에서 **Python: Select Interpreter**를 실행해 프로젝트의 .venv를 선택합니다. 첫 실행에는 공개 모델과 토크나이저를 내려받을 인터넷 연결과 모델 캐시를 저장할 디스크 공간이 필요합니다. 모델을 캐시한 뒤에는 같은 환경에서 캐시를 재사용할 수 있습니다. [Qwen 모델 카드](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)

## 실행

### 터미널 채팅

~~~powershell
.\.venv\Scripts\python.exe app.py
~~~

질문을 입력해 대화합니다. **/reset**은 기록을 초기화하고 **/exit**는 프로그램을 종료합니다.

### 브라우저 채팅

~~~powershell
.\.venv\Scripts\python.exe web_app.py
~~~

터미널에 표시되는 로컬 주소를 브라우저에서 엽니다. 채팅 기록을 지우면 새 대화를 시작할 수 있습니다. 서버를 종료하려면 터미널에서 **Ctrl+C**를 누릅니다.

## 대화 예시

1. “비 오는 날 집에서 할 수 있는 활동을 세 가지 추천해줘.”
2. “그중 두 번째 활동에 필요한 준비물만 알려줘.”
3. 대화를 초기화한 뒤 같은 후속 질문을 보내 앞 기록이 이어지지 않는지 확인합니다.

## 동작 방식과 한계

두 화면은 app.py의 generate_reply()를 공유합니다. 토크나이저가 시스템 메시지, 최근 대화, 새 질문을 모델 입력으로 바꾸고 모델이 새로 생성한 토큰만 화면에 표시합니다. [Hugging Face 채팅 템플릿 문서](https://huggingface.co/docs/transformers/chat_templating)

현재 모델은 **Qwen/Qwen2.5-0.5B-Instruct**입니다. 답변은 부정확하거나 요청한 형식을 놓칠 수 있으며, 최근 기록 범위보다 오래된 정보는 다음 응답의 입력에서 빠집니다. 품질 검수와 비교 일정은 [PLAN.md](PLAN.md), 검수 진행 상태는 [PROGRESS.md](PROGRESS.md)에서 확인할 수 있습니다.

## GitHub에 올릴 파일

프로젝트 폴더에 세 문서를 추가하고 app.py, web_app.py와 함께 올립니다. .venv/, __pycache__/, 로컬 모델 캐시, 토큰이 들어 있는 파일은 Git 추적에서 제외합니다. requirements.txt는 의존성 정리 및 재현 검수 단계에서 추가할 예정입니다.
