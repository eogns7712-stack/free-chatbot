"""직접 학습한 v3 모델의 단일 질문 채팅 화면. 프로젝트 루트에서 실행하세요."""

# 프로젝트 루트에서 실행하되, 이동 후에도 데이터·모델 경로와 공통 함수 import를 유지합니다.
import sys as _sys
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path
from threading import Lock

import torch
from tokenizers import Tokenizer
from transformers import GPT2LMHeadModel

ROOT = PROJECT_ROOT
DEFAULT_MODEL = ROOT / 'artifacts/short-dialogue-mixed-v3/epochs/epoch-06'


class LocalChat:
    def __init__(self, folder):
        folder = Path(folder).resolve()
        if not (folder / 'config.json').is_file() or not (folder / 'tokenizer.json').is_file():
            raise FileNotFoundError(f'모델과 토크나이저 파일을 확인하세요: {folder}')
        torch.set_num_threads(4)
        self.tokenizer = Tokenizer.from_file(str(folder / 'tokenizer.json'))
        self.model = GPT2LMHeadModel.from_pretrained(str(folder), local_files_only=True).eval()
        self.ids = {s: self.tokenizer.token_to_id(s) for s in ('<user>', '<assistant>', '<eos>', '<pad>')}
        if any(v is None for v in self.ids.values()):
            raise ValueError('필요한 특수 토큰이 없습니다.')
        if self.model.config.vocab_size != self.tokenizer.get_vocab_size():
            raise ValueError('모델과 토크나이저 어휘 크기가 다릅니다.')
        self.lock = Lock()

    def reply(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError('메시지를 입력해 주세요.')
        question = question.strip()
        for token in ('<pad>', '<unk>', '<user>', '<assistant>', '<eos>'):
            if token in question:
                raise ValueError('모델용 특수 토큰은 입력에 사용할 수 없습니다.')
        prompt = [self.ids['<user>'], *self.tokenizer.encode(question).ids,
                  self.ids['<eos>'], self.ids['<assistant>']]
        # 평가 때와 같은 최대 48토큰 응답 공간을 확보합니다.
        if len(prompt) + 48 > self.model.config.n_positions:
            raise ValueError('입력이 길어요. 짧은 문장으로 나누어 보내 주세요.')
        inputs = torch.tensor([prompt], dtype=torch.long)
        with self.lock, torch.inference_mode():
            output = self.model.generate(input_ids=inputs, attention_mask=torch.ones_like(inputs),
                                         max_new_tokens=48, do_sample=False, num_beams=1,
                                         repetition_penalty=1.0, eos_token_id=self.ids['<eos>'],
                                         pad_token_id=self.ids['<pad>'])
        answer = self.tokenizer.decode(output[0, len(prompt):].tolist(), skip_special_tokens=True).strip()
        return answer if answer else '(모델이 빈 답변을 생성했습니다.)'


def build_demo(chat):
    import gradio as gr

    def respond(message, history):
        # 화면 기록은 유지하지만, 단일 문답으로 학습한 모델에는 현재 입력만 전달합니다.
        try:
            return chat.reply(message)
        except ValueError as error:
            raise gr.Error(str(error)) from error

    return gr.ChatInterface(
        fn=respond,
        title='직접 학습한 챗봇 · v3',
        description=('인사·피곤함·산책·영화·식물에 관한 짧은 대화를 시험해 보세요. '
                     '답변은 현재 메시지만 보고 생성합니다. 이전 대화를 기억하는 기능은 아직 없습니다.'),
        textbox=gr.Textbox(placeholder='짧은 메시지를 입력해 주세요.', lines=2, max_lines=5,
                           submit_btn='보내기', stop_btn='중지'),
        examples=['안녕, 이야기하러 왔어.', '잠을 적게 자서 졸려.', '공원에서 산책하고 왔어.',
                  '오늘 영화를 보려고 해.', '키우는 식물에 새잎이 났어.'],
        examples_label='시험할 메시지',
        cache_examples=False, save_history=False, analytics_enabled=False,
        flagging_mode='never', concurrency_limit=1,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--check', action='store_true', help='서버 없이 예시 응답과 화면 구성을 확인')
    args = parser.parse_args()
    chat = LocalChat(args.model)
    print(f'모델: {args.model.resolve()}', flush=True)
    demo = build_demo(chat)
    if args.check:
        for question in ('안녕, 이야기하러 왔어.', '잠을 적게 자서 졸려.', '오늘 영화를 보려고 해.'):
            print(f'질문: {question}\n답변: {chat.reply(question)}', flush=True)
        print('화면 구성 및 추론 확인 완료. 서버는 시작하지 않았습니다.', flush=True)
        return
    demo.launch(server_name='127.0.0.1', server_port=args.port, share=False, inbrowser=False)


if __name__ == '__main__':
    main()
