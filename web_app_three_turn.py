"""직전 문답 두 쌍과 현재 질문을 전달하는 실험 UI."""
import argparse
from pathlib import Path
import gradio as gr
from three_turn_runtime import ThreeTurnModel


def text(content):
    if isinstance(content,str):return content
    if isinstance(content,list):
        return '\n'.join(b['text'] for b in content if isinstance(b,dict) and b.get('type')=='text' and isinstance(b.get('text'),str))
    return ''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--port',type=int,default=7860)
    a=p.parse_args();engine=ThreeTurnModel(a.model)

    def respond(message,history):
        if not isinstance(message,str) or not message.strip():raise gr.Error('메시지를 입력해 주세요.')
        if any(t in message for t in ('<user>','<assistant>','<eos>','<pad>','<unk>')):
            raise gr.Error('모델용 특수 토큰은 입력할 수 없습니다.')
        past=[{'role':m['role'],'content':text(m.get('content'))} for m in history if m.get('role') in ('user','assistant')]
        past=[m for m in past if m['content'].strip()]
        recent=past[-4:] if len(past)>=4 else past[-2:]
        if [m['role'] for m in recent] != ['user','assistant']*(len(recent)//2):
            raise gr.Error('대화 기록을 초기화한 뒤 다시 시작해 주세요.')
        try:return engine.generate([*recent,{'role':'user','content':message.strip()}]) or '(빈 답변이 생성됐습니다.)'
        except ValueError as error:raise gr.Error(str(error)) from error

    demo=gr.ChatInterface(fn=respond,title='직접 학습한 챗봇 · 짧은 후속 대화',
        description='최근 두 문답과 현재 메시지를 함께 봅니다. 짧은 3턴 대화 실험이며 장기 기억은 지원하지 않습니다. 주제 전환도 시험할 수 있습니다. 대화가 길면 새 대화를 시작하세요.',
        textbox=gr.Textbox(lines=2,placeholder='짧게 입력해 주세요.',submit_btn='보내기',stop_btn='중지'),
        save_history=False,analytics_enabled=False,cache_examples=False,flagging_mode='never',concurrency_limit=1)
    demo.launch(server_name='127.0.0.1',server_port=a.port,share=False,inbrowser=False)


if __name__=='__main__':main()
