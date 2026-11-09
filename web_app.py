import gradio as gr

from app import HISTORY_LIMIT, SYSTEM_MESSAGE, generate_reply


def reply_in_browser(message, history):
    question = message.strip()
    if not question:
        raise gr.Error("메시지를 입력해 주세요.")

    text_history = [
        {"role": entry["role"], "content": entry["content"]}
        for entry in history
        if entry["role"] in ("user", "assistant")
        and isinstance(entry["content"], str)
    ]

    messages = [
        SYSTEM_MESSAGE,
        *text_history[-HISTORY_LIMIT:],
        {"role": "user", "content": question},
    ]
    return generate_reply(messages)


if __name__ == "__main__":
    demo = gr.ChatInterface(fn=reply_in_browser, title="자유형 AI 챗봇")
    demo.launch(share=False)