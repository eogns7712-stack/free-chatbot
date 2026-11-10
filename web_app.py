import gradio as gr

from app import HISTORY_LIMIT, SYSTEM_MESSAGE, generate_reply


def reply_in_browser(message, history):
    question = message.strip()
    if not question:
        raise gr.Error("메시지를 입력해 주세요.")

    text_history = []

    for entry in history:
        role = entry["role"]
        if role not in ("user", "assistant"):
            continue

        content = entry["content"]
        if isinstance(content, list):
            content = "\n".join(
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            )

        if isinstance(content, str) and content.strip():
            text_history.append({"role": role, "content": content})

    messages = [
        SYSTEM_MESSAGE,
        *text_history[-HISTORY_LIMIT:],
        {"role": "user", "content": question},
    ]
    return generate_reply(messages)


if __name__ == "__main__":
    demo = gr.ChatInterface(fn=reply_in_browser, title="자유형 AI 챗봇")
    demo.launch(share=False)