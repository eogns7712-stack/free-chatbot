from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
model.eval()


def generate_reply(messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    output_ids = model.generate(**inputs, max_new_tokens=128)
    input_length = inputs["input_ids"].shape[-1]
    reply_ids = output_ids[0, input_length:]
    return tokenizer.decode(reply_ids, skip_special_tokens=True).strip()


SYSTEM_MESSAGE = {
    "role": "system",
    "content": "사용자의 질문에 한국어로 간결하게 답해 주세요.",
}
HISTORY_LIMIT = 6  # 질문 3개와 답변 3개


def run_chat():
    history = []
    print("자유형 챗봇을 시작합니다. /reset으로 초기화, /exit로 종료합니다.")

    while True:
        question = input("\n나: ").strip()

        if not question:
            continue
        if question == "/exit":
            break
        if question == "/reset":
            history.clear()
            print("대화 기록을 초기화했습니다.")
            continue

        user_message = {"role": "user", "content": question}
        messages = [SYSTEM_MESSAGE, *history[-HISTORY_LIMIT:], user_message]

        reply = generate_reply(messages)
        print(f"챗봇: {reply}")

        history.extend([user_message, {"role": "assistant", "content": reply}])
        history = history[-HISTORY_LIMIT:]


if __name__ == "__main__":
    run_chat()