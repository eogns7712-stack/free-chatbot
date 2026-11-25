"""프로젝트 루트에서 실행하는 CPU 학습 속도 측정입니다."""

# 프로젝트 루트에서 실행하되, 이동 후에도 데이터·모델 경로와 공통 함수 import를 유지합니다.
import sys as _sys
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(PROJECT_ROOT))

import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = PROJECT_ROOT


def main():
    torch.manual_seed(42)
    torch.set_num_threads(4)
    tokenizer = Tokenizer.from_file(
        str(ROOT / "artifacts" / "sns-tokenizer-v1" / "tokenizer.json")
    )
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=tokenizer.get_vocab_size(),
        n_positions=128,
        n_embd=128,
        n_layer=4,
        n_head=4,
        bos_token_id=None,
        eos_token_id=tokenizer.token_to_id("<eos>"),
        pad_token_id=tokenizer.token_to_id("<pad>"),
    ))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    model.train()

    # 속도 측정용 가상 토큰입니다. 학습 결과는 저장하지 않습니다.
    inputs = torch.randint(5, tokenizer.get_vocab_size(), (4, 128))
    labels = inputs.clone()
    labels[:, :64] = -100

    def step():
        optimizer.zero_grad()
        loss = model(input_ids=inputs, labels=labels).loss
        loss.backward()
        optimizer.step()

    print("CPU | 스레드 4 | 배치 4 | 길이 128 | 4층·128차원", flush=True)
    print("파라미터 수:", sum(p.numel() for p in model.parameters()), flush=True)
    for _ in range(3):
        step()

    started = time.perf_counter()
    for index in range(1, 31):
        step()
        if index % 10 == 0:
            print(f"측정 {index}/30", flush=True)

    seconds = (time.perf_counter() - started) / 30
    # 30분 중 20분을 학습 연산에 배정한 대략적인 예산입니다.
    steps = int(20 * 60 / seconds)
    print(f"1스텝 평균: {seconds:.3f}초")
    print(f"학습 연산 20분의 예상 스텝 수: {steps}")
    print(f"예상 예제 처리 횟수: {steps * 4}")
    print("가상 입력 기준의 추정입니다. 데이터 로딩·검증·저장 시간은 별도입니다.")


if __name__ == "__main__":
    main()
