"""현재 모델과 4층·256차원 후보의 CPU 연산 비용 비교."""
import gc
import json
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

ROOT = Path(__file__).resolve().parent


def benchmark(vocab_size, width, heads, eos, pad):
    torch.manual_seed(42)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=vocab_size, n_positions=128,
        n_embd=width, n_layer=4, n_head=heads,
        bos_token_id=None, eos_token_id=eos, pad_token_id=pad,
    ))
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    inputs = torch.randint(5, vocab_size, (4, 128))

    def step():
        optimizer.zero_grad()
        loss = model(input_ids=inputs, labels=inputs).loss
        loss.backward()
        optimizer.step()

    for _ in range(3):
        step()
    started = time.perf_counter()
    for _ in range(20):
        step()
    seconds = (time.perf_counter() - started) / 20
    return {
        "width": width, "layers": 4, "heads": heads,
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds_per_step": seconds,
        "estimated_steps_in_20min": int(1200 / seconds),
        "note": "가상 입력 학습 연산. 데이터 읽기·검증·저장 시간은 별도.",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=ROOT / "artifacts/sns-tokenizer-v1/tokenizer.json")
    parser.add_argument("--output", type=Path, default=ROOT / "capacity_benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    tok = Tokenizer.from_file(str(args.tokenizer))
    results = []
    for width, heads in [(128,4),(256,8)]:
        result = benchmark(tok.get_vocab_size(),width,heads,tok.token_to_id("<eos>"),tok.token_to_id("<pad>"))
        result["note"] = "가상 입력 학습 연산. 데이터 읽기·검증·저장 시간은 별도."
        results.append(result)
        print(json.dumps(result,ensure_ascii=False),flush=True)
        gc.collect()
    args.output.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
