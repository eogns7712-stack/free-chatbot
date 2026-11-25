"""기존 체크포인트를 목표 누적 스텝까지 학습합니다."""

# 프로젝트 루트에서 실행하되, 이동 후에도 데이터·모델 경로와 공통 함수 import를 유지합니다.
import sys as _sys
from pathlib import Path as _Path
PROJECT_ROOT = _Path(__file__).resolve().parents[1]
_sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from tokenizers import Tokenizer
from transformers import GPT2Config, GPT2LMHeadModel

import numpy as np
from train_sns_30min import make_batch, validation_loss

ROOT = PROJECT_ROOT
BATCH_SIZE = 4
LEARNING_RATE = 0.0003



class BlockDataset:
    def __init__(self, path, tokenizer):
        report = json.loads((path.parent / "report.json").read_text(encoding="utf-8"))
        if report["status"] != "complete":
            raise ValueError("완료되지 않은 말뭉치입니다.")
        if report["block_size"] != 128 or report["dtype"] != "<u2":
            raise ValueError("말뭉치 형식이 다릅니다.")
        # 파일 직렬화와 무관하게 토크나이저의 어휘 크기와 패딩 ID를 확인합니다.
        if tokenizer.get_vocab_size() != report["vocab_size"]:
            raise ValueError("어휘 크기가 다릅니다.")
        if tokenizer.token_to_id("<pad>") != report["pad_token_id"]:
            raise ValueError("패딩 토큰 ID가 다릅니다.")
        count = report["splits"][path.stem]["blocks"]
        if path.stat().st_size != count * 128 * 2:
            raise ValueError("말뭉치 파일 크기가 기록과 다릅니다.")
        self.rows = np.memmap(path, dtype="<u2", mode="r", shape=(count, 128))
        self.indices = None
        if path.stem == "validation":
            # 모든 실행에서 같은 1024개 블록을 검증합니다.
            self.indices = np.random.default_rng(42).choice(
                count, size=min(1024, count), replace=False
            )
        self.pad_id = report["pad_token_id"]

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.rows)

    def __getitem__(self, index):
        actual = int(self.indices[index]) if self.indices is not None else index
        ids = torch.tensor(self.rows[actual].astype(np.int64))
        labels = ids.clone()
        labels[ids == self.pad_id] = -100
        return ids, labels


def load_examples(path, tokenizer):
    return BlockDataset(path, tokenizer)


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_checkpoint(folder, model, optimizer, state, signature):
    checkpoint = {
        "version": 1,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "state": state,
        "signature": signature,
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
    }
    temporary = folder / "checkpoint.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, folder / "checkpoint.pt")
    log = {**state, "signature": signature}
    log.pop("order", None)
    temporary_log = folder / "training_log.tmp"
    temporary_log.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_log, folder / "training_log.json")


def check_validation(model, examples, pad_id, deadline, state, folder, tokenizer):
    if state["last_validation_step"] == state["step"]:
        return True
    value = validation_loss(model, examples, pad_id, deadline)
    if value is None:
        return False
    row = {
        "epoch": state["epoch"],
        "step": state["step"],
        "epoch_complete": state["cursor"] >= len(state["order"]),
        "train_loss": state["loss_sum"] / state["token_count"],
        "validation_loss": value,
    }
    if state["best_loss"] is None or value < state["best_loss"]:
        relative = f"models/step-{state['step']:08d}"
        target = folder / relative
        model.save_pretrained(str(target))
        tokenizer.save(str(target / "tokenizer.json"))
        state["best_loss"] = value
        state["best_model"] = relative
    state["history"].append(row)
    state["last_validation_step"] = state["step"]
    print(json.dumps(row, ensure_ascii=False), flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/sns_pretrain_blocks")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/capacity-128-v1")
    parser.add_argument("--init-from", type=Path)
    parser.add_argument("--width", type=int, choices=[128, 256], default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--steps", type=int, help="이번 실행의 스텝 상한. 짧은 동작 검사용")
    parser.add_argument("--target-step", type=int, required=True, help="도달 후 추가 학습하지 않는 누적 스텝 목표")
    args = parser.parse_args()
    if args.target_step < 1:
        parser.error("--target-step은 1 이상이어야 합니다.")
    if args.init_from:
        parser.error("규모 비교는 양쪽 모두 무작위 초기화합니다. --init-from을 사용하지 마세요.")
    if args.resume and args.init_from:
        parser.error("--resume과 --init-from은 함께 사용할 수 없습니다.")
    if args.minutes <= 0 or (args.steps is not None and args.steps <= 0):
        parser.error("시간과 스텝은 양수여야 합니다.")
    if not args.resume and args.output.exists():
        raise FileExistsError(f"이미 폴더가 있습니다. 재개하려면 --resume: {args.output}")

    started = time.perf_counter()
    # 실행 예산의 2/3는 학습, 나머지는 검증·저장용입니다.
    training_deadline = started + args.minutes * 60 * (2 / 3)
    validation_deadline = started + args.minutes * 60 * (28 / 30)
    torch.set_num_threads(4)
    random.seed(42)
    torch.manual_seed(42)
    if args.resume:
        checkpoint = torch.load(
            args.output / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        tokenizer_path = args.output / "tokenizer.json"
    else:
        checkpoint = None
        tokenizer_path = (
            args.init_from / "tokenizer.json" if args.init_from
            else ROOT / "artifacts/sns-tokenizer-v1/tokenizer.json"
        )
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    corpus_report = json.loads((args.data / "report.json").read_text(encoding="utf-8"))
    if fingerprint(tokenizer_path) != corpus_report["tokenizer_sha256"]:
        raise ValueError("말뭉치를 만든 토크나이저와 다릅니다.")
    signature = {
        "train_sha256": fingerprint(args.data / "train.bin"),
        "validation_sha256": fingerprint(args.data / "validation.bin"),
        "tokenizer_sha256": fingerprint(tokenizer_path),
        "report_sha256": fingerprint(args.data / "report.json"),
        "validation_limit": 1024,
        "model_width": args.width,
        "model_heads": args.width // 32,
        "objective": "fulltext_all_real_tokens",
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }
    if checkpoint and checkpoint["signature"] != signature:
        raise ValueError("데이터·토크나이저·배치·학습률이 저장된 조건과 다릅니다.")
    train = load_examples(args.data / "train.bin", tokenizer)
    validation = load_examples(args.data / "validation.bin", tokenizer)
    if checkpoint:
        model = GPT2LMHeadModel(GPT2Config.from_dict(checkpoint["config"]))
        model.load_state_dict(checkpoint["model"])
    elif args.init_from:
        model = GPT2LMHeadModel.from_pretrained(str(args.init_from))
    else:
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=tokenizer.get_vocab_size(), n_positions=128,
            n_embd=args.width, n_layer=4, n_head=args.width // 32, bos_token_id=None,
            eos_token_id=tokenizer.token_to_id("<eos>"),
            pad_token_id=tokenizer.token_to_id("<pad>"),
        ))
    if model.config.vocab_size != tokenizer.get_vocab_size() or model.config.n_positions != 128:
        raise ValueError("128토큰 모델과 해당 토크나이저가 필요합니다.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        state = checkpoint["state"]
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
    else:
        state = {
            "epoch": 1, "step": 0, "cursor": 0, "order": [],
            "loss_sum": 0.0, "token_count": 0,
            "best_loss": None, "best_model": None,
            "last_validation_step": None, "history": [],
            "initial_model": str(args.init_from) if args.init_from else "random",
        }
        args.output.mkdir(parents=True)
        tokenizer.save(str(args.output / "tokenizer.json"))
        save_checkpoint(args.output, model, optimizer, state, signature)

    start_step = state["step"]
    pad_id = tokenizer.token_to_id("<pad>")
    if start_step >= args.target_step:
        print(f"목표 이미 도달: {start_step} / {args.target_step}. 추가 학습 없이 종료합니다.", flush=True)
        return
    print(f"목표 누적 {args.target_step}스텝 | 남은 {args.target_step-start_step}스텝", flush=True)
    print(f"학습 {len(train)} | 검증 {len(validation)} | 재개 스텝 {start_step}", flush=True)
    if args.init_from:
        print("기존 가중치에서 시작합니다. 이전 옵티마이저 기록은 없어 새로 초기화했습니다.", flush=True)
    try:
        if state["step"] == 0 and state["best_loss"] is None:
            baseline = validation_loss(model, validation, pad_id, validation_deadline)
            if baseline is None:
                print("시작 검증 시간이 부족합니다. 다음 실행에 --resume을 사용하세요.", flush=True)
                return
            relative = "models/step-00000000"
            target = args.output / relative
            model.save_pretrained(str(target))
            tokenizer.save(str(target / "tokenizer.json"))
            state["best_loss"] = baseline
            state["best_model"] = relative
            state["last_validation_step"] = 0
            state["history"].append({
                "epoch": 0, "step": 0, "train_loss": None,
                "validation_loss": baseline,
            })
            save_checkpoint(args.output, model, optimizer, state, signature)
            print(f"시작 검증 손실: {baseline:.4f}", flush=True)
        while time.perf_counter() < training_deadline:
            if state["step"] >= args.target_step:
                break
            if args.steps and state["step"] - start_step >= args.steps:
                break
            if not state["order"]:
                state["order"] = list(range(len(train)))
                random.shuffle(state["order"])
            if state["cursor"] >= len(state["order"]):
                if not check_validation(model, validation, pad_id, validation_deadline, state, args.output, tokenizer):
                    break
                state["epoch"] += 1
                state["cursor"] = 0
                state["order"] = []
                state["loss_sum"] = 0.0
                state["token_count"] = 0
                save_checkpoint(args.output, model, optimizer, state, signature)
                continue
            indices = state["order"][state["cursor"]:state["cursor"] + BATCH_SIZE]
            ids, labels, mask, count = make_batch(train, indices, pad_id)
            model.train()
            optimizer.zero_grad()
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss).item():
                raise RuntimeError("손실이 유한하지 않습니다.")
            loss.backward()
            optimizer.step()
            state["cursor"] += len(indices)
            state["step"] += 1
            state["loss_sum"] += loss.item() * count
            state["token_count"] += count
            if state["step"] % 100 == 0:
                save_checkpoint(args.output, model, optimizer, state, signature)
                print(f"{state['epoch']}회차 | 누적 {state['step']}스텝 | {(time.perf_counter()-started)/60:.1f}분", flush=True)
        # 검증 중 종료해도 현재 학습 스텝은 재개할 수 있게 먼저 저장합니다.
        save_checkpoint(args.output, model, optimizer, state, signature)
        if state["step"] > start_step:
            check_validation(model, validation, pad_id, validation_deadline, state, args.output, tokenizer)
            save_checkpoint(args.output, model, optimizer, state, signature)
    except KeyboardInterrupt:
        print("\n중단했습니다. 다음 실행은 마지막 완전 저장 스텝에서 재개합니다.", flush=True)
        return

    print("완료 | 이번 학습 스텝:", state["step"] - start_step, flush=True)
    print("누적 스텝:", state["step"], flush=True)
    print("현재 실험의 최저 검증 손실 모델:", state["best_model"], flush=True)
    if state["step"] >= args.target_step:
        target_dir = args.output / "targets" / f"step-{state['step']:08d}"
        model.save_pretrained(str(target_dir))
        tokenizer.save(str(target_dir / "tokenizer.json"))
        print("목표 시점 모델 별도 저장:", target_dir, flush=True)
        print("목표 도달. 다음은 문답 추가 학습과 생성 평가입니다.", flush=True)
    else:
        print("목표 미달. 같은 --resume --target-step 명령으로 이어가세요.", flush=True)


if __name__ == "__main__":
    main()
