"""Demo 1: LoRA vs QLoRA vs full fine-tuning on Banking77 intent classification (Qwen2.5-1.5B-Instruct).

One script per arm, same data, same prompt, same seed. Writes results/<arm>/{metrics.json,preds.jsonl,train_log.jsonl}.
Usage (one GPU):
  python run_arm.py --arm base|lora|qlora|full
"""
import argparse
import csv
import json
import math
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42


def load_split(name):
    with open(os.path.join(ROOT, "data", f"banking77_{name}.csv"), newline="", encoding="utf-8") as f:
        return [(r["text"].strip(), r["category"].strip()) for r in csv.DictReader(f)]


def build_prompt(tok, labels, text):
    msgs = [
        {"role": "system", "content": "You are an intent classifier for a bank's customer support."},
        {"role": "user", "content": "Classify the customer message into exactly one of these intents:\n"
                                    + ", ".join(labels) + f"\n\nMessage: {text}\nAnswer with the intent label only."},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def encode_train(tok, labels, rows, max_len=768):
    out = []
    for text, label in rows:
        p = tok(build_prompt(tok, labels, text), add_special_tokens=False)["input_ids"]
        t = tok(label + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (p + t)[:max_len]
        lab = ([-100] * len(p) + t)[:max_len]
        out.append((ids, lab))
    return out


def batches(data, bs, shuffle, rng):
    idx = list(range(len(data)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [data[j] for j in idx[i:i + bs]]


def collate(batch, pad_id, device):
    n = max(len(x[0]) for x in batch)
    ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
    lab = torch.full((len(batch), n), -100, dtype=torch.long)
    att = torch.zeros((len(batch), n), dtype=torch.long)
    for i, (a, b) in enumerate(batch):
        ids[i, :len(a)] = torch.tensor(a)
        lab[i, :len(b)] = torch.tensor(b)
        att[i, :len(a)] = 1
    return ids.to(device), lab.to(device), att.to(device)


@torch.no_grad()
def evaluate(model, tok, labels, rows, bs=64):
    model.eval()
    tok.padding_side = "left"
    preds = []
    t0 = time.time()
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        enc = tok([build_prompt(tok, labels, t) for t, _ in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(model.device)
        out = model.generate(**enc, max_new_tokens=16, do_sample=False, pad_token_id=tok.pad_token_id)
        for (text, gold), seq in zip(chunk, out[:, enc["input_ids"].shape[1]:]):
            ans = tok.decode(seq, skip_special_tokens=True).strip().split("\n")[0].strip().strip(".").strip()
            preds.append({"text": text, "gold": gold, "pred": ans, "valid": ans in labels, "correct": ans == gold})
    tok.padding_side = "right"
    return preds, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["base", "lora", "qlora", "full"])
    ap.add_argument("--epochs", type=float, default=2)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--save-adapter", default="", help="lora/qlora only: also save the trained adapter and tokenizer here")
    a = ap.parse_args()

    random.seed(SEED)
    torch.manual_seed(SEED)
    # shared GPU: hard cap so colleagues' services keep their memory
    torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_FRACTION", "0.3")), 0)
    out_dir = os.path.join(ROOT, "results", a.arm)
    os.makedirs(out_dir, exist_ok=True)

    labels = sorted({c for _, c in load_split("train")})
    train, test = load_split("train"), load_split("test")
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "right"

    kw = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if a.arm == "qlora":
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).to("cuda") if a.arm != "qlora" else \
        AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, **kw)
    torch.cuda.reset_peak_memory_stats()

    total_params = sum(p.numel() for p in model.parameters())
    trainable = 0
    train_secs = 0.0
    log = []
    if a.arm != "base":
        if a.arm in ("lora", "qlora"):
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if a.arm == "qlora":
                model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                        gradient_checkpointing_kwargs={"use_reentrant": False})
            else:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                model.enable_input_require_grads()
            model.config.use_cache = False
            cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
            model = get_peft_model(model, cfg)
            lr = 2e-4
        else:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.config.use_cache = False
            lr = 1e-5
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        data = encode_train(tok, labels, train)
        steps_per_epoch = math.ceil(len(data) / (a.bs * a.accum))
        total_steps = int(steps_per_epoch * a.epochs)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)
        warm = max(1, int(0.03 * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: min(1.0, (s + 1) / warm) * max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))))
        rng = random.Random(SEED)
        model.train()
        step, micro, run_loss = 0, 0, 0.0
        t0 = time.time()
        while step < total_steps:
            for b in batches(data, a.bs, True, rng):
                ids, lab, att = collate(b, tok.pad_token_id, model.device)
                loss = model(input_ids=ids, attention_mask=att, labels=lab).loss / a.accum
                loss.backward()
                run_loss += loss.item()
                micro += 1
                if micro % a.accum == 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    if step % 10 == 0 or step == total_steps:
                        log.append({"step": step, "epoch": round(step / steps_per_epoch, 3), "loss": round(run_loss / 10 if step % 10 == 0 else run_loss, 4),
                                    "lr": sched.get_last_lr()[0], "elapsed_s": round(time.time() - t0, 1)})
                        print(json.dumps(log[-1]), flush=True)
                        run_loss = 0.0
                    if step >= total_steps:
                        break
        train_secs = time.time() - t0
        model.config.use_cache = True
        if hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()
        with open(os.path.join(out_dir, "train_log.jsonl"), "w") as f:
            f.writelines(json.dumps(r) + "\n" for r in log)
        if a.save_adapter and a.arm in ("lora", "qlora"):
            model.save_pretrained(a.save_adapter)
            tok.save_pretrained(a.save_adapter)
            with open(os.path.join(a.save_adapter, "labels.json"), "w") as f:
                json.dump(labels, f, indent=1)

    preds, eval_secs = evaluate(model, tok, labels, test)
    with open(os.path.join(out_dir, "preds.jsonl"), "w") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in preds)
    acc = sum(p["correct"] for p in preds) / len(preds)
    valid = sum(p["valid"] for p in preds) / len(preds)
    metrics = {"arm": a.arm, "model": MODEL, "seed": SEED, "n_test": len(preds), "accuracy": acc, "valid_label_rate": valid,
               "total_params": total_params, "trainable_params": trainable, "train_seconds": round(train_secs, 1),
               "eval_seconds": round(eval_secs, 1), "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
               "gpu": torch.cuda.get_device_name(0), "epochs": a.epochs if a.arm != "base" else 0}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print("METRICS", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
