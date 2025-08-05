
from __future__ import annotations

import argparse
import os
import re
from typing import List

# ─────────────────────────────────────── Disable static KV‑cache *before* Unsloth import
os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "0")

import numpy as np
import torch
import pandas as pd
from datasets import Dataset
from tqdm import tqdm
from unsloth import FastLanguageModel
import evaluate

# ─────────────────────────────────────── Helpers

def load_dataset(csv_path: str) -> Dataset:
    """Read a two‑column CSV with *prompt* and *completion* fields."""
    df = pd.read_csv(csv_path)
    required = {"prompt", "completion"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required} — got {df.columns.tolist()}")
    return Dataset.from_pandas(df[list(required)])


def format_prompts(tokenizer, prompts: List[str]) -> List[str]:
    """Wrap each article exactly like during training."""
    messages = [
        [{"role": "user", "content": f"Summarize the following text:\n{p}"}]
        for p in prompts
    ]

    if getattr(tokenizer, "chat_template", None):
        return [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages
        ]

    template = (
        "<|im_start|>user\nSummarize the following text:\n{content}\n<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    return [template.format(content=p) for p in prompts]


def _strip_prompt_echo(text: str) -> str:
    """Remove leading template echo like "Summarize the following text:" if present."""
    pattern = r"^(?:\s*Summarize the following text:\s*)+"
    cleaned = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return cleaned.lstrip()


@torch.inference_mode()

def generate(
    model,
    tokenizer,
    prompts: List[str],
    *,
    max_new_tokens: int,
    length_penalty: float,
    device: torch.device,
    do_sample: bool,
    num_beams: int,
    top_p: float,
    temperature: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
):
    """Unified generation wrapper supporting beam‑search *and* sampling."""
    toks = tokenizer(
        prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True,
    )

    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            num_beams=1,  # beam search off when sampling
        )
    else:
        gen_kwargs.update(
            do_sample=False,
            num_beams=num_beams,
        )

    outputs = model.generate(**toks, **gen_kwargs)
    gen = outputs[:, toks.input_ids.shape[1]:]
    decoded = [tokenizer.decode(ids, skip_special_tokens=True).strip() for ids in gen]
    return [_strip_prompt_echo(t) for t in decoded]


# ─────────────────────────────────────── Main

def main():
    ap = argparse.ArgumentParser(description="Evaluate a summarisation QLoRA‑adapted Qwen3 model.")
    ap.add_argument("--adapter_dir", default="outputs", help="Directory with LoRA adapter (adapter_config.json etc.)")
    ap.add_argument("--base_model", default="unsloth/Qwen3-30B-A3B-Base-bnb-4bit", help="4‑bit base model")
    ap.add_argument("--test_csv", default="test.csv", help="CSV with prompt & completion columns")
    ap.add_argument("--results_csv", default="results.csv", help="Where to write per‑example results")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--length_penalty", type=float, default=1.2, help=">1.0 → shorter, <1.0 → longer")

    # Generation knobs mirrored from official docs
    ap.add_argument("--do_sample", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)

    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    # 1️⃣ Load 4‑bit base model (static KV‑cache disabled)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        load_in_4bit=True,
        trust_remote_code=True,
        max_seq_length=2048,
        fast_inference=False,  # critical for Qwen3
        device_map={"": 0}, 
    )

    # 2️⃣ Load LoRA
    if not os.path.isdir(args.adapter_dir):
        raise FileNotFoundError(f"Adapter dir '{args.adapter_dir}' does not exist")
    model.load_adapter(args.adapter_dir)

    # 3️⃣ Switch to inference kernels & enforce dynamic cache
    FastLanguageModel.for_inference(model)
    model.generation_config.cache_implementation = "dynamic"

    device = torch.device(args.device)
    model.eval()

    # 4️⃣ Data & metrics
    dataset = load_dataset(args.test_csv)
    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")

    preds, refs = [], []
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Generating"):
        batch = dataset[i : i + args.batch_size]
        prompt_texts = format_prompts(tokenizer, batch["prompt"])
        preds.extend(
            generate(
                model,
                tokenizer,
                prompt_texts,
                max_new_tokens=args.max_new_tokens,
                length_penalty=args.length_penalty,
                device=device,
                do_sample=args.do_sample,
                num_beams=args.num_beams,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
        )
        refs.extend(batch["completion"])

    # 5️⃣ Metrics – individual & aggregate
    rouge_indiv = rouge.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True,
        use_aggregator=False,
    )
    rouge_agg = {k: float(np.mean(v)) for k, v in rouge_indiv.items()}

    bert_scores = bert.compute(predictions=preds, references=refs, lang="en")
    bert_agg = {
        "precision": float(np.mean(bert_scores["precision"])),
        "recall":    float(np.mean(bert_scores["recall"])),
        "f1":        float(np.mean(bert_scores["f1"])),
    }

    # 6️⃣ Save per‑example details
    df_out = pd.DataFrame({
        "prompt":          dataset["prompt"],
        "reference":       refs,
        "prediction":      preds,
        "rouge1":          rouge_indiv["rouge1"],
        "rouge2":          rouge_indiv["rouge2"],
        "rougeL":          rouge_indiv["rougeL"],
        "rougeLsum":       rouge_indiv["rougeLsum"],
        "bert_precision":  bert_scores["precision"],
        "bert_recall":     bert_scores["recall"],
        "bert_f1":         bert_scores["f1"],
    })
    df_out.to_csv(args.results_csv, index=False)
    print(f"\nPer‑example results written to {args.results_csv}")

    # 7️⃣ Print aggregate overview
    print("\n=== Aggregate ROUGE ===")
    for k, v in rouge_agg.items():
        print(f"{k}: {v:.4f}")

    print("\n=== Aggregate BERTScore ===")
    for k, v in bert_agg.items():
        print(f"{k.title()}: {v:.4f}")
    print()


if __name__ == "__main__":
    main()
