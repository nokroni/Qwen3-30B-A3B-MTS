
from __future__ import annotations

import argparse
import os
import re
from typing import List

# ─────────────────────────────────────── Disable static KV-cache *before* import
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
    df = pd.read_csv(csv_path)
    required = {"prompt", "completion"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV must contain columns {required} — got {df.columns.tolist()}")
    return Dataset.from_pandas(df[list(required)])


def format_prompts(tokenizer, prompts: List[str]) -> List[str]:
    """Apply the same ChatML-style template used during fine-tuning."""
    messages = [
        [{"role": "user", "content": f"Summarize the following text:\n{p}"}] for p in prompts
    ]
    if getattr(tokenizer, "chat_template", None):
        return [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    template = (
        "<|im_start|>user\nSummarize the following text:\n{content}\n<|im_end|>\n<|im_start|>assistant\n"
    )
    return [template.format(content=p) for p in prompts]


def _strip_prompt_echo(text: str) -> str:
    return re.sub(r"^(?:\s*Summarize the following text:\s*)+", "", text, flags=re.I).lstrip()


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
    toks = tokenizer(prompts, padding=True, return_tensors="pt", add_special_tokens=False).to(device)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        length_penalty=length_penalty,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=repetition_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=True,
    )
    if do_sample:
        gen_kwargs.update(dict(do_sample=True, top_p=top_p, temperature=temperature, num_beams=1))
    else:
        gen_kwargs.update(dict(do_sample=False, num_beams=num_beams))

    outputs = model.generate(**toks, **gen_kwargs)
    gen = outputs[:, toks.input_ids.shape[1]:]
    return [_strip_prompt_echo(tokenizer.decode(ids, skip_special_tokens=True).strip()) for ids in gen]


# ─────────────────────────────────────── Main

def main():
    ap = argparse.ArgumentParser(description="Zero-shot evaluation of Qwen3-30B 4-bit model (no LoRA)")
    ap.add_argument("--base_model", default="unsloth/Qwen3-30B-A3B-Base-bnb-4bit", help="4-bit model to evaluate")
    ap.add_argument("--test_csv", default="test.csv")
    ap.add_argument("--results_csv", default="zeroshot_results.csv")
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--length_penalty", type=float, default=1.2)
    ap.add_argument("--do_sample", type=lambda s: s.lower() == "true", default=False)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--no_repeat_ngram_size", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Load base model only (no LoRA)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        load_in_4bit=True,
        trust_remote_code=True,
        max_seq_length=2048,
        fast_inference=False,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    model.generation_config.cache_implementation = "dynamic"
    model.eval()

    # Data & metrics
    dataset = load_dataset(args.test_csv)
    rouge = evaluate.load("rouge")
    bert = evaluate.load("bertscore")

    preds, refs = [], []
    for i in tqdm(range(0, len(dataset), args.batch_size), desc="Generating"):
        batch = dataset[i : i + args.batch_size]
        prompt_batch = format_prompts(tokenizer, batch["prompt"])
        preds.extend(
            generate(
                model,
                tokenizer,
                prompt_batch,
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

    # Metrics
    rouge_indiv = rouge.compute(predictions=preds, references=refs, use_stemmer=True, use_aggregator=False)
    rouge_agg = {k: float(np.mean(v)) for k, v in rouge_indiv.items()}

    bert_scores = bert.compute(predictions=preds, references=refs, lang="en")
    bert_agg = {k: float(np.mean(bert_scores[k])) for k in ("precision", "recall", "f1")}

    # Save CSV
    pd.DataFrame({
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
    }).to_csv(args.results_csv, index=False)
    print(f"Per-example results saved to {args.results_csv}")

    # Aggregate
    print("\n=== Aggregate ROUGE ===")
    for k, v in rouge_agg.items():
        print(f"{k}: {v:.4f}")
    print("\n=== Aggregate BERTScore ===")
    for k, v in bert_agg.items():
        print(f"{k.title()}: {v:.4f}")
    print()


if __name__ == "__main__":
    main()
