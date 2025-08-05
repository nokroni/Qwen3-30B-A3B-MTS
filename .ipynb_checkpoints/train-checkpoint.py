

import argparse
import os
from datasets import Dataset
import pandas as pd
import torch
from unsloth import FastModel, FastLanguageModel
from trl import SFTTrainer, SFTConfig

def load_csv_dataset(path):
    """Load a CSV with `prompt` and `completion` columns and return HF Dataset."""
    df = pd.read_csv(path)
    # Build a single ChatML-formatted text field expected by Unsloth
    df["text"] = df.apply(
        lambda row: (
            f"<|im_start|>user\nSummarize the following text:\n{row['prompt']}\n<|im_end|>\n"
            f"<|im_start|>assistant\n{row['completion']}<|im_end|>"
        ),
        axis=1,
    )
    return Dataset.from_pandas(df[["text"]])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--val_csv", required=True)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=12)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    args = parser.parse_args()

    # 1. Prepare dataset
    train_ds = load_csv_dataset(args.train_csv)
    val_ds   = load_csv_dataset(args.val_csv)

    # 2. Load 30B‑A3B MoE model in 4‑bit
    model_name = "unsloth/Qwen3-30B-A3B-Base-bnb-4bit"  # or your HF hub path
    model, tokenizer = FastModel.from_pretrained(
    model_name = model_name,
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

    # 3. Attach fast LoRA adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 32,
        lora_dropout = 0.05,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 42,
        max_seq_length = args.max_seq_len,
        use_rslora = False,
        loftq_config = None,
    )

    # 4. Configure trainer
    trainer = SFTTrainer(
        model = model,
        train_dataset = train_ds,
        eval_dataset  = val_ds,
        processing_class = tokenizer,
        args = SFTConfig(
            per_device_train_batch_size = args.batch_size,
            gradient_accumulation_steps = args.grad_accum,
            warmup_steps = 10,
            max_steps = args.max_steps,
            logging_steps = 1,
            output_dir = args.output_dir,
            optim = "adamw_8bit",
            seed = 42,
            max_length = args.max_seq_len,
        ),
    )

    # 5. Train
    trainer.train()

    # 6. Save adapter + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Training complete. Artifacts saved to {args.output_dir}")

if __name__ == "__main__":
    main()
