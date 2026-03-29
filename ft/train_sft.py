"""LoRA SFT on JSONL from prepare_dataset.py."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def _lora_targets(model_id: str) -> list[str] | None:
    lower = model_id.lower()
    if any(x in lower for x in ("qwen", "llama", "mistral", "phi")):
        return [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train-file", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--fourbit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    ft_dir = Path(__file__).resolve().parent
    train_file = args.train_file or (ft_dir / "data" / "train.jsonl")
    if not train_file.is_file():
        print(f"Missing {train_file}. Run prepare_dataset.py first.", file=sys.stderr)
        sys.exit(1)

    model_stem = args.model.replace("/", "_")
    output_dir = args.output_dir or (ft_dir / "runs" / model_stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files=str(train_file), split="train")

    def to_text(batch: dict) -> dict:
        texts = [
            tokenizer.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False,
            )
            for m in batch["messages"]
        ]
        return {"text": texts}

    dataset = dataset.map(to_text, batched=True, remove_columns=dataset.column_names)

    quant_config = None
    if args.fourbit and torch.cuda.is_available():
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=dtype if torch.cuda.is_available() else torch.float32,
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=_lora_targets(args.model),
        ),
    )

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        logging_steps=1,
        save_strategy="no",
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=True,
        report_to="none",
        max_length=args.max_seq_length,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved adapter to {output_dir}")


if __name__ == "__main__":
    main()
