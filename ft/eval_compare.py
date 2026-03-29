"""Compare base vs LoRA: python ft/eval_compare.py --adapter ft/runs/..."""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_PROMPTS = [
    "Summarize what the Dama2 app does in three short bullet points.",
    "How would you add conversation memory to a RAG app?",
]


def _adapter_base_id(adapter_dir: Path) -> str:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing {cfg_path}")
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    base = data.get("base_model_name_or_path")
    if not base:
        raise SystemExit("adapter_config.json has no base_model_name_or_path")
    return str(base)


def _device_for(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


@torch.inference_mode()
def generate_reply(
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    user_text: str,
    max_new_tokens: int,
) -> str:
    messages = [{"role": "user", "content": user_text}]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    dev = _device_for(model)
    inputs = {k: v.to(dev) for k, v in inputs.items()}
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=pad_id,
    )
    new_tokens = out[0, inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _load_tokenizer(adapter_dir: Path, base_id: str, trust_remote_code: bool) -> AutoTokenizer:
    tok_dir = adapter_dir if (adapter_dir / "tokenizer_config.json").is_file() else base_id
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def _load_base(base_id: str, trust_remote_code: bool) -> AutoModelForCausalLM:
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kw: dict = {
        "trust_remote_code": trust_remote_code,
        "torch_dtype": dtype if torch.cuda.is_available() else torch.float32,
    }
    if torch.cuda.is_available():
        kw["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(base_id, **kw)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--base-model", default=None)
    p.add_argument("--prompts-file", type=Path, default=None)
    p.add_argument("--prompt", action="append", default=[])
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--trust-remote-code", action="store_true", default=True)
    args = p.parse_args()

    adapter_dir = args.adapter.resolve()
    if not adapter_dir.is_dir():
        raise SystemExit(f"Not a directory: {adapter_dir}")

    base_id = args.base_model or _adapter_base_id(adapter_dir)
    tokenizer = _load_tokenizer(adapter_dir, base_id, args.trust_remote_code)

    prompts: list[str] = []
    if args.prompts_file:
        text = args.prompts_file.read_text(encoding="utf-8")
        prompts.extend(line.strip() for line in text.splitlines() if line.strip())
    prompts.extend(args.prompt)
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)

    print(f"Base: {base_id}\nAdapter: {adapter_dir}\n", flush=True)
    failures = 0
    for i, user_text in enumerate(prompts, 1):
        print("=" * 72, f"\n[{i}/{len(prompts)}] USER:\n{user_text}\n", flush=True)

        base = _load_base(base_id, args.trust_remote_code)
        base.eval()
        try:
            base_out = generate_reply(base, tokenizer, user_text, args.max_new_tokens)
        except Exception as e:
            base_out = f"<error: {e}>"
            failures += 1
        del base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        b2 = _load_base(base_id, args.trust_remote_code)
        lora = PeftModel.from_pretrained(b2, str(adapter_dir))
        lora.eval()
        try:
            lora_out = generate_reply(lora, tokenizer, user_text, args.max_new_tokens)
        except Exception as e:
            lora_out = f"<error: {e}>"
            failures += 1
        del lora
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("--- BASE ---\n", base_out, "\n", sep="", flush=True)
        print("--- LORA ---\n", lora_out, "\n", sep="", flush=True)

    if failures:
        sys.exit(1)
    print("Done.")


if __name__ == "__main__":
    main()
