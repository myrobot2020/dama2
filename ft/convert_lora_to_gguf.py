"""Optional LoRA -> GGUF adapter using a local llama.cpp checkout.

Ollama accepts either a Safetensors adapter directory (set OLLAMA_ADAPTER_GGUF to
ft/runs/<stem>) or a GGUF file. This script is only needed if you want a .gguf.

Set LLAMA_CPP to the root of a built llama.cpp repo. If unset or conversion
fails, exit 0 unless --strict (pipeline uses output file presence to switch path).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _find_convert_script(llama_root: Path) -> Path | None:
    for rel in (
        Path("examples/convert_legacy_lora.py"),
        Path("convert_lora_to_gguf.py"),
        Path("tools/convert_lora_to_gguf.py"),
    ):
        p = llama_root / rel
        if p.is_file():
            return p
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output .gguf path (default: ft/data/adapter.gguf)",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 if LLAMA_CPP missing or conversion did not write --out",
    )
    args = p.parse_args()

    ft_dir = Path(__file__).resolve().parent
    out = args.out or (ft_dir / "data" / "adapter.gguf")
    adapter_dir = args.adapter_dir.resolve()
    if not adapter_dir.is_dir():
        print(f"Not a directory: {adapter_dir}", file=sys.stderr)
        sys.exit(1 if args.strict else 0)
    cfg = adapter_dir / "adapter_config.json"
    if not cfg.is_file():
        print(f"Missing adapter_config.json in {adapter_dir}", file=sys.stderr)
        sys.exit(1 if args.strict else 0)

    llama = os.environ.get("LLAMA_CPP", "").strip()
    if not llama:
        print(
            "LLAMA_CPP not set; skipping GGUF export. "
            "Point OLLAMA_ADAPTER_GGUF at the Safetensors adapter directory instead.",
            file=sys.stderr,
        )
        sys.exit(1 if args.strict else 0)

    root = Path(llama).resolve()
    script = _find_convert_script(root)
    if script is None:
        print(
            f"No known convert script under {root}. "
            "Install llama.cpp or update _find_convert_script in convert_lora_to_gguf.py.",
            file=sys.stderr,
        )
        sys.exit(1 if args.strict else 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    # llama.cpp CLIs vary by version; try a generic python invocation with --help-only skip
    cmd = [sys.executable, str(script), str(adapter_dir), str(out)]
    print("Running:", " ".join(cmd), file=sys.stderr)
    r = subprocess.run(cmd, cwd=str(root), check=False)
    if r.returncode != 0 or not out.is_file():
        print(
            "GGUF conversion failed or produced no file; "
            "use Safetensors ADAPTER directory with Ollama.",
            file=sys.stderr,
        )
        sys.exit(1 if args.strict else 0)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
