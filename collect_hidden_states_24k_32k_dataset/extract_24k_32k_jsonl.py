#!/usr/bin/env python3
"""Extract raw conversations with token length in [24k, 32k)."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--min-length", type=int, default=24576)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    total = selected = 0
    batch: list[tuple[str, list[dict[str, object]]]] = []

    with args.input.open("r", encoding="utf-8") as source, tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, delete=False
    ) as destination:
        temporary = Path(destination.name)

        def consume() -> None:
            nonlocal selected
            if not batch:
                return
            conversations = [row[1] for row in batch]
            encoded = tokenizer.apply_chat_template(
                conversations, tokenize=True, add_generation_prompt=False, padding=False
            )
            lengths = [len(ids) for ids in encoded["input_ids"]]
            for (raw, _), length in zip(batch, lengths, strict=True):
                if args.min_length <= length < args.max_length:
                    destination.write(raw if raw.endswith("\n") else raw + "\n")
                    selected += 1
            batch.clear()

        for raw in source:
            if not raw.strip():
                continue
            record = json.loads(raw)
            conversations = record.get("conversations")
            if not isinstance(conversations, list):
                raise ValueError("Every nonblank row must contain a conversations list")
            batch.append((raw, conversations))
            total += 1
            if len(batch) == args.batch_size:
                consume()
            if total % 1000 == 0:
                print(f"scanned={total:,} selected={selected:,}", flush=True)
        consume()
        destination.flush()
        os.fsync(destination.fileno())

    os.replace(temporary, args.output)
    print(f"Wrote {selected:,} of {total:,} rows to {args.output}.")


if __name__ == "__main__":
    main()
