#!/usr/bin/env python3
"""Prepare SFT dataset from exported training data JSONL.

Post-processing pipeline:
  1. Deduplicate consecutive identical frames (keep every 3rd)
  2. Balance action distribution (undersample MOVE_FORWARD if >60%)
  3. Augment with negative examples from failed traces (optional)
  4. Split into train.jsonl + val.jsonl (90/10)

Usage:
    python scripts/prepare_sft_dataset.py --input training_data.jsonl
    python scripts/prepare_sft_dataset.py --input training_data.jsonl --output-dir ./sft_data/
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path


def extract_action_name(payload: str) -> str | None:
    """Extract action name from TOOLCALL payload."""
    if not payload.startswith("TOOLCALL:"):
        return None
    try:
        data = json.loads(payload[len("TOOLCALL:"):])
        return data.get("name")
    except (json.JSONDecodeError, KeyError):
        return None


def load_jsonl(path: str) -> list[dict]:
    """Load JSONL file into list of dicts."""
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
    return examples


def deduplicate_consecutive(examples: list[dict], keep_every: int = 3) -> list[dict]:
    """Remove consecutive examples with identical assistant content, keeping every Nth."""
    if not examples:
        return examples

    result = []
    streak = 0
    for i, ex in enumerate(examples):
        assistant_content = ex["messages"][2]["content"]
        if i > 0 and assistant_content == examples[i - 1]["messages"][2]["content"]:
            streak += 1
            if streak % keep_every == 0:
                result.append(ex)
        else:
            streak = 0
            result.append(ex)

    return result


def balance_actions(examples: list[dict], max_ratio: float = 0.6) -> list[dict]:
    """Undersample overrepresented actions."""
    action_buckets: dict[str, list[dict]] = {}

    for ex in examples:
        name = extract_action_name(ex["messages"][2]["content"]) or "__unknown__"
        action_buckets.setdefault(name, []).append(ex)

    total = len(examples)
    max_per_action = int(total * max_ratio)

    result = []
    for name, bucket in action_buckets.items():
        if len(bucket) > max_per_action:
            random.shuffle(bucket)
            result.extend(bucket[:max_per_action])
        else:
            result.extend(bucket)

    random.shuffle(result)
    return result


def validate_example(ex: dict) -> bool:
    """Check that an example has valid structure."""
    if "messages" not in ex:
        return False
    msgs = ex["messages"]
    if len(msgs) != 3:
        return False
    if msgs[0]["role"] != "system" or msgs[1]["role"] != "user" or msgs[2]["role"] != "assistant":
        return False
    # Must have TOOLCALL in assistant response
    if not msgs[2]["content"].startswith("TOOLCALL:"):
        return False
    # Must have non-empty user content
    if len(msgs[1]["content"].strip()) < 10:
        return False
    return True


def print_stats(examples: list[dict], label: str) -> dict:
    """Print and return dataset statistics."""
    action_counts: Counter[str] = Counter()
    scene_lengths: list[int] = []

    for ex in examples:
        name = extract_action_name(ex["messages"][2]["content"]) or "unknown"
        action_counts[name] += 1
        scene_lengths.append(len(ex["messages"][1]["content"]))

    avg_len = sum(scene_lengths) / len(scene_lengths) if scene_lengths else 0
    avg_tokens = avg_len / 4  # rough estimate

    print(f"\n=== {label} ===")
    print(f"Total examples: {len(examples)}")
    print(f"Avg scene length: {avg_len:.0f} chars (~{avg_tokens:.0f} tokens)")
    print(f"Action distribution:")
    for name, count in action_counts.most_common():
        pct = count / len(examples) * 100 if examples else 0
        print(f"  {name}: {count} ({pct:.1f}%)")

    return {
        "total": len(examples),
        "avg_scene_chars": round(avg_len),
        "avg_scene_tokens_est": round(avg_tokens),
        "action_distribution": dict(action_counts.most_common()),
    }


def main():
    parser = argparse.ArgumentParser(description="Prepare SFT dataset from training data JSONL")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output-dir", default="./sft_data", help="Output directory")
    parser.add_argument("--max-ratio", type=float, default=0.6, help="Max ratio for any action")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--no-balance", action="store_true", help="Skip action balancing")
    args = parser.parse_args()

    random.seed(args.seed)

    if not Path(args.input).exists():
        print(f"Error: input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Load
    print(f"Loading {args.input}...")
    examples = load_jsonl(args.input)
    print(f"Loaded {len(examples)} examples")

    # Validate
    valid = [ex for ex in examples if validate_example(ex)]
    if len(valid) < len(examples):
        print(f"Filtered {len(examples) - len(valid)} invalid examples")
    examples = valid

    if not examples:
        print("No valid examples found.")
        sys.exit(0)

    raw_stats = print_stats(examples, "Raw Input")

    # Deduplicate
    if not args.no_dedup:
        before = len(examples)
        examples = deduplicate_consecutive(examples)
        print(f"\nDedup: {before} -> {len(examples)}")

    # Balance
    if not args.no_balance:
        before = len(examples)
        examples = balance_actions(examples, max_ratio=args.max_ratio)
        print(f"Balance: {before} -> {len(examples)}")

    processed_stats = print_stats(examples, "Processed")

    # Split
    random.shuffle(examples)
    val_size = max(1, int(len(examples) * args.val_split))
    val_data = examples[:val_size]
    train_data = examples[val_size:]

    print(f"\nSplit: {len(train_data)} train, {len(val_data)} val")

    # Write
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train.jsonl", train_data), ("val.jsonl", val_data)]:
        path = out_dir / name
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} examples to {path}")

    # Stats
    stats = {
        "raw": raw_stats,
        "processed": processed_stats,
        "train_size": len(train_data),
        "val_size": len(val_data),
        "settings": {
            "max_ratio": args.max_ratio,
            "val_split": args.val_split,
            "dedup": not args.no_dedup,
            "balance": not args.no_balance,
        },
    }
    stats_path = out_dir / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
