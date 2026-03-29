#!/usr/bin/env python3
"""Export training data from evolving-memory SQLite DB as JSONL.

Reads the SQLite database directly (no server needed) and produces
training_data.jsonl suitable for SFT fine-tuning of Qwen3-VL on the
Cognitive ISA (TOOLCALL format).

Usage:
    python scripts/export_training_data.py --db memory.db --output training_data.jsonl
    python scripts/export_training_data.py --db memory.db --output-dir ./data/
"""

from __future__ import annotations

import argparse
import json
import random
import sqlite3
import sys
from collections import Counter
from pathlib import Path

# System prompt matching bytecode_compiler.ts TEXT_SCENE_SYSTEM_PROMPT
SYSTEM_PROMPT = (
    "You are a robot motor controller. You see a text description of what "
    "the robot's camera sees and output exactly ONE motor command.\n\n"
    "ACTIONS:\n"
    "- move_forward(speed_l, speed_r) — Speed 0-255. Equal = straight.\n"
    "- move_backward(speed_l, speed_r)\n"
    "- turn_left(speed_l, speed_r) — speed_l < speed_r\n"
    "- turn_right(speed_l, speed_r) — speed_l > speed_r\n"
    "- rotate_cw(degrees, speed) — Clockwise 0-180deg\n"
    "- rotate_ccw(degrees, speed) — Counter-clockwise 0-180deg\n"
    "- stop() — ONLY when target < 20cm\n\n"
    "Output format: TOOLCALL:{\"name\":\"<action>\",\"args\":{...}}\n"
    "Output ONLY the TOOLCALL line. No explanation."
)


def extract_action_name(payload: str) -> str | None:
    """Extract the action name from a TOOLCALL payload."""
    if not payload.startswith("TOOLCALL:"):
        return None
    try:
        data = json.loads(payload[len("TOOLCALL:"):])
        return data.get("name")
    except (json.JSONDecodeError, KeyError):
        return None


def is_duplicate_frame(a: dict, b: dict) -> bool:
    """Check if two examples have identical assistant content."""
    return a["messages"][2]["content"] == b["messages"][2]["content"]


def load_examples(db_path: str, outcome: str = "success", min_actions: int = 3) -> list[dict]:
    """Load training examples from SQLite database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get qualifying traces
    rows = conn.execute(
        """
        SELECT t.trace_id, t.goal
        FROM trace_entries t
        JOIN (
            SELECT trace_id, COUNT(*) as cnt
            FROM action_entries GROUP BY trace_id
        ) ac ON ac.trace_id = t.trace_id
        WHERE t.outcome = ? AND ac.cnt >= ?
        ORDER BY t.created_at
        """,
        (outcome, min_actions),
    ).fetchall()

    examples = []
    for row in rows:
        actions = conn.execute(
            "SELECT reasoning, action_payload FROM action_entries "
            "WHERE trace_id = ? ORDER BY rowid",
            (row["trace_id"],),
        ).fetchall()

        for action in actions:
            reasoning = (action["reasoning"] or "").strip()
            payload = (action["action_payload"] or "").strip()
            if not reasoning or not payload:
                continue
            if not payload.startswith("TOOLCALL:"):
                continue

            examples.append({
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": reasoning},
                    {"role": "assistant", "content": payload},
                ]
            })

    conn.close()
    return examples


def deduplicate_consecutive(examples: list[dict], keep_every: int = 3) -> list[dict]:
    """Remove consecutive duplicate frames, keeping every Nth."""
    if not examples:
        return examples

    result = []
    streak = 0
    for i, ex in enumerate(examples):
        if i > 0 and is_duplicate_frame(ex, examples[i - 1]):
            streak += 1
            if streak % keep_every == 0:
                result.append(ex)
        else:
            streak = 0
            result.append(ex)

    return result


def balance_actions(examples: list[dict], max_ratio: float = 0.6) -> list[dict]:
    """Undersample overrepresented actions to ensure balance."""
    # Count action distribution
    action_counts: Counter[str] = Counter()
    action_buckets: dict[str, list[dict]] = {}

    for ex in examples:
        name = extract_action_name(ex["messages"][2]["content"])
        if name is None:
            name = "__unknown__"
        action_counts[name] += 1
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


def print_stats(examples: list[dict], label: str = "Dataset") -> None:
    """Print dataset statistics."""
    action_counts: Counter[str] = Counter()
    scene_lengths: list[int] = []

    for ex in examples:
        name = extract_action_name(ex["messages"][2]["content"]) or "unknown"
        action_counts[name] += 1
        scene_lengths.append(len(ex["messages"][1]["content"]))

    print(f"\n=== {label} Stats ===")
    print(f"Total examples: {len(examples)}")

    if scene_lengths:
        avg_len = sum(scene_lengths) / len(scene_lengths)
        print(f"Avg scene length: {avg_len:.0f} chars (~{avg_len / 4:.0f} tokens)")

    print("\nAction distribution:")
    for name, count in action_counts.most_common():
        pct = count / len(examples) * 100 if examples else 0
        print(f"  {name}: {count} ({pct:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Export training data from evolving-memory DB")
    parser.add_argument("--db", required=True, help="Path to SQLite database")
    parser.add_argument("--output", default=None, help="Output JSONL file path")
    parser.add_argument("--output-dir", default=None, help="Output directory for train/val split")
    parser.add_argument("--outcome", default="success", help="Filter by outcome (default: success)")
    parser.add_argument("--min-actions", type=int, default=3, help="Min actions per trace")
    parser.add_argument("--max-ratio", type=float, default=0.6, help="Max ratio for any single action")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-dedup", action="store_true", help="Skip deduplication")
    parser.add_argument("--no-balance", action="store_true", help="Skip action balancing")
    args = parser.parse_args()

    random.seed(args.seed)

    if not Path(args.db).exists():
        print(f"Error: database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    # Load
    print(f"Loading examples from {args.db} (outcome={args.outcome}, min_actions={args.min_actions})...")
    examples = load_examples(args.db, outcome=args.outcome, min_actions=args.min_actions)
    print(f"Loaded {len(examples)} raw examples")

    if not examples:
        print("No examples found. Check filters and database content.")
        sys.exit(0)

    print_stats(examples, "Raw")

    # Deduplicate consecutive identical frames
    if not args.no_dedup:
        before = len(examples)
        examples = deduplicate_consecutive(examples, keep_every=3)
        print(f"\nDedup: {before} -> {len(examples)} examples")

    # Balance action distribution
    if not args.no_balance:
        before = len(examples)
        examples = balance_actions(examples, max_ratio=args.max_ratio)
        print(f"Balance: {before} -> {len(examples)} examples")

    print_stats(examples, "Processed")

    # Split into train/val
    random.shuffle(examples)
    val_size = max(1, int(len(examples) * args.val_split))
    val_examples = examples[:val_size]
    train_examples = examples[val_size:]

    print(f"\nSplit: {len(train_examples)} train, {len(val_examples)} val")

    # Write output
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        train_path = out_dir / "train.jsonl"
        val_path = out_dir / "val.jsonl"
    elif args.output:
        base = Path(args.output)
        train_path = base.parent / f"train_{base.name}"
        val_path = base.parent / f"val_{base.name}"
    else:
        train_path = Path("train.jsonl")
        val_path = Path("val.jsonl")

    for path, data in [(train_path, train_examples), (val_path, val_examples)]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"Wrote {len(data)} examples to {path}")

    # Stats summary
    stats = {
        "total_raw": len(examples) + (len(examples) - len(train_examples) - len(val_examples)),
        "total_processed": len(train_examples) + len(val_examples),
        "train": len(train_examples),
        "val": len(val_examples),
        "outcome_filter": args.outcome,
        "min_actions": args.min_actions,
    }
    stats_path = (Path(args.output_dir) if args.output_dir else Path(".")) / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
