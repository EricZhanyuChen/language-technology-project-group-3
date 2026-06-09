"""Parse climate OOD results and append to summary."""
import argparse
import json
import os
import re
import pandas as pd
from sklearn.metrics import classification_report, f1_score

LABELS = [
    "ad hominem", "ad populum", "appeal to emotion", "circular reasoning",
    "equivocation", "fallacy of credibility", "fallacy of extension",
    "fallacy of logic", "fallacy of relevance", "false causality",
    "false dilemma", "faulty generalization", "intentional",
]

def parse_label(text):
    text = text.lower().strip()
    for lbl in sorted(LABELS, key=len, reverse=True):
        if re.search(r"\b" + re.escape(lbl) + r"\b", text):
            return lbl
    return "__unknown__"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path")
    parser.add_argument("--append-summary", action="store_true")
    args = parser.parse_args()

    rows = []
    with open(args.jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))

    true_labels, pred_labels = [], []
    for r in rows:
        true_labels.append(parse_label(r.get("label", "")))
        pred_labels.append(parse_label(r.get("predict", "")))

    report = classification_report(
        true_labels, pred_labels,
        labels=LABELS, target_names=LABELS, zero_division=0, output_dict=True,
    )

    for label in LABELS:
        info = report.get(label, {})
        print(f"  {label:25s}  F1={info.get('f1-score', 0):.3f}")

    macro_f1 = f1_score(true_labels, pred_labels, labels=LABELS, average="macro", zero_division=0)
    acc = report["accuracy"]
    n_null = sum(1 for p in pred_labels if p == "__unknown__")
    print(f"\n  {'MACRO AVG':25s}  F1={macro_f1:.4f}")
    print(f"  {'ACCURACY':25s}  {acc:.4f}")

    if args.append_summary:
        summary_path = "results/lora/summary.csv"
        adapter_name = os.path.basename(os.path.dirname(args.jsonl_path))
        row = pd.DataFrame([{
            "adapter": adapter_name,
            "dataset": "climate",
            "split": "test",
            "macro_f1": macro_f1,
            "accuracy": acc,
            "n_samples": len(true_labels),
            "n_null": n_null,
            "results_file": args.jsonl_path,
        }])
        if os.path.exists(summary_path):
            existing = pd.read_csv(summary_path)
            row = pd.concat([existing, row], ignore_index=True)
        row.to_csv(summary_path, index=False)
        print(f"Summary appended to {summary_path}")

if __name__ == "__main__":
    main()
