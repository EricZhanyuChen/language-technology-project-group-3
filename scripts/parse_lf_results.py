"""
Parse LLaMA Factory generated_predictions.jsonl and compute macro-F1.

Usage:
    python scripts/parse_lf_results.py lora_results/gemma4-31b-aug/generated_predictions.jsonl
    python scripts/parse_lf_results.py lora_results/gemma4-31b-aug/generated_predictions.jsonl --append-summary
"""

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
    sorted_labels = sorted(LABELS, key=len, reverse=True)
    for lbl in sorted_labels:
        if re.search(r"\b" + re.escape(lbl) + r"\b", text):
            return lbl
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="Path to generated_predictions.jsonl")
    parser.add_argument("--append-summary", action="store_true",
                        help="Append results to results/lora/summary.csv")
    args = parser.parse_args()

    rows = []
    with open(args.jsonl_path) as f:
        for line in f:
            rows.append(json.loads(line))

    true_labels, pred_labels, responses = [], [], []
    for r in rows:
        label_text = r.get("label", "")
        predict_text = r.get("predict", "")
        true_lbl = parse_label(label_text)
        pred_lbl = parse_label(predict_text)
        true_labels.append(true_lbl if true_lbl is not None else "__unknown__")
        pred_labels.append(pred_lbl if pred_lbl is not None else "__unknown__")
        responses.append(predict_text)

    eval_labels = LABELS + ["__unknown__"]
    report = classification_report(
        true_labels, pred_labels,
        labels=eval_labels, target_names=eval_labels, zero_division=0, output_dict=True,
    )

    for label in LABELS:
        info = report.get(label, {})
        print(
            f"  {label:25s}  "
            f"P={info.get('precision', 0):.3f}  "
            f"R={info.get('recall', 0):.3f}  "
            f"F1={info.get('f1-score', 0):.3f}"
        )
    macro_f1 = f1_score(true_labels, pred_labels, labels=LABELS, average="macro", zero_division=0)
    acc = report["accuracy"]
    n_null = sum(1 for p in pred_labels if p == "__unknown__")
    print(f"\n  {'MACRO AVG':25s}  F1={macro_f1:.4f}")
    print(f"  {'ACCURACY':25s}  {acc:.4f}")
    print(f"  {'NULL PREDS':25s}  {n_null}")

    if args.append_summary:
        summary_path = "results/lora/summary.csv"
        adapter_name = os.path.basename(os.path.dirname(args.jsonl_path))
        row = pd.DataFrame([{
            "adapter": adapter_name,
            "dataset": "edu",
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
