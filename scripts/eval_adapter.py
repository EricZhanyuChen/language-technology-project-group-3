"""
Evaluate a LoRA adapter on the edu test/dev set.

Loads a base model + LoRA adapter, generates predictions for each sample,
and computes per-class and macro-averaged F1.

Usage:
    python scripts/eval_adapter.py \
        --model google/gemma-4-31B-it \
        --adapter_path lora_results/gemma4-31b-cot \
        --split test

The model generates text autoregressively. For CoT-trained models, the output
includes thinking tokens followed by the label. `parse_label` extracts the
fallacy name from the full generated response.
"""

import os
import re
import argparse
import warnings

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LABELS = [
    "ad hominem", "ad populum", "appeal to emotion", "circular reasoning",
    "equivocation", "fallacy of credibility", "fallacy of extension",
    "fallacy of logic", "fallacy of relevance", "false causality",
    "false dilemma", "faulty generalization", "intentional",
]

SYSTEM = """Classify the logical fallacy in the given text.
ad hominem: attacks the speaker instead of the argument.
ad populum: believes something is true because many people believe it.
appeal to emotion: uses emotion to manipulate instead of logic.
circular reasoning: uses the conclusion as its own premise.
equivocation: uses ambiguous language or shifts word meanings.
fallacy of credibility: appeals to authority, tradition, or novelty.
fallacy of extension: distorts an argument to make it easier to attack.
fallacy of logic: has a structural flaw in reasoning.
fallacy of relevance: diverts attention with irrelevant points.
false causality: assumes correlation proves causation.
false dilemma: presents only two extreme options.
faulty generalization: generalizes from insufficient evidence.
intentional: uses tactics to win an argument without evidence.

Answer with only the fallacy name from the list above. Do not explain."""


def load_model(model_name, adapter_path):
    """Load a base model in bf16 and attach a LoRA adapter."""
    print(f"Loading base model {model_name} in bf16 ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"Loading adapter from {adapter_path} ...", flush=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def parse_label(response, is_cot=False):
    """Extract a fallacy label from model-generated text.

    For CoT outputs, the response contains thinking/reasoning text followed by
    the actual label. We strip the thinking part first, then match.

    Strategy:
      1. If is_cot, strip everything before the last double-newline (the label
         appears after the thinking block ends).
      2. Find all exact word-boundary matches in the label portion.
      3. Return the *first* match (most confident).
      4. Fallback: search the full response (last match).

    Returns None if no label can be identified.
    """
    response_lower = response.lower().strip()

    # For CoT: try to isolate the label after thinking ends
    search_text = response_lower
    if is_cot:
        # Thinking ends with </think> or </analysis> or double-newline pattern
        # Strategy: find the last occurrence of common thinking-end markers
        for marker in ["</think>", "</analysis>", "</channel>"]:
            idx = response_lower.rfind(marker)
            if idx != -1:
                search_text = response_lower[idx + len(marker):]
                break
        else:
            # No marker found — try splitting on last double-newline
            parts = response_lower.strip().rsplit("\n\n", 1)
            if len(parts) == 2:
                search_text = parts[1]

    # Pass 1: word-boundary match in the label portion
    sorted_labels = sorted(LABELS, key=len, reverse=True)
    for lbl in sorted_labels:
        if re.search(r"\b" + re.escape(lbl) + r"\b", search_text):
            return lbl

    # Pass 2: prefix match on first token of label portion
    first = search_text.split()[0] if search_text.split() else ""
    for lbl in LABELS:
        if lbl.startswith(first) or first.startswith(lbl):
            return lbl

    # Pass 3: fallback — search full response, take last match
    matches = []
    for lbl in sorted_labels:
        for m in re.finditer(r"\b" + re.escape(lbl) + r"\b", response_lower):
            matches.append((m.start(), m.end(), lbl))
    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[-1][2]

    return None


def evaluate(model, tokenizer, df, max_new=512, max_length=512, enable_thinking=False, output_path=None):
    """Run inference on every row in df and collect predictions.

    Each result is written immediately to a JSONL file so partial results
    are available even if the job is cancelled.
    """
    import json as _json

    pred_labels, true_labels, responses = [], [], []
    jsonl_path = output_path.replace(".csv", ".jsonl") if output_path else None

    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Eval")):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'Text: "{row["source_article"]}"'},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        pred = parse_label(response, is_cot=enable_thinking)
        pred_labels.append(pred)
        true_labels.append(row["updated_label"])
        responses.append(response)

        # Write each result immediately to JSONL
        if jsonl_path:
            with open(jsonl_path, "a") as f:
                _json.dump({
                    "idx": i,
                    "text": row["source_article"],
                    "true": row["updated_label"],
                    "pred": pred,
                    "response_len": len(response),
                    "response": response,
                }, f, ensure_ascii=False)
                f.write("\n")

    return pred_labels, true_labels, responses


def print_report(true_labels, pred_labels):
    """Print per-class P/R/F1 and return the sklearn report dict."""
    results_df = pd.DataFrame({"true": true_labels, "pred": pred_labels})
    valid = results_df.dropna(subset=["pred", "true"])

    report = classification_report(
        valid["true"], valid["pred"],
        labels=LABELS, target_names=LABELS, zero_division=0, output_dict=True,
    )

    for label in LABELS:
        info = report.get(label, {})
        print(
            f"  {label:25s}  "
            f"P={info.get('precision', 0):.3f}  "
            f"R={info.get('recall', 0):.3f}  "
            f"F1={info.get('f1-score', 0):.3f}"
        )
    print(f"\n  {'MACRO AVG':25s}  F1={report['macro avg']['f1-score']:.4f}")
    print(f"  {'ACCURACY':25s}  {report['accuracy']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a LoRA adapter on edu test/dev set"
    )
    parser.add_argument("--model", type=str, required=True, help="Base HF model name")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--data_dir", type=str, default="logic_data")
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev", "train"])
    parser.add_argument("--dataset", type=str, default="edu", choices=["edu", "climate"],
                        help="Dataset to evaluate on (edu or climate)")
    parser.add_argument("--output_dir", type=str, default="results/lora")
    parser.add_argument("--max_new", type=int, default=1024,
                        help="Max tokens to generate (must cover CoT + label)")
    parser.add_argument("--max_length", type=int, default=1024,
                        help="Max input length (prompt) in tokens")
    parser.add_argument("--enable_thinking", action="store_true",
                        help="Enable thinking tokens for CoT-trained models")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapter_path)

    if args.dataset == "climate":
        data_path = f"{args.data_dir}/climate_{args.split}.csv"
        df = pd.read_csv(data_path, usecols=["source_article", "logical_fallacies"])
        df = df.rename(columns={"logical_fallacies": "updated_label"})
    else:
        data_path = f"{args.data_dir}/edu_{args.split}.csv"
        df = pd.read_csv(data_path, usecols=["source_article", "updated_label"])
    print(f"Eval dataset={args.dataset} split={args.split} ({len(df)} samples)", flush=True)

    # Save per-sample results
    adapter_name = args.adapter_path.rstrip("/").split("/")[-1]
    results_path = f"{args.output_dir}/{adapter_name}_{args.dataset}_{args.split}_results.csv"
    pred_labels, true_labels, responses = evaluate(
        model, tokenizer, df, args.max_new, args.max_length,
        enable_thinking=args.enable_thinking,
        output_path=results_path,
    )

    results_df = pd.DataFrame({
        "text": df["source_article"],
        "true": true_labels,
        "pred": pred_labels,
        "correct": [p == t for p, t in zip(pred_labels, true_labels)],
        "response": responses,
    })
    os.makedirs(args.output_dir, exist_ok=True)
    results_df.to_csv(results_path, index=False)

    report = print_report(true_labels, pred_labels)
    print(f"\nResults saved to {results_path}")

    # Append to summary file
    summary_path = f"{args.output_dir}/summary.csv"
    summary_row = pd.DataFrame([{
        "adapter": adapter_name,
        "dataset": args.dataset,
        "split": args.split,
        "macro_f1": report["macro avg"]["f1-score"],
        "accuracy": report["accuracy"],
        "n_samples": len(true_labels),
        "n_null": sum(1 for p in pred_labels if p is None),
        "results_file": results_path,
    }])
    if os.path.exists(summary_path):
        existing = pd.read_csv(summary_path)
        summary_row = pd.concat([existing, summary_row], ignore_index=True)
    summary_row.to_csv(summary_path, index=False)
    print(f"Summary appended to {summary_path}")


if __name__ == "__main__":
    main()
