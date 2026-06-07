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


def parse_label(response):
    """Extract a fallacy label from model-generated text.

    Strategy:
      1. Find all exact word-boundary matches across the response.
      2. Return the *last* match — this handles CoT outputs where the label
         appears both in thinking/reasoning (early) and as the final answer
         (late).  The last match is most likely the model's actual prediction.
      3. Fallback: prefix match on the first token of the response.

    Returns None if no label can be identified.
    """
    response = response.lower().strip()

    # Pass 1: collect all word-boundary matches, sorted by position
    sorted_labels = sorted(LABELS, key=len, reverse=True)
    matches = []
    for lbl in sorted_labels:
        for m in re.finditer(r"\b" + re.escape(lbl) + r"\b", response):
            matches.append((m.start(), m.end(), lbl))

    if matches:
        matches.sort(key=lambda x: x[0])
        return matches[-1][2]

    # Pass 2: prefix match on first token
    first = response.split()[0] if response.split() else ""
    for lbl in LABELS:
        if lbl.startswith(first) or first.startswith(lbl):
            return lbl

    return None


def evaluate(model, tokenizer, df, max_new=512, max_length=512):
    """Run inference on every row in df and collect predictions.

    For CoT-trained models the output typically looks like:
        <thinking tokens> ... reasoning ... <end thinking>
        <actual label>
    `skip_special_tokens=True` strips the thinking tags but keeps the
    reasoning text, so `parse_label` searches the full decoded output.
    """
    pred_labels, true_labels = [], []
    for i, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Eval")):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f'Text: "{row["source_article"]}"'},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,  # disable thinking for direct label output
        )
        inputs = tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=max_length
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,           # greedy decoding — deterministic
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode only the newly generated tokens (strip prompt)
        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        pred = parse_label(response)
        pred_labels.append(pred)
        true_labels.append(row["updated_label"])

        # Print first 5 samples for debugging
        if i < 5:
            print(f"\n  [DEBUG {i}] true={row['updated_label']}", flush=True)
            print(f"  [DEBUG {i}] response ({len(response)} chars): {response[:300]}", flush=True)
            print(f"  [DEBUG {i}] parsed: {pred}", flush=True)

    return pred_labels, true_labels


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
    parser.add_argument("--output_dir", type=str, default="results/lora")
    parser.add_argument("--max_new", type=int, default=512,
                        help="Max tokens to generate (must cover CoT + label)")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max input length (prompt) in tokens")
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapter_path)

    data_path = f"{args.data_dir}/edu_{args.split}.csv"
    df = pd.read_csv(data_path, usecols=["source_article", "updated_label"])
    print(f"Eval split: {args.split} ({len(df)} samples)", flush=True)

    pred_labels, true_labels = evaluate(
        model, tokenizer, df, args.max_new, args.max_length
    )

    # Save per-sample results
    adapter_name = args.adapter_path.rstrip("/").split("/")[-1]
    results_df = pd.DataFrame({
        "text": df["source_article"],
        "true": true_labels,
        "pred": pred_labels,
        "correct": [p == t for p, t in zip(pred_labels, true_labels)],
    })
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = f"{args.output_dir}/{adapter_name}_{args.split}_results.csv"
    results_df.to_csv(results_path, index=False)

    print_report(true_labels, pred_labels)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
