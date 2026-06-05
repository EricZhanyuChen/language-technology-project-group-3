import os, re, argparse, warnings
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
intentional: uses tactics to win an argument without evidence."""


def load_model(model_name, adapter_path):
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
    response = response.lower().strip()
    for lbl in LABELS:
        if re.search(r"\b" + re.escape(lbl) + r"\b", response):
            return lbl
    first = response.split()[0] if response.split() else ""
    for lbl in LABELS:
        if lbl.startswith(first) or first.startswith(lbl):
            return lbl
    return None


def evaluate(model, tokenizer, df, max_new=16, max_length=512):
    pred_labels, true_labels = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Eval"):
        messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"Text: {row['source_article']}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=max_new,
                do_sample=False, temperature=0.1,
                pad_token_id=tokenizer.pad_token_id,
            )
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_labels.append(parse_label(response))
        true_labels.append(row["updated_label"])
    return pred_labels, true_labels


def print_report(true_labels, pred_labels):
    results_df = pd.DataFrame({"true": true_labels, "pred": pred_labels})
    valid = results_df.dropna(subset=["pred", "true"])
    report = classification_report(
        valid["true"], valid["pred"],
        labels=LABELS, target_names=LABELS, zero_division=0, output_dict=True,
    )
    for label in LABELS:
        info = report.get(label, {})
        print(f"  {label:25s}  P={info.get('precision', 0):.3f}  R={info.get('recall', 0):.3f}  F1={info.get('f1-score', 0):.3f}")
    print(f"\n  {'MACRO AVG':25s}  F1={report['macro avg']['f1-score']:.4f}")
    print(f"  {'ACCURACY':25s}  {report['accuracy']:.4f}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Evaluate a LoRA adapter on edu test/dev set")
    parser.add_argument("--model", type=str, required=True, help="Base HF model name")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--data_dir", type=str, default="logic_data")
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev", "train"])
    parser.add_argument("--output_dir", type=str, default="results/lora")
    parser.add_argument("--max_new", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    model, tokenizer = load_model(args.model, args.adapter_path)

    data_path = f"{args.data_dir}/edu_{args.split}.csv"
    df = pd.read_csv(data_path, usecols=["source_article", "updated_label"])
    print(f"Eval split: {args.split} ({len(df)} samples)", flush=True)

    pred_labels, true_labels = evaluate(model, tokenizer, df, args.max_new, args.max_length)

    adapter_name = args.adapter_path.rstrip("/").split("/")[-1]
    results_df = pd.DataFrame({
        "text": df["source_article"], "true": true_labels,
        "pred": pred_labels, "correct": [p == t for p, t in zip(pred_labels, true_labels)],
    })
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = f"{args.output_dir}/{adapter_name}_{args.split}_results.csv"
    results_df.to_csv(results_path, index=False)

    print_report(true_labels, pred_labels)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
