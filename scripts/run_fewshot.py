import os, re, sys, json, time, argparse, warnings
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

SEED = 42
LABELS = [
    "ad hominem", "ad populum", "appeal to emotion", "circular reasoning",
    "equivocation", "fallacy of credibility", "fallacy of extension",
    "fallacy of logic", "fallacy of relevance", "false causality",
    "false dilemma", "faulty generalization", "intentional",
]

DEFINITIONS = '''ad hominem
Definition: This fallacy occurs when a speaker trying to argue the opposing view on a topic makes claims against the other speaker instead of the position they are maintaining.
Example: "Bernie Saunders wouldn't make a good president because he looks like a sad muppet."

ad populum
Definition: This fallacy occurs when an argument is based on affirming that something is true because a statistical majority believes so.
Example: "All my friends are doing a low carb diet. That must be the only way to lose weight."

appeal to emotion
Definition: This fallacy is when emotion is used to support an argument, such as pity, fear, anger, etc.
Example: "I know the exam is graded based on performance, but you should give me an A. My cat has been sick, my car broke down, and I've had a cold, so it was really hard for me to study."

circular reasoning
Definition: This fallacy occurs when an argument uses the claim it is trying to prove as proof that the claim is true.
Example: "All eighteen-year-olds have the right to vote because it's legal for them to vote."

equivocation
Definition: This fallacy occurs when ambiguous or evasive language is used to avoid committing to a position, or when the same word is used with different meanings in an argument.
Example: "A warm beer is better than a cold beer. After all, nothing is better than a cold beer, and a warm beer is better than nothing."

fallacy of credibility
Definition: This fallacy is when an appeal is made to some form of ethics, authority, or credibility.
Example: "Stephen Hawking is the smartest scientist on the planet, so if he says global warming is real, it must be true!"

fallacy of extension
Definition: Also known as straw man, this is when an argument appears to be refuted by being replaced with an argument with a similar but weaker position.
Example: "Senator Jones says that we should not fund the attack submarine program. I disagree entirely. I can't understand why he wants to leave us defenseless."

fallacy of logic
Definition: This fallacy occurs when there is a logical flaw in the reasoning behind the argument, such as a propositional logic flaw.
Example: "If my hair looks nice, people will love me."

fallacy of relevance
Definition: Also known as red herring, this fallacy occurs when the speaker attempts to divert attention from the primary argument by offering an irrelevant point.
Example: "Mother: It's bedtime Jane. Jane: Mom, how do ants feed their babies? Mother: Don't know dear, close your eyes now."

false causality
Definition: This fallacy occurs when an argument assumes that since two events are correlated, they must also have a cause and effect relationship.
Example: "Every time I wear this necklace, I pass my exams. Therefore, wearing this necklace causes me to pass my exams."

false dilemma
Definition: This fallacy is when incorrect limitations are made on the possible options in a scenario when there could be other options.
Example: "Either you are for us, or you're against us!"

faulty generalization
Definition: This fallacy occurs when an argument applies a belief to a large population without having a large enough sample to do so.
Example: "Annie must like Starbucks because all girls like Starbucks."

intentional
Definition: This occurs when an argument has an element that shows "intent" of a speaker to win an argument without actual supporting evidence.
Example: "Aliens must exist because there is no evidence that they don't exist."'''

SYSTEM_PROMPT = "Classify the logical fallacy in the given text. Here are the definitions and examples:\n\n" + DEFINITIONS + "\n\nAnswer with only the fallacy name from the list above. Do not explain."
USER_TEMPLATE = "Text: {text}\nFallacy:"


def load_model(model_name):
    from transformers import (
        AutoTokenizer, AutoModelForCausalLM, AutoModel,
        BitsAndBytesConfig, pipeline
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"Loading {model_name} with 4-bit ...", flush=True)
    t0 = time.time()

    kwargs = dict(
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    name_lower = model_name.lower()

    if "nemotron" in name_lower:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, **kwargs
        )
    elif "gemma" in name_lower and "4" in name_lower:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb_config, **kwargs
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, quantization_config=bnb_config, **kwargs
            )
        except Exception:
            model = AutoModel.from_pretrained(
                model_name, quantization_config=bnb_config, **kwargs
            )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  done in {time.time()-t0:.1f}s", flush=True)
    return model, tokenizer


def parse_output(text):
    if not text:
        return None
    text_lower = text.lower()
    sorted_labels = sorted(LABELS, key=len, reverse=True)
    matches = []
    for label in sorted_labels:
        for m in re.finditer(r"\b" + re.escape(label) + r"\b", text_lower):
            matches.append((m.start(), m.end(), label))
    if not matches:
        return None
    matches.sort(key=lambda x: x[0])
    return matches[-1][2]


def build_messages(system, user_text):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_TEMPLATE.format(text=user_text)},
    ]


def generate(model, tokenizer, messages, model_name):
    name_lower = model_name.lower()

    enable_thinking = False
    chat_kwargs = {}

    if "nemotron" in name_lower:
        chat_kwargs["enable_thinking"] = False
    elif "gemma" in name_lower:
        pass
    elif "qwen" in name_lower:
        chat_kwargs["enable_thinking"] = False

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **chat_kwargs,
    )
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="HF model name or path")
    parser.add_argument("--data_dir", type=str, default="logic_data")
    parser.add_argument("--output_dir", type=str, default="fewshot_results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Stratified subsample to this many (e.g. 150)")
    args = parser.parse_args()

    model_short = args.model.split("/")[-1]
    os.makedirs(args.output_dir, exist_ok=True)

    test_df = pd.read_csv(
        f"{args.data_dir}/edu_test.csv",
        usecols=["source_article", "updated_label"]
    )

    if args.max_samples and args.max_samples < len(test_df):
        from sklearn.model_selection import train_test_split
        _, sampled = train_test_split(
            test_df, train_size=args.max_samples,
            stratify=test_df["updated_label"],
            random_state=SEED,
        )
        test_df = sampled.reset_index(drop=True)
        print(f"Subsampled to {len(test_df)} (stratified)", flush=True)

    texts = test_df["source_article"].tolist()
    true_labels = test_df["updated_label"].tolist()

    print(f"Test samples: {len(texts)}", flush=True)

    model, tokenizer = load_model(args.model)

    predictions = []
    t_start = time.time()

    pbar = tqdm(total=len(texts), desc="Inference", unit="sample")
    for text, true in zip(texts, true_labels):
        messages = build_messages(SYSTEM_PROMPT, text)
        try:
            response = generate(model, tokenizer, messages, args.model)
            pred = parse_output(response)
        except Exception as e:
            tqdm.write(f"Error: {e}")
            pred = None

        predictions.append(pred)
        pbar.update(1)
    pbar.close()

    results_df = pd.DataFrame({
        "text": texts,
        "true": true_labels,
        "pred": predictions,
        "correct": [p == t for p, t in zip(predictions, true_labels)],
    })
    out_path = f"{args.output_dir}/{model_short}_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved to {out_path}", flush=True)

    valid = results_df.dropna(subset=["pred"])
    true_valid = valid["true"].tolist()
    pred_valid = valid["pred"].tolist()

    report = classification_report(
        true_valid, pred_valid,
        labels=LABELS,
        target_names=LABELS,
        zero_division=0,
        output_dict=True,
    )

    for label in LABELS:
        info = report.get(label, {})
        print(f"  {label:25s}  P={info.get('precision', 0):.3f}  R={info.get('recall', 0):.3f}  F1={info.get('f1-score', 0):.3f}")

    macro_f1 = report["macro avg"]["f1-score"]
    weighted_f1 = report["weighted avg"]["f1-score"]
    acc = report["accuracy"]
    print(f"\n  {'MACRO AVG':25s}  F1={macro_f1:.4f}")
    print(f"  {'WEIGHTED AVG':25s}  F1={weighted_f1:.4f}")
    print(f"  {'ACCURACY':25s}  {acc:.4f}")

    total_time = time.time() - t_start
    print(f"\nTotal time: {total_time:.1f}s ({total_time/len(texts):.2f}s per sample)", flush=True)


if __name__ == "__main__":
    main()
