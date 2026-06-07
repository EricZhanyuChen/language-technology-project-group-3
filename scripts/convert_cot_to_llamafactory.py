"""
Convert train_cot_filtered.json to LLaMA Factory conversation format.

Usage:
    python scripts/convert_cot_to_llamafactory.py --model qwen3_5
    python scripts/convert_cot_to_llamafactory.py --model gemma4
    python scripts/convert_cot_to_llamafactory.py --both
"""

import argparse
import json
from pathlib import Path

DATA_DIR = Path("logic_data/augmented")
INPUT = DATA_DIR / "train_cot_filtered.json"

SYSTEM_PROMPT = (
    "Classify the logical fallacy in the given text.\n"
    "ad hominem: attacks the speaker instead of the argument.\n"
    "ad populum: believes something is true because many people believe it.\n"
    "appeal to emotion: uses emotion to manipulate instead of logic.\n"
    "circular reasoning: uses the conclusion as its own premise.\n"
    "equivocation: uses ambiguous language or shifts word meanings.\n"
    "fallacy of credibility: appeals to authority, tradition, or novelty.\n"
    "fallacy of extension: distorts an argument to make it easier to attack.\n"
    "fallacy of logic: has a structural flaw in reasoning.\n"
    "fallacy of relevance: diverts attention with irrelevant points.\n"
    "false causality: assumes correlation proves causation.\n"
    "false dilemma: presents only two extreme options.\n"
    "faulty generalization: generalizes from insufficient evidence.\n"
    "intentional: uses tactics to win an argument without evidence."
)

THOUGHT_WORDS = {
    "qwen3_5": ("<think>\n", "\n</think>\n\n"),
    "gemma4": ("<|channel>thought\n", "<channel|>\n"),
}


def convert(data, model):
    thought_open, thought_close = THOUGHT_WORDS[model]
    output = []
    for d in data:
        text = d["source_article"]
        reasoning = d["reasoning"]
        label = d["updated_label"]
        assistant_value = f"{thought_open}{reasoning}{thought_close}{label}"
        output.append({
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": f'Text: "{text}"'},
                {"from": "gpt", "value": assistant_value},
            ]
        })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["qwen3_5", "gemma4"])
    parser.add_argument("--both", action="store_true")
    args = parser.parse_args()

    with open(INPUT, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {INPUT}")

    models = ["qwen3_5", "gemma4"] if args.both else [args.model]

    for model in models:
        output = convert(data, model)
        out_path = DATA_DIR / f"llamafactory_cot_{model}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(output)} samples to {out_path}")

        # Show sample
        print(f"\nSample ({model}):")
        sample = output[0]
        print(f"  system: {sample['conversations'][0]['value'][:80]}...")
        print(f"  human: {sample['conversations'][1]['value'][:80]}...")
        gpt_val = sample['conversations'][2]['value']
        thought_end = gpt_val.find(THOUGHT_WORDS[model][1])
        thought_content = gpt_val[len(THOUGHT_WORDS[model][0]):thought_end]
        label = gpt_val[thought_end + len(THOUGHT_WORDS[model][1]):]
        print(f"  gpt thought: {thought_content[:100]}...")
        print(f"  gpt label: {label}")
        print()


if __name__ == "__main__":
    main()
