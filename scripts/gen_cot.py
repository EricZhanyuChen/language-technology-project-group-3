"""
Generate 4-step CoT reasoning for original rows in combined_train.json.

Usage:
    python scripts/gen_cot.py                          # run with defaults
    python scripts/gen_cot.py --batch-size 5           # smaller batches
    python scripts/gen_cot.py --dry-run                 # test without API calls
    python scripts/gen_cot.py --checkpoint-every 5      # save more frequently

Supports checkpoint/resume: skips rows that already have reasoning.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

DATA_DIR = Path("logic_data/augmented")
COMBINED = DATA_DIR / "merged_dataset_V3.json"
CHECKPOINT = DATA_DIR / "merged_dataset_V3_cot_checkpoint.json"

FALLACY_KB = {
    "faulty generalization": {
        "definition": "An informal fallacy wherein a conclusion is drawn about all or many instances of a phenomenon on the basis of one or a few instances of that phenomenon.",
        "canonical": "[MSK1] has attribute [MSK2]. [MSK1] is a subset of [MSK3]. Therefore, all [MSK3] has attribute [MSK2].",
        "subtypes": [
            "Hasty Generalization", "Slippery Slope", "Accident", "Fallacy of Division",
            "Error of Division", "Error of Composition", "Causal Oversimplification",
            "Association Fallacy", "Guilt by Association", "Composition Fallacy",
            "Ecological Fallacy", "Conjunction Fallacy", "False Analogy",
            "Inconsistent Comparison", "Overwhelming Exception", "False Equivalence",
            "All Things Are Equal", "McNamara Fallacy",
        ],
    },
    "false causality": {
        "definition": "A statement that jumps to a conclusion implying a causal relationship without supporting evidence.",
        "canonical": "[MSK1] occurred, then [MSK2] occurred. Therefore, [MSK1] caused [MSK2].",
        "subtypes": [
            "Post hoc ergo propter hoc", "Cum hoc ergo propter hoc",
            "Regression Fallacy", "Consecutive Relation", "Magical Thinking",
            "Gambler's Fallacy", "Ludic Fallacy",
        ],
    },
    "circular reasoning": {
        "definition": "A fallacy where the end of an argument comes back to the beginning without having proven itself.",
        "canonical": "[MSK1] is true because of [MSK2]. [MSK2] is true because of [MSK1].",
        "subtypes": ["Circular Reasoning", "Homunculus Fallacy"],
    },
    "ad populum": {
        "definition": "A fallacious argument which is based on affirming that something is real or better because the majority thinks so.",
        "canonical": "A lot of people believe [MSK1]. Therefore, [MSK1] must be true.",
        "subtypes": [
            "Appeal to the Public", "Ad Numerum", "Appeal to the Numbers", "Bandwagon Fallacy",
        ],
    },
    "ad hominem": {
        "definition": "An irrelevant attack towards the person or some aspect of the person who is making the argument, instead of addressing the argument or position directly.",
        "canonical": "[MSK1] is claiming [MSK2]. [MSK1] is a moron. Therefore, [MSK2] is not true.",
        "subtypes": [
            "Genetic Fallacy", "Tu quoque", "Bulverism", "Poisoning the Well",
            "Appeal to Hypocrisy", "Traitorous Critic",
        ],
    },
    "fallacy of logic": {
        "definition": "An error in the logical structure of an argument.",
        "canonical": "If [MSK1] is true, then [MSK2] is true. [MSK2] is true. Therefore, [MSK1] is true.",
        "subtypes": [
            "Affirming the Consequent", "Denying the Antecedent", "False Analogy",
            "Non sequitur", "Affirming the Disjunct", "Argument From Fallacy",
            "Appeal to Probability", "Undistributed Middle", "Moral Equivalence",
            "Self Contradiction", "Internal Contradiction", "Masked-man Fallacy",
            "Illicit Major", "Illicit Minor", "Existential Fallacy",
            "Kettle Logic", "Affirmative Conclusion from a Negative Premise",
            "Negative Conclusion from a Negative Premise", "Exclusive Premises",
        ],
    },
    "appeal to emotion": {
        "definition": "Manipulation of the recipient's emotions in order to win an argument.",
        "canonical": "[MSK1] is made without evidence. In place of evidence, emotion is used to convince the interlocutor that [MSK1] is true.",
        "subtypes": [
            "Appeal to Pity", "Appeal to Fear", "Ad baculum (appeal to force)",
            "Appeal to Ridicule", "Appeal to Gallery", "Wishful Thinking",
            "Appeal to Consequences", "Appeal to Spite", "Appeal to Flattery",
        ],
    },
    "false dilemma": {
        "definition": "A claim presenting only two options or sides when there are many options or sides.",
        "canonical": "Either [MSK1] or [MSK2] is true.",
        "subtypes": [
            "Either/Or thinking", "Black-or-White Fallacy", "False Dichotomy",
            "Nirvana Fallacy", "Perfect Solution",
        ],
    },
    "equivocation": {
        "definition": "Way 1: same word used with different meanings in premise vs conclusion. Way 2: evasive language avoids committing to a position. NOTE: standalone evasion without an argument is NOT equivocation.",
        "canonical": "[MSK1] is used to mean [MSK2] in the premise. [MSK1] is used to mean [MSK3] in the conclusion.",
        "subtypes": [
            "Uncertain use of term or concept", "Reification", "Continuum fallacy",
            "False attribution", "Moral equivalence", "Etymological Fallacy",
        ],
    },
    "fallacy of extension": {
        "definition": "Also known as straw man, this is when an argument appears to be refuted by being replaced with an argument with a similar but weaker argument.",
        "canonical": "[MSK1] makes claim [MSK2]. [MSK3] restates [MSK2] (in a distorted way). [MSK3] attacks the distorted version of [MSK2]. Therefore, [MSK2] is false.",
        "subtypes": ["Straw man", "Suppressed Correlative"],
    },
    "fallacy of relevance": {
        "definition": "Also known as red herring, this fallacy occurs when the speaker attempts to divert attention from the primary argument by offering a point that does not suffice as counterpoint/supporting evidence (even if it is true).",
        "canonical": "It is claimed that [MSK1] implies [MSK2], whereas [MSK1] is unrelated to [MSK2].",
        "subtypes": [
            "Red herring", "Two wrongs make a right", "Argument to moderation",
            "Moralistic fallacy", "Moral equivalence", "Logic chopping",
            "Proof by assertion", "Argument from silence", "Irrelevant material",
            "Relative privation",
        ],
    },
    "fallacy of credibility": {
        "definition": "An appeal is made to some form of ethics, authority, or credibility.",
        "canonical": "[MSK1] claims that [MSK2]. [MSK1] are experts in the field concerning [MSK2]. Therefore, [MSK2] should be believed.",
        "subtypes": [
            "Appeal to authority", "Appeal to nature", "Naturalistic fallacy",
            "Appeal to tradition", "Chronological snobbery", "Appeal to novelty",
            "Ipse dixit", "Etymological fallacy", "Appeal to poverty",
            "Appeal to accomplishment",
        ],
    },
    "intentional": {
        "definition": "A custom category for when an argument has some element that shows intent of a speaker to win an argument without actual supporting evidence.",
        "canonical": "[MSK1] knows [MSK2] is incorrect. [MSK1] still claims that [MSK2] is correct using an incorrect argument.",
        "subtypes": [
            "Texas sharpshooter", "Cherry picking", "McNamara fallacy",
            "No true scotsman", "Appeal to ignorance", "Complex question",
            "Moving the goalposts", "Loaded question", "Special pleading",
            "Hiding information", "Many questions", "Incredulity",
            "Divine Fallacy", "Quoting out of context", "Shifted burden of proof",
        ],
    },
}

SYSTEM_PROMPT = (
    "You are an expert in logical fallacy detection. "
    "Given a text and its fallacy label, produce a 4-step Chain-of-Thought reasoning.\n\n"
    "FORMAT (each Step MUST start on its OWN LINE):\n"
    "Step 1 - Structure: Premise: [...]. Conclusion: [...].\n"
    "Step 2 - Pattern: [natural language explanation of the fallacy pattern]\n"
    "Step 3 - Label + Canonical Form: This is {label}. Canonical form: [...]. Instantiation: [...].\n"
    "  - Use [MSK1], [MSK2], etc. placeholders in Canonical form if they fit naturally.\n"
    "  - If the canonical form does not map well to the text, describe it in natural language instead.\n"
    "Step 4 - Verdict: [why conclusion does NOT follow from the premise]\n\n"
    "Output ONLY the 4 steps. No JSON, no markdown, no extra text."
)


def get_client():
    key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY not set")
    return OpenAI(api_key=key, base_url="https://api.deepseek.com")


def call_api(client, messages, temperature=0.7, max_tokens=600, retries=3):
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** (attempt + 1)
            print(f"  [retry {attempt+1}/{retries}] {e} -- waiting {wait}s")
            time.sleep(wait)
    raise RuntimeError("API failed after retries")


def load_fewshot(data, label, n=2):
    """Get n generated examples with reasoning for this label as few-shot."""
    examples = [d for d in data if d["source"] == "generated"
                and d["updated_label"] == label
                and d.get("reasoning")]
    if not examples:
        return ""
    import random
    random.seed(42)
    picks = random.sample(examples, min(n, len(examples)))
    parts = []
    for ex in picks:
        parts.append(f'Text: "{ex["source_article"]}"\n{ex["reasoning"]}')
    return "\n\n---\n\n".join(parts)


def build_prompt(text, label, fewshot):
    kb = FALLACY_KB.get(label, {})
    definition = kb.get("definition", "A logical fallacy.")
    canonical = kb.get("canonical", "Not available")
    subtypes = kb.get("subtypes", [])
    subs_text = ", ".join(subtypes) if subtypes else "N/A"

    user = (
        f'FALLACY TYPE: {label}\n'
        f'DEFINITION: {definition}\n'
        f'CANONICAL FORM: {canonical}\n'
        f'SUBTYPES: {subs_text}\n\n'
        f'TEXT TO ANALYZE:\n"{text}"\n\n'
    )
    if fewshot:
        user += f"EXAMPLES (follow this format exactly):\n{fewshot}\n\n"
    user += (
        f"Now analyze the TEXT above. Output the 4 steps for \"{label}\":\n"
        "Step 1 - Structure: ...\nStep 2 - Pattern: ...\n"
        "Step 3 - Label + Canonical Form: ... (use [MSK] placeholders if they fit, "
        "otherwise describe in natural language)\nStep 4 - Verdict: ..."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def validate_reasoning(r):
    """Check that reasoning has all 4 steps."""
    if not r:
        return False
    for s in range(1, 5):
        if f"Step {s}" not in r:
            return False
    return True




def build_filter_prompt(text, label, reasoning):
    """Second-pass validation: scoring-based strict check."""
    prompt = (
        f"Fallacy type: {label}\n\n"
        f"Text:\n\"{text}\"\n\n"
        f"Reasoning:\n{reasoning}\n\n"
        f"Rate each 1-5 (5=perfect, 1=wrong):\n"
        f'1. Does the text clearly demonstrate "{label}"? '
        f"2. Is Step 2 specific to this text (not generic boilerplate)?\n"
        f"3. Is Step 4 substantive (explains why conclusion fails)?\n\n"
        f"Format: 1:X 2:X 3:X  (each X is 1-5)"
    )
    return [
        {"role": "system", "content": "Rate strictly. Format: 1:X 2:X 3:X"},
        {"role": "user", "content": prompt},
    ]




def main():
    parser = argparse.ArgumentParser(description="Generate CoT for original rows")
    parser.add_argument("--batch-size", type=int, default=1, help="API calls per batch (for logging)")
    parser.add_argument("--checkpoint-every", type=int, default=10, help="Save checkpoint every N rows")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument("--limit", type=int, default=0, help="Stop after N rows (0=all)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent API workers")
    parser.add_argument("--filter", action="store_true", help="Run second-pass validation on rows with reasoning")
    args = parser.parse_args()

    with open(COMBINED, encoding="utf-8") as f:
        data = json.load(f)

    # Resume from checkpoint if exists
    if CHECKPOINT.exists():
        print(f"Loading checkpoint: {CHECKPOINT}")
        with open(CHECKPOINT, encoding="utf-8") as f:
            data = json.load(f)

    # Filter mode: validate existing reasoning
    if args.filter:
        cot_path = DATA_DIR / "train_cot.json"
        if cot_path.exists():
            print(f"Loading CoT data from {cot_path}")
            with open(cot_path, encoding="utf-8") as f:
                data = json.load(f)
        filter_todo = [i for i, d in enumerate(data) if d.get("reasoning")]
        print(f"Filtering {len(filter_todo)} rows with reasoning...")
        kept, discarded = 0, 0
        def filter_one(idx):
            d = data[idx]
            label = d["updated_label"]
            text = d["source_article"]
            reasoning = d["reasoning"]
            msgs = build_filter_prompt(text, label, reasoning)
            try:
                ans = call_api(client, msgs, temperature=0.0, max_tokens=30)
                scores = re.findall(r':\s*([1-5])', ans.strip()[:30])
                passed = len(scores) >= 3 and all(int(s) >= 4 for s in scores[:3])
            except:
                passed = True  # keep on API error
            return idx, passed

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(filter_one, idx): idx for idx in filter_todo}
            for num, future in enumerate(as_completed(futures), 1):
                idx, passed = future.result()
                if passed:
                    kept += 1
                else:
                    data[idx]["reasoning"] = ""
                    discarded += 1
                if num % 50 == 0:
                    print(f"  {num}/{len(filter_todo)} — kept {kept}, discarded {discarded}")
        # Save
        with open(CHECKPOINT, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        missing = sum(1 for d in data if not d.get("reasoning"))
        print(f"\nFilter done: kept {kept}, discarded {discarded}")
        print(f"Rows still needing CoT: {missing}")
        print(f"Saved to {CHECKPOINT}")
        return

    # Find all rows without reasoning
    todo_indices = []
    for i, d in enumerate(data):
        if not d.get("reasoning"):
            todo_indices.append(i)

    total = len(todo_indices)
    print(f"Total rows in file: {len(data)}")
    print(f"Rows needing CoT: {total}")
    if args.limit > 0:
        todo_indices = todo_indices[:args.limit]
        print(f"Limited to: {len(todo_indices)}")

    if not todo_indices:
        print("Nothing to do — all original rows already have reasoning.")
        return

    if args.dry_run:
        # Show first 3 prompts
        for idx in todo_indices[:3]:
            d = data[idx]
            fewshot = load_fewshot(data, d["updated_label"])
            msgs = build_prompt(d["source_article"], d["updated_label"], fewshot)
            print(f"\n{'='*60}")
            print(f"[{idx}] label={d['updated_label']}")
            print(f"text: {d['source_article'][:100]}")
            print(f"--- system ({len(msgs[0]['content'])} chars) ---")
            print(msgs[0]["content"][:200] + "...")
            print(f"--- user ({len(msgs[1]['content'])} chars) ---")
            print(msgs[1]["content"][:300] + "...")
        print(f"\n[DRY RUN] Would process {len(todo_indices)} rows.")
        return

    client = get_client()
    checkpoint_counter = 0
    done = 0
    failed = 0

    def process_one(idx):
        """Generate CoT for a single row. Returns (idx, reasoning, status)."""
        d = data[idx]
        label = d["updated_label"]
        text = d["source_article"]
        fewshot = load_fewshot(data, label)
        msgs = build_prompt(text, label, fewshot)
        try:
            reasoning = call_api(client, msgs, temperature=args.temperature)
            if validate_reasoning(reasoning):
                return idx, reasoning, "OK"
            # Retry once
            msgs.append({"role": "assistant", "content": reasoning})
            msgs.append({"role": "user", "content":
                "Your output is missing Step(s). Output ALL 4 steps exactly:\n"
                "Step 1 - Structure: ...\nStep 2 - Pattern: ...\n"
                "Step 3 - Label + Canonical Form: ...\nStep 4 - Verdict: ..."})
            reasoning = call_api(client, msgs, temperature=0.5)
            if validate_reasoning(reasoning):
                return idx, reasoning, "OK (retry)"
            return idx, reasoning, "WARN (incomplete steps)"
        except Exception as e:
            return idx, None, f"FAIL: {e}"

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, idx): idx for idx in todo_indices}
        for num, future in enumerate(as_completed(futures), 1):
            idx, reasoning, status = future.result()
            label = data[idx]["updated_label"]
            if reasoning is not None:
                data[idx]["reasoning"] = reasoning
                done += 1
            else:
                failed += 1
            print(f"[{num}/{len(todo_indices)}] {label[:25]:25s} | {status}")

            checkpoint_counter += 1
            if checkpoint_counter >= args.checkpoint_every:
                with open(CHECKPOINT, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  >>> checkpoint saved ({done} done, {failed} failed)")
                checkpoint_counter = 0

    # Final save to checkpoint
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    # Count final status
    has_cot = sum(1 for d in data if d.get("reasoning"))
    no_cot = sum(1 for d in data if not d.get("reasoning"))
    print(f"\nDone: {done} generated, {failed} failed")
    print(f"Rows with CoT: {has_cot}/{len(data)}, still missing: {no_cot}")
    print(f"Checkpoint saved to: {CHECKPOINT}")
    print(f"\nWhen all rows are done, run:")
    print(f"  python scripts/gen_cot.py --finalize")


if __name__ == "__main__":
    import sys
    if "--finalize" in sys.argv:
        with open(CHECKPOINT, encoding="utf-8") as f:
            data = json.load(f)
        with open(COMBINED, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        has_cot = sum(1 for d in data if d.get("reasoning"))
        print(f"Finalized: {len(data)} rows written to {COMBINED}")
        print(f"Rows with reasoning: {has_cot}/{len(data)}")
    else:
        main()
