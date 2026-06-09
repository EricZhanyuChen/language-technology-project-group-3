import json
import random
from collections import Counter, defaultdict
from itertools import cycle

random.seed(42)

INPUT_DIR = "logic_data/augmented"

with open(f"{INPUT_DIR}/translated.json") as f:
    train_data = json.load(f)
with open(f"{INPUT_DIR}/translated_test.json") as f:
    test_data = json.load(f)
with open(f"{INPUT_DIR}/translated_dev.json") as f:
    dev_data = json.load(f)


def make_split(data, lang_field):
    return [{"updated_label": d["updated_label"], "source_article": d[lang_field]} for d in data]


# Experiment 1 test files + English baseline
test_en = make_split(test_data, "source_article")
test_nl = make_split(test_data, "source_article_nl")
test_zh = make_split(test_data, "source_article_zh")

dev_en = make_split(dev_data, "source_article")
dev_nl = make_split(dev_data, "source_article_nl")
dev_zh = make_split(dev_data, "source_article_zh")

# Experiment 2: balanced multilingual training (language + class balanced)
lang_fields = ["source_article", "source_article_nl", "source_article_zh"]
lang_codes = ["en", "nl", "zh"]

# Step 1: group by class, shuffle within each class, then assign language round-robin
from collections import defaultdict
groups = defaultdict(list)
for d in train_data:
    groups[d["updated_label"]].append(d)

max_count = max(len(v) for v in groups.values())

train_multi = []
for label, items in groups.items():
    random.shuffle(items)
    # Oversample to max_count
    pooled = []
    for s in cycle(items):
        pooled.append(s)
        if len(pooled) == max_count:
            break
    # Assign language round-robin within this class
    for i, item in enumerate(pooled):
        lang_idx = i % 3
        train_multi.append({
            "source_article": item[lang_fields[lang_idx]],
            "updated_label": item["updated_label"],
            "reasoning": item.get("reasoning", ""),
            "source": item.get("source", ""),
            "lang": lang_codes[lang_idx],
        })

random.shuffle(train_multi)

# Write files
output_dir = INPUT_DIR
files = {
    "test_en.json": test_en,
    "test_nl.json": test_nl,
    "test_zh.json": test_zh,
    "dev_en.json": dev_en,
    "dev_nl.json": dev_nl,
    "dev_zh.json": dev_zh,
    "train_multilingual.json": train_multi,
}

for fname, data in files.items():
    path = f"{output_dir}/{fname}"
    with open(path, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    labels = Counter(d["updated_label"] for d in data)
    print(f"\n{fname}: {len(data)} samples")
    for lbl, cnt in sorted(labels.items(), key=lambda x: -x[1]):
        print(f"  {lbl}: {cnt}")
