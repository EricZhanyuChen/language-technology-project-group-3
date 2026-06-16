# GitHub Upload Index

> Generated: 2026-06-16
> Exclusions based on `.gitignore`: `lt_env/`, `lora_results/`, `scripts/slurm/`, `memory/`, `AGENTS.md`, `presentation1/`, `presentation2/`, `paper/`, `results/logs/`

---

## Root

| File | Description |
|------|-------------|
| .gitignore | Git ignore rules |

---

## notebooks/

| File | Description |
|------|-------------|
| notebooks/LT_Baseline.ipynb | RoBERTa-base full fine-tune baseline (Macro F1=0.57) |

---

## logic_data/ — Raw Dataset

| File | Size | Description |
|------|------|-------------|
| logic_data/edu_all.csv | 751K | Full edu data (2452 rows, includes rationale) |
| logic_data/edu_train.csv | 619K | Edu train split (1849 rows) |
| logic_data/edu_dev.csv | 100K | Edu dev split (300 rows) |
| logic_data/edu_test.csv | 103K | Edu test split (300 rows) |
| logic_data/climate_all.csv | 424K | Full climate data (1351 rows, OOD) |
| logic_data/climate_train.csv | 269K | Climate train split (855 rows, not used for training) |
| logic_data/climate_dev.csv | 95K | Climate dev split (277 rows) |
| logic_data/climate_test.csv | 61K | Climate test split (219 rows, OOD eval) |
| logic_data/climate_dev_mh.csv | 99K | Climate dev multi-label version |
| logic_data/climate_test_mh.csv | 65K | Climate test multi-label version |
| logic_data/climate_train_mh.csv | 287K | Climate train multi-label version |
| logic_data/mappings.csv | 5.0K | 13-class definitions + logical forms + masked forms |

---

## logic_data/augmented/ — Generated Data

| File | Size | Description |
|------|------|-------------|
| logic_data/augmented/merged_dataset_V3.json | 655K | Augmented data (1849 + 1079 = 2928 samples, no CoT) |
| logic_data/augmented/train_cot.json | 3.6M | CoT generation output (2928 samples, 4-step reasoning) |
| logic_data/augmented/train_cot_filtered.json | 3.6M | CoT filtered (2915 samples, rule-based filter removed 13) |
| logic_data/augmented/combined_train.json | 1.5M | Merged training data |
| logic_data/augmented/translated.json | 4.5M | Translation: 2915 training samples (NL/ZH) |
| logic_data/augmented/translated_dev.json | 161K | Translation: 300 dev samples |
| logic_data/augmented/translated_test.json | 164K | Translation: 300 test samples |
| logic_data/augmented/failed_examples.json | 882B | 3 failed translation samples |
| logic_data/augmented/train_multilingual.json | 5.1M | Multilingual training set (4147 samples, EN/NL/ZH balanced) |
| logic_data/augmented/dev_en.json | 63K | Dev English |
| logic_data/augmented/dev_nl.json | 66K | Dev Dutch |
| logic_data/augmented/dev_zh.json | 60K | Dev Chinese |
| logic_data/augmented/test_en.json | 64K | Test English |
| logic_data/augmented/test_nl.json | 68K | Test Dutch |
| logic_data/augmented/test_zh.json | 61K | Test Chinese |
| logic_data/augmented/dataset_info.json | 4.9K | LLaMA Factory dataset registration |

### LLaMA Factory Training Format

| File | Size | Description |
|------|------|-------------|
| logic_data/augmented/llamafactory_train.json | 3.5M | Augmented data training format (2928 samples) |
| logic_data/augmented/llamafactory_train_original.json | 2.4M | Original data training format (1849 samples) |
| logic_data/augmented/llamafactory_dev.json | 366K | Dev format |
| logic_data/augmented/llamafactory_test.json | 368K | Test format |
| logic_data/augmented/llamafactory_test_en.json | 368K | Test English format |
| logic_data/augmented/llamafactory_test_nl.json | 372K | Test Dutch format |
| logic_data/augmented/llamafactory_test_zh.json | 365K | Test Chinese format |
| logic_data/augmented/llamafactory_climate_test.json | 289K | Climate OOD test format |
| logic_data/augmented/llamafactory_cot_gemma4.json | 6.4M | CoT training data (Gemma4 thinking format) |
| logic_data/augmented/llamafactory_cot_qwen3_5.json | 6.4M | CoT training data (Qwen thinking format) |
| logic_data/augmented/llamafactory_train_multilingual.json | 5.0M | Multilingual training format (4147 samples) |
| logic_data/augmented/llamafactory_train_multilingual_balanced.json | 7.0M | Multilingual balanced training format (5517 samples) |
| logic_data/augmented/llamafactory_train_aug50.json | 443K | Aug50 training format (2199 samples) |
| logic_data/augmented/llamafactory_train_targeted_aug.json | 2.5M | Targeted augmentation training format (2049 samples) |
| logic_data/augmented/llamafactory_train_orig_dist.json | 3.6M | Augmented data with preserved original distribution |

---

## scripts/ — Python Scripts

| File | Description |
|------|-------------|
| scripts/gen_cot.py | CoT generation (DeepSeek API, concurrent + checkpoint resume + filtering) |
| scripts/convert_cot_to_llamafactory.py | Convert CoT data to LLaMA Factory format |
| scripts/prepare_multilingual_data.py | Multilingual data preparation (trilingual test/dev + balanced training set) |
| scripts/run_fewshot.py | Few-shot evaluation script (Qwen/Gemma) |
| scripts/eval_adapter.py | Generic LoRA adapter evaluation script |
| scripts/parse_lf_results.py | Parse LLaMA Factory predictions, compute macro-F1 |
| scripts/parse_climate_results.py | Parse climate_test predictions |

---

## scripts/yaml/ — LLaMA Factory Configs

### Gemma 4 31B-it

| File | Description |
|------|-------------|
| scripts/yaml/exp004.yaml | Gemma4 + augmented data (label-only) |
| scripts/yaml/ablation_gemma4_orig.yaml | Gemma4 + original data (ablation) |
| scripts/yaml/ablation_gemma4_orig_r8.yaml | Gemma4 + original data r=8 |
| scripts/yaml/exp006_gemma4_cot.yaml | Gemma4 CoT fine-tuning |
| scripts/yaml/multilingual_gemma4.yaml | Gemma4 multilingual training |
| scripts/yaml/multi_balanced_gemma4.yaml | Gemma4 multilingual balanced training |
| scripts/yaml/targeted_aug.yaml | Gemma4 targeted augmentation |
| scripts/yaml/orig_dist_aug.yaml | Gemma4 augmented data with original distribution |
| scripts/yaml/eval_gemma4_orig_en.yaml | Gemma4 eval config |
| scripts/yaml/eval_gemma4_cot_thinking.yaml | Gemma4 CoT eval (enable_thinking=true) |

### Qwen 3.5-27B

| File | Description |
|------|-------------|
| scripts/yaml/exp005_qwen3.5-27b.yaml | Qwen 27B nothink (DISCARDED) |
| scripts/yaml/exp005b_qwen3.5-27b-aug.yaml | Qwen 27B augmented data |
| scripts/yaml/exp005b_qwen3.5-27b-reasoning.yaml | Qwen 27B thinking mode (DISCARDED) |
| scripts/yaml/ablation_qwen_orig.yaml | Qwen 27B + original data (ablation) |
| scripts/yaml/exp006_qwen3.5_cot.yaml | Qwen 27B CoT fine-tuning |
| scripts/yaml/multilingual_qwen3.5.yaml | Qwen 27B multilingual training |
| scripts/yaml/eval_qwen27b_orig_en.yaml | Qwen 27B eval config |
| scripts/yaml/eval_qwen27b_cot_thinking.yaml | Qwen 27B CoT eval (enable_thinking=true) |

### Qwen 3.5-9B

| File | Description |
|------|-------------|
| scripts/yaml/9b/label_qwen9b_orig_r16.yaml | Qwen 9B original data r=16 |
| scripts/yaml/9b/label_qwen9b_orig_r8.yaml | Qwen 9B original data r=8 |
| scripts/yaml/9b/label_qwen9b_aug_r16.yaml | Qwen 9B augmented data r=16 |
| scripts/yaml/9b/label_qwen9b_aug_r8.yaml | Qwen 9B augmented data r=8 |
| scripts/yaml/9b/label_qwen9b_aug50_r16.yaml | Qwen 9B Aug50 r=16 |
| scripts/yaml/9b/cot_qwen9b_r16.yaml | Qwen 9B CoT r=16 |
| scripts/yaml/9b/cot_qwen9b_r8.yaml | Qwen 9B CoT r=8 |
| scripts/yaml/9b/multi_qwen9b_r16.yaml | Qwen 9B multilingual r=16 |
| scripts/yaml/9b/multi_qwen9b_r8.yaml | Qwen 9B multilingual r=8 |

### r=8 Gemma/Qwen Comparison

| File | Description |
|------|-------------|
| scripts/yaml/r8/label_gemma_orig_r8.yaml | Gemma4 original r=8 |
| scripts/yaml/r8/label_gemma_aug_r8.yaml | Gemma4 augmented r=8 |
| scripts/yaml/r8/label_qwen_orig_r8.yaml | Qwen 27B original r=8 |
| scripts/yaml/r8/label_qwen_aug_r8.yaml | Qwen 27B augmented r=8 |
| scripts/yaml/r8/cot_gemma_r8.yaml | Gemma4 CoT r=8 |
| scripts/yaml/r8/cot_qwen_r8.yaml | Qwen 27B CoT r=8 |
| scripts/yaml/r8/multi_gemma_r8.yaml | Gemma4 multilingual r=8 |
| scripts/yaml/r8/multi_qwen_r8.yaml | Qwen 27B multilingual r=8 |

### Eval Configs (NL/ZH)

| File | Description |
|------|-------------|
| scripts/yaml/eval_nl_zh/eval_gemma4_multilingual_nl.yaml | Gemma4 multilingual -> test NL |
| scripts/yaml/eval_nl_zh/eval_gemma4_multilingual_zh.yaml | Gemma4 multilingual -> test ZH |
| scripts/yaml/eval_nl_zh/eval_gemma4_orig_nl.yaml | Gemma4 original -> test NL |
| scripts/yaml/eval_nl_zh/eval_gemma4_orig_zh.yaml | Gemma4 original -> test ZH |
| scripts/yaml/eval_nl_zh/eval_qwen27b_multilingual_nl.yaml | Qwen 27B multilingual -> test NL |
| scripts/yaml/eval_nl_zh/eval_qwen27b_multilingual_zh.yaml | Qwen 27B multilingual -> test ZH |
| scripts/yaml/eval_nl_zh/eval_qwen27b_orig_nl.yaml | Qwen 27B original -> test NL |
| scripts/yaml/eval_nl_zh/eval_qwen27b_orig_zh.yaml | Qwen 27B original -> test ZH |
| scripts/yaml/eval_nl_zh/eval_qwen9b_multilingual_nl.yaml | Qwen 9B multilingual -> test NL |
| scripts/yaml/eval_nl_zh/eval_qwen9b_multilingual_zh.yaml | Qwen 9B multilingual -> test ZH |

### Unused / Templates

| File | Description |
|------|-------------|
| scripts/yaml/exp006_template.yaml | CoT training template (not used directly) |
| scripts/yaml/eval_template.yaml | Eval template (not used directly) |

---

## results/ — Evaluation Results

### Baselines

| File | Description |
|------|-------------|
| results/baseline_roberta/roberta_results.csv | RoBERTa-base evaluation results (Macro F1=0.5689) |
| results/baseline_fewshot/gemma-4-31B-it_results.csv | Gemma 4 few-shot results |
| results/baseline_fewshot/Qwen3.5-27B_results.csv | Qwen 27B few-shot results |
| results/baseline_fewshot/Qwen3.5-9B_results.csv | Qwen 9B few-shot results |

### LoRA Results

| File | Description |
|------|-------------|
| results/lora/summary.csv | **All experiments Macro F1 summary** |
| results/lora/summary_clean.csv | Cleaned summary table |
| results/lora/gemma4-31b-cot_climate_test_results.csv | Gemma4 CoT climate OOD per-sample results |
| results/lora/gemma4-31b-cot_climate_test_results.jsonl | Gemma4 CoT climate OOD JSONL |
| results/lora/qwen3.5-27b-ablation-orig_edu_test_results.jsonl | Qwen 27B ablation edu test JSONL |


