import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import pandas as pd
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer
import json
import time

# ========================
# Configuration
# ========================
MODEL_NAME = "unsloth/gemma-3-4b-it"
MODEL_SHORT = "gemma-3-4b-it"
MAX_SEQ_LENGTH = 2048

TRAIN_DATA_PATH = "/home/mshahidul/llm_safety/sc_project/data/processed/community_notes_cleaned_train.parquet"
TEST_DATA_PATH = "/home/mshahidul/llm_safety/sc_project/data/processed/community_notes_cleaned_test.parquet"
OUTPUT_DIR = "/home/mshahidul/llm_safety/sc_project/outputs/evaluations/finetuned_v2/gemma-3-4b-it"
MODEL_SAVE_DIR = "/home/mshahidul/llm_safety/sc_project/models/gemma-3-4b-it-lora"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================
# 1. Load and format data
# ========================
print(f"Loading data...")
train_df = pd.read_parquet(TRAIN_DATA_PATH)
test_df = pd.read_parquet(TEST_DATA_PATH)

print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

def format_chat_example(row):
    label_text = "helpful and accurate" if row["label"] == 1 else "not helpful or inaccurate"
    classification = row["classification"]
    messages = [
        {
            "role": "user",
            "content": (
                "You are a fact-checking assistant. Analyze the following community note "
                "summary and classify whether it is helpful and accurate or not helpful and inaccurate.\n\n"
                "Community note summary:\n"
                f"{row['summary']}\n\n"
                "Classification: " + classification + "\n\n"
                "Is this community note helpful and accurate? Answer with only 'helpful and accurate' or 'not helpful and inaccurate'."
            ),
        },
        {
            "role": "assistant",
            "content": label_text,
        },
    ]
    return {"messages": messages}

train_dataset = Dataset.from_list(train_df.apply(format_chat_example, axis=1).to_list())
test_dataset = Dataset.from_list(test_df.apply(format_chat_example, axis=1).to_list())

def apply_chat_template(examples):
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        texts.append(text)
    return {"text": texts}

# ========================
# 2. Load model
# ========================
print(f"\n--- Loading model: {MODEL_NAME} ---")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,
    load_in_4bit=True,
)

# ========================
# 3. Add LoRA adapters
# ========================
print("--- Adding LoRA adapters ---")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# ========================
# 4. Format datasets
# ========================
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

train_dataset = train_dataset.map(apply_chat_template, batched=True)
test_dataset = test_dataset.map(apply_chat_template, batched=True)

print("\n--- Sample formatted prompt (truncated) ---")
print(train_dataset[0]["text"][:500])

# ========================
# 5. Train
# ========================
print("\n--- Training ---")
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_ratio=0.05,
        num_train_epochs=3,
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.001,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUTPUT_DIR,
        report_to="none",
        save_strategy="epoch",
        eval_strategy="no",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    ),
)

start_time = time.time()
trainer_stats = trainer.train()
elapsed = time.time() - start_time

used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
print(f"\n{round(elapsed, 1)} seconds used for training.")
print(f"{round(elapsed/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")

metrics_path = os.path.join(OUTPUT_DIR, "training_metrics.json")
train_metrics = {k: float(v) for k, v in trainer_stats.metrics.items()}
train_metrics["elapsed_seconds"] = elapsed
with open(metrics_path, "w") as f:
    json.dump(train_metrics, f, indent=2)
print(f"Training metrics saved to {metrics_path}")

# ========================
# 6. Save LoRA adapters
# ========================
print(f"\n--- Saving LoRA adapters to {MODEL_SAVE_DIR} ---")
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print(f"LoRA adapters saved to {MODEL_SAVE_DIR}")

# ========================
# 7. Batch evaluation
# ========================
print("\n--- Evaluating on test set (batch) ---")
FastLanguageModel.for_inference(model)

BATCH_SIZE = 16
correct = 0
total = len(test_dataset)
predictions_log = []

test_prompts = []
test_expected = []
for sample in test_df.itertuples():
    label_text = "helpful and accurate" if sample.label == 1 else "not helpful or inaccurate"
    classification = sample.classification
    messages = [{
        "role": "user",
        "content": (
            "You are a fact-checking assistant. Analyze the following community note "
            "summary and classify whether it is helpful and accurate or not helpful and inaccurate.\n\n"
            "Community note summary:\n"
            f"{sample.summary}\n\n"
            "Classification: " + classification + "\n\n"
            "Is this community note helpful and accurate? Answer with only 'helpful and accurate' or 'not helpful and inaccurate'."
        ),
    }]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    test_prompts.append(prompt_text)
    test_expected.append(label_text)

for batch_start in range(0, total, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, total)
    prompts_batch = test_prompts[batch_start:batch_end]
    expected_batch = test_expected[batch_start:batch_end]

    inputs = tokenizer(prompts_batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LENGTH).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=32,
        use_cache=True,
        temperature=0.1,
    )

    for i, out_ids in enumerate(outputs):
        prompt_len = inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum().item()
        generated_ids = out_ids[prompt_len:]
        generated = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        expected = expected_batch[i]

        is_correct = False
        if expected == "helpful and accurate":
            is_correct = "helpful" in generated.lower() and "accurate" in generated.lower()
        else:
            is_correct = ("not helpful" in generated.lower() or "inaccurate" in generated.lower())

        if is_correct:
            correct += 1

        predictions_log.append({
            "index": batch_start + i,
            "generated": generated,
            "expected": expected,
            "correct": is_correct,
        })

accuracy = correct / total * 100
print(f"\nTest Accuracy: {correct}/{total} = {accuracy:.2f}%")

tp = sum(1 for p in predictions_log if p["correct"] and p["expected"] == "helpful and accurate")
fp = sum(1 for p in predictions_log if not p["correct"] and p["expected"] == "not helpful or inaccurate")
fn = sum(1 for p in predictions_log if not p["correct"] and p["expected"] == "helpful and accurate")
tn = sum(1 for p in predictions_log if p["correct"] and p["expected"] == "not helpful or inaccurate")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

preds_path = os.path.join(OUTPUT_DIR, "test_predictions.json")
with open(preds_path, "w") as f:
    json.dump({
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predictions": predictions_log,
    }, f, indent=2)

eval_report = {
    "model": MODEL_NAME,
    "max_seq_length": MAX_SEQ_LENGTH,
    "lora_r": 16,
    "lora_alpha": 16,
    "epochs": 3,
    "learning_rate": 2e-4,
    "batch_size": 2,
    "gradient_accumulation_steps": 4,
    "train_samples": len(train_dataset),
    "test_samples": total,
    "test_accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "training_metrics": train_metrics,
    "elapsed_seconds": elapsed,
}
report_path = os.path.join(OUTPUT_DIR, "evaluation_report.json")
with open(report_path, "w") as f:
    json.dump(eval_report, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}")
print("=== DONE ===")
