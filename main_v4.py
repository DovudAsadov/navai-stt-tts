# train_whisper_fixed.py
import os
import sys
import math
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate
from datasets import load_dataset, DatasetDict, Audio

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Optional: dotenv to load env vars if you have .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ----------------------
# Config (edit as needed)
# ----------------------
HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN", None)
WANDB_API_KEY = os.getenv("WANDB_API_KEY", None)
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "whisper-uzbek-stt")

# Choose model: tiny for quick iteration; switch to large-* when ready & you have resources.
MODEL_ID = os.getenv("MODEL_ID", "openai/whisper-tiny")

# ----------------------
# Optional logins
# ----------------------
if HF_TOKEN:
    try:
        from huggingface_hub import login as hf_login
        hf_login(token=HF_TOKEN)
    except Exception as e:
        print(f"[warn] HF login failed: {e}", file=sys.stderr)

if WANDB_API_KEY:
    try:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=WANDB_PROJECT, name="whisper-uzbek-stt")
    except Exception as e:
        print(f"[warn] wandb init failed: {e}", file=sys.stderr)

# ----------------------
# Load processor (feature_extractor + tokenizer)
# ----------------------
print("Loading processor...")
processor = WhisperProcessor.from_pretrained(MODEL_ID, language="uzbek", task="transcribe")

# ----------------------
# Load dataset
# Expect metadata.json with lines like: {"audio_path": "data/x.wav", "text": "transcript"}
# ----------------------
print("Loading dataset from metadata.json ...")
dataset = load_dataset("json", data_files="metadata.json", split="train")
# Cast the path column to Audio so huggingface will load arrays
dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))

# Keep only the columns we need (audio_path and text)
keep_cols = {"audio_path", "text"}
drop_cols = [c for c in dataset.column_names if c not in keep_cols]
if drop_cols:
    dataset = dataset.remove_columns(drop_cols)

# Split
split = dataset.train_test_split(test_size=0.1, seed=42)
ds = DatasetDict({"train": split["train"], "test": split["test"]})
print(f"Train samples: {len(ds['train'])}, Eval samples: {len(ds['test'])}")

# ----------------------
# Preprocess: create input_features (leave text as-is for collator to tokenize with fast tokenizer)
# ----------------------
def prepare_example(ex, apply_augment=False):
    audio = ex["audio_path"]
    arr = audio["array"]

    if arr is None or len(arr) == 0:
        arr = np.zeros(16000, dtype=np.float32)

    # (Optional) small augmentation example:
    if apply_augment and np.random.random() < 0.3:
        noise = np.random.normal(0, 0.002, size=arr.shape).astype(np.float32)
        arr = (arr.astype(np.float32) + noise).astype(np.float32)

    feats = processor.feature_extractor(arr, sampling_rate=audio["sampling_rate"]).input_features
    # feature_extractor may return a list; take first
    if isinstance(feats, list):
        feats = feats[0]

    # ensure numpy float32
    feats = np.asarray(feats, dtype=np.float32)

    # store input_features; DO NOT touch 'text' column (collator will tokenize text)
    return {"input_features": feats}

print("Preparing datasets (feature extraction)...")
ds["train"] = ds["train"].map(lambda x: prepare_example(x, apply_augment=True), remove_columns=[], num_proc=1)
ds["test"] = ds["test"].map(lambda x: prepare_example(x, apply_augment=False), remove_columns=[], num_proc=1)

# Keep dataset columns: input_features + text
# ----------------------
# Data collator: pads audio features and tokenizes text via tokenizer.__call__ (fast path)
# ----------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[np.ndarray, str]]]) -> Dict[str, torch.Tensor]:
        # features: list of dicts with keys "input_features" (np.ndarray) and "text" (str)
        input_feats = []
        texts = []
        for f in features:
            feat = f["input_features"]
            if isinstance(feat, list):
                feat = np.array(feat, dtype=np.float32)
            # ensure 2D (n_mels, time)
            if feat.ndim == 1:
                feat = feat.reshape(80, -1)
            elif feat.ndim > 2:
                feat = np.squeeze(feat)
            input_feats.append({"input_features": feat})
            texts.append(f["text"])

        # Pad audio features -> returns "input_features" tensor shaped (batch, n_mels, seq_len)
        batch = self.processor.feature_extractor.pad(input_feats, return_tensors="pt")

        # Create encoder attention mask for audio: ones across seq_len (some models expect this)
        seq_len = batch["input_features"].shape[-1]
        batch_size = batch["input_features"].shape[0]
        # shape (batch, seq_len)
        batch["attention_mask"] = torch.ones((batch_size, seq_len), dtype=torch.long)

        # Tokenize target text using fast tokenizer __call__ (faster than encode+pad)
        tokenized = self.processor.tokenizer(texts, padding=True, return_tensors="pt")

        labels = tokenized["input_ids"]
        # Replace padding token id's (from tokenizer attention mask) with -100 for loss ignore
        labels = labels.masked_fill(tokenized["attention_mask"].ne(1), -100)

        # If every sequence starts with decoder_start_token_id (BOS), remove it
        if labels.size(1) > 0 and (labels[:, 0] == self.decoder_start_token_id).all().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        # expose decoder_attention_mask for generation if needed
        batch["decoder_attention_mask"] = tokenized["attention_mask"]

        return batch

# ----------------------
# Model
# ----------------------
print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
# disable cached key/value when using gradient checkpointing or DDP
if hasattr(model.config, "use_cache"):
    model.config.use_cache = False

model.generation_config.language = "uzbek"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# ----------------------
# Instantiate collator
# ----------------------
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ----------------------
# Metrics
# ----------------------
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    # predictions are sequences when predict_with_generate=True; here we will not generate during training
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    # replace -100 with pad_token_id for decoding
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ----------------------
# Training arguments (tuned for small dataset + 2x A100)
# ----------------------
# NOTE: For small dataset, using num_train_epochs is simpler than max_steps.
per_device_batch = 4  # per GPU
num_gpus = int(os.environ.get("CUDA_VISIBLE_DEVICES", "").count(",") + 1) if os.environ.get("CUDA_VISIBLE_DEVICES") else 2
# effective batch = per_device_batch * num_gpus * gradient_accumulation_steps

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-uzbek-stt-fixed",
    per_device_train_batch_size=per_device_batch,
    per_device_eval_batch_size=per_device_batch,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_train_epochs=20,
    warmup_steps=50,
    gradient_checkpointing=False,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=10,
    predict_with_generate=False,   # disable expensive generation during training
    generation_max_length=100,
    report_to=["wandb"] if WANDB_API_KEY else [],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,   # we need "text" in collator
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    save_total_limit=3,
    seed=42,
)

# ----------------------
# Trainer
# ----------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=ds["train"],
    eval_dataset=ds["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.tokenizer,  # important for generation/decoding
)

# ----------------------
# Optional: shorter debug prints and a quick ETA
# ----------------------
print("=== Train config summary ===")
print(f"Model: {MODEL_ID}")
print(f"Train samples: {len(ds['train'])}, Eval: {len(ds['test'])}")
print(f"Per-device batch size: {per_device_batch}, GPUs (assumed): 2")
eff_batch = per_device_batch * 2 * training_args.gradient_accumulation_steps
steps_per_epoch = math.ceil(len(ds["train"]) / eff_batch)
print(f"Effective batch size: {eff_batch}, steps/epoch ≈ {steps_per_epoch}, total steps ≈ {steps_per_epoch * training_args.num_train_epochs}")

# ----------------------
# Launch training
# ----------------------
if __name__ == "__main__":
    # Optional: reduce DDP overhead if you know there are no unused params (use before accelerate launch)
    # export ACCELERATE_DDP_FIND_UNUSED_PARAMETERS=false
    print("Starting training...")
    trainer.train()
    print("Saving final model locally...")
    trainer.save_model("./final_model_fixed")
    processor.save_pretrained("./final_model_fixed")
    print("Done.")
