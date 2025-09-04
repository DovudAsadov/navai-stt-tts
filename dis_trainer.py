import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

import torch
import evaluate
from dotenv import load_dotenv

from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Audio

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    get_linear_schedule_with_warmup,
)

# =========================
# Config & env
# =========================
load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "whisper-uzbek-stt")

MODEL_ID = "openai/whisper-large-v3"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# optional: fewer CuDNN autotuner surprises with variable-length audio
torch.backends.cudnn.benchmark = False

if HF_TOKEN:
    try:
        login(token=HF_TOKEN)
    except Exception as e:
        print(f"[warn] HF login failed (continuing): {e}", file=sys.stderr)

# =========================
# Data loading
# Expect metadata.json lines like:
# {"audio_path": "path/to/file.wav", "text": "transcript ..."}
# =========================
print("Loading local dataset from metadata.json...")
try:
    dataset = load_dataset("json", data_files="metadata.json", split="train")
    # cast path -> Audio
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))

    split_ds = dataset.train_test_split(test_size=0.1, seed=42)
    ds = DatasetDict({"train": split_ds["train"], "test": split_ds["test"]})
    print(f"Dataset loaded: {len(ds['train'])} train / {len(ds['test'])} test")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

# =========================
# Processor / model
# =========================
processor = WhisperProcessor.from_pretrained(
    MODEL_ID,
    language="uzbek",
    task="transcribe",
)

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Recommended when using grad checkpointing
model.config.use_cache = False

# Set generation config (avoid forcing decoder prompt during training)
model.generation_config.language = "uzbek"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# Enable gradient checkpointing - but disable use_cache first
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()
    # Ensure gradient checkpointing is properly configured
    model.config.gradient_checkpointing = True

# =========================
# (Optional) simple augmentation
# =========================
def maybe_add_noise(waveform: np.ndarray, p: float = 0.3) -> np.ndarray:
    if np.random.random() >= p:
        return waveform
    if waveform.size == 0:
        return waveform
    noise = np.random.normal(0, 0.005, size=waveform.shape).astype(np.float32)
    return (waveform.astype(np.float32) + noise).astype(np.float32)

# =========================
# Preprocess
# =========================
def prepare_example(batch: Dict[str, Any], do_aug: bool) -> Dict[str, Any]:
    audio = batch["audio_path"]
    arr = audio["array"]
    sr = audio["sampling_rate"]

    # guard empty
    if arr is None or len(arr) == 0:
        arr = np.zeros(16000, dtype=np.float32)
        sr = 16000

    # optional aug on train
    if do_aug:
        arr = maybe_add_noise(arr, p=0.3)

    # extract log-mel input features
    input_features = processor.feature_extractor(
        arr, sampling_rate=sr
    ).input_features

    if isinstance(input_features, list):
        input_features = input_features[0]

    # ensure (#mels, T)
    if input_features.ndim == 1:
        input_features = np.reshape(input_features, (80, -1))
    elif input_features.ndim > 2:
        input_features = np.squeeze(input_features)

    # labels
    labels = processor.tokenizer(batch["text"]).input_ids

    return {
        "input_features": input_features,
        "labels": labels,
    }

# keep only needed cols before mapping for speed
keep_cols = {"audio_path", "text"}
drop_cols = [c for c in ds["train"].column_names if c not in keep_cols]
if drop_cols:
    ds = ds.remove_columns(drop_cols)

print("Preparing train set...")
train_ds = ds["train"].map(
    lambda ex: prepare_example(ex, do_aug=True),
    remove_columns=ds["train"].column_names,
)

print("Preparing eval set...")
eval_ds = ds["test"].map(
    lambda ex: prepare_example(ex, do_aug=False),
    remove_columns=ds["test"].column_names,
)

# =========================
# Data collator
# =========================
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], np.ndarray]]]) -> Dict[str, torch.Tensor]:
        # pad inputs
        inputs = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(inputs, return_tensors="pt")

        # pad labels
        labels_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(labels_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # strip BOS if present everywhere
        if labels.size(1) > 0:
            if (labels[:, 0] == self.decoder_start_token_id).all().item():
                labels = labels[:, 1:]

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor, decoder_start_token_id=model.config.decoder_start_token_id
)

# =========================
# Metrics
# =========================
metric_wer = evaluate.load("wer")

def compute_metrics(pred):
    # predictions are sequences when predict_with_generate=True
    pred_ids = pred.predictions
    label_ids = pred.label_ids.copy()

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# =========================
# Training args
# Scale batch/accumulation to your VRAM & GPU count.
# Multi-GPU is handled by `accelerate launch ...`
# =========================
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-uzbek-stt",
    per_device_train_batch_size=2,  # Further reduced batch size for stability
    per_device_eval_batch_size=2,   # Further reduced batch size
    gradient_accumulation_steps=8,  # Increased accumulation to maintain effective batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=500,
    save_steps=500,
    logging_steps=25,
    predict_with_generate=True,
    generation_max_length=225,
    report_to=(["wandb"] if WANDB_API_KEY else []),
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    remove_unused_columns=False,
    dataloader_num_workers=0,  # Disable multiprocessing for stability
    dataloader_pin_memory=False,
    save_total_limit=3,
    seed=42,
    ddp_find_unused_parameters=False,
    optim="adamw_torch",  # Explicit optimizer
    max_grad_norm=1.0,    # Gradient clipping
)


# =========================
# Trainer
# =========================
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,  # Use processing_class instead of tokenizer
)

# (Optional) W&B auth just once on main process
if WANDB_API_KEY and int(os.environ.get("RANK", "0")) == 0:
    try:
        import wandb
        wandb.login(key=WANDB_API_KEY)
        wandb.init(project=WANDB_PROJECT, name="whisper-uzbek-stt-trainer")
    except Exception as e:
        print(f"[warn] wandb init failed (continuing): {e}", file=sys.stderr)

if trainer.is_world_process_zero:
    print(f"Starting training with {len(train_ds)} samples...")

trainer.train()

if trainer.is_world_process_zero:
    print("Saving final model...")
    trainer.save_model("./final_model")
    processor.save_pretrained("./final_model")

if trainer.is_world_process_zero:
    print("Done.")
