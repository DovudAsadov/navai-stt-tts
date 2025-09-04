import os
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

import torch
import evaluate
import wandb
from dotenv import load_dotenv

from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from stt.multi_gpu.augmentation import AudioAugmentation

load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "whisper-uzbek-stt")

MODEL_ID = "openai/whisper-large-v3"

if HF_TOKEN:
    login(token=HF_TOKEN)

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
wandb.init(project=WANDB_PROJECT, name="whisper-uzbek-stt")

# Initialize audio augmentation
audio_augmenter = AudioAugmentation()

print("Loading datasets...")
try:
    ds_podcast = load_dataset("islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset", split="train")
    ds_news = load_dataset("islomov/news_youtube_uzbek_speech_dataset", split="train")
    ds_it = load_dataset("islomov/it_youtube_uzbek_speech_dataset", split="train")
    
    # Combine datasets
    combined_ds = concatenate_datasets([ds_podcast, ds_news, ds_it])
    combined_ds = combined_ds.shuffle(seed=42)
    
    # Split dataset
    split_ds = combined_ds.train_test_split(test_size=0.1, seed=42)
    
    common_voice = DatasetDict({
        "train": split_ds["train"],
        "test": split_ds["test"]
    })
    
    print(f"Dataset loaded successfully: {common_voice}")
    
except Exception as e:
    print(f"Error loading datasets: {e}")
    exit(1)


if "id" in common_voice["train"].column_names:
    common_voice = common_voice.remove_columns(["id"])


# Initialize processor and model
print("Loading Whisper processor and model...")
processor = WhisperProcessor.from_pretrained(
    MODEL_ID, 
    language="uzbek",
    task="transcribe"
)

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

def prepare_dataset(batch, apply_augmentation=True):
    audio = batch["audio"]
    audio_array = audio["array"]
    
    # Apply augmentation during training
    if apply_augmentation and np.random.random() < 0.3: 
        audio_array = audio_augmenter.apply_random_augmentation(audio_array)
    
    # Compute log-Mel input features
    batch["input_features"] = processor.feature_extractor(
        audio_array, sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    
    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids
    
    return batch

print("Preparing training dataset...")
train_dataset = common_voice["train"].map(
    lambda batch: prepare_dataset(batch, apply_augmentation=True),
    remove_columns=common_voice["train"].column_names,
    num_proc=4
)

print("Preparing test dataset...")
test_dataset = common_voice["test"].map(
    lambda batch: prepare_dataset(batch, apply_augmentation=False),
    remove_columns=common_voice["test"].column_names,
    num_proc=4
)

common_voice = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Set model configurationv
model.generation_config.language = "uzbek"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Get tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
    
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# Initialize evaluation metric
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with the pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode predictions and labels
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Calculate WER
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-uzbek-stt",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,
)

print("Starting training...")
trainer.train()


# Save final model locally
print("Saving model locally...")
trainer.save_model("./final_model")
processor.save_pretrained("./final_model")


# Push to hub with correct metadata
kwargs = {
    "dataset_tags": ["uzbek", "speech-recognition"],
    "dataset": "Uzbek Speech Datasets",
    "dataset_args": "Combined: podcasts, news",
    "language": "uz",
    "model_name": "Whisper Large Uzbek",
    "finetuned_from": "openai/whisper-large",
    "tasks": "automatic-speech-recognition",
    "tags": ["whisper", "uzbek", "speech-recognition"],
}

print("Pushing model to hub...")
trainer.push_to_hub(**kwargs)