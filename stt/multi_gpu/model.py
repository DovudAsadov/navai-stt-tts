import os
import sys
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
from pathlib import Path

import torch
import evaluate
import wandb
from dotenv import load_dotenv

from huggingface_hub import login
from datasets import load_dataset, Dataset, DatasetDict, Audio, Features, Value

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from stt.multi_gpu.augmentation import AudioAugmentation

# Add project root to Python path and import config
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT))
from config import TrainingConfig

# Initialize configuration
config = TrainingConfig()

# Print configuration summary
print("🚀 Whisper Multi-GPU Training Configuration")
print("=" * 50)
print(f"📁 Metadata file: {config.METADATA_FILE}")
print(f"🤖 Model ID: {config.MODEL_ID}")
print(f"🌍 Language: {config.LANGUAGE}")
print(f"🎯 Task: {config.TASK}")
print(f"📊 Output directory: {config.OUTPUT_DIR}")
print(f"🔄 Batch size (train/eval): {config.PER_DEVICE_TRAIN_BATCH_SIZE}/{config.PER_DEVICE_EVAL_BATCH_SIZE}")
print(f"📈 Learning rate: {config.LEARNING_RATE}")
print(f"🔥 Warmup steps: {config.WARMUP_STEPS}")
print("=" * 50)

load_dotenv()

# Get environment variables from config
env_vars = config.get_env_vars()
HF_TOKEN = os.getenv(config.HF_TOKEN_ENV)
WANDB_API_KEY = os.getenv(config.WANDB_API_KEY_ENV)
WANDB_PROJECT = env_vars["wandb_project"]

# Use model ID from config
MODEL_ID = config.MODEL_ID

if HF_TOKEN:
    login(token=HF_TOKEN)

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
wandb.init(project=WANDB_PROJECT, name="whisper-uzbek-stt")

audio_augmenter = AudioAugmentation()

print("Loading local dataset from metadata.json...")
print(f"Using metadata file: {config.METADATA_FILE}")
try:
    # Use config for metadata file path
    dataset = load_dataset("json", data_files=str(config.METADATA_FILE), split="train")
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    split_ds = dataset.train_test_split(test_size=0.1, seed=42)
    
    common_voice = DatasetDict({
        "train": split_ds["train"],
        "test": split_ds["test"]
    })
    
    print(f"Dataset loaded successfully: {common_voice}")
    print(f"Train samples: {len(common_voice['train'])}")
    print(f"Test samples: {len(common_voice['test'])}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)

columns_to_remove = [col for col in common_voice["train"].column_names if col not in ["audio_path", "text"]]
if columns_to_remove:
    common_voice = common_voice.remove_columns(columns_to_remove)

print("Loading Whisper processor and model...")
processor = WhisperProcessor.from_pretrained(
    MODEL_ID, 
    language=config.LANGUAGE,
    task=config.TASK
)

def prepare_dataset(batch, apply_augmentation=True):
    """Prepare dataset for training/evaluation"""
    audio = batch["audio_path"]
    audio_array = audio["array"]
    
    if len(audio_array) == 0:
        audio_array = np.zeros(16000)
    
    if apply_augmentation and np.random.random() < 0.3: 
        try:
            audio_array = audio_augmenter.add_noise(audio_array)
        except Exception:
            pass
    
    # Compute log-Mel input features
    input_features = processor.feature_extractor(
        audio_array, sampling_rate=audio["sampling_rate"]
    ).input_features
    
    if isinstance(input_features, list):
        batch["input_features"] = input_features[0]
    else:
        batch["input_features"] = input_features
    
    if len(batch["input_features"].shape) > 2:
        batch["input_features"] = batch["input_features"].squeeze()
    
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch

print("Preparing training dataset...")
train_dataset = common_voice["train"].map(
    lambda batch: prepare_dataset(batch, apply_augmentation=True),
    remove_columns=common_voice["train"].column_names,
    num_proc=1 
)

print("Preparing test dataset...")
test_dataset = common_voice["test"].map(
    lambda batch: prepare_dataset(batch, apply_augmentation=False),
    remove_columns=common_voice["test"].column_names,
    num_proc=1
)

common_voice = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

print("Verifying prepared dataset...")
try:
    sample = common_voice["train"][0]
    input_features = sample['input_features']
    
    # Handle both numpy arrays and tensors
    if hasattr(input_features, 'shape'):
        print(f"Sample input_features shape: {input_features.shape}")
    elif isinstance(input_features, list):
        print(f"Sample input_features length: {len(input_features)}")
        if len(input_features) > 0 and hasattr(input_features[0], 'shape'):
            print(f"Sample input_features[0] shape: {input_features[0].shape}")
    else:
        print(f"Sample input_features type: {type(input_features)}")
    
    print(f"Sample labels length: {len(sample['labels'])}")
    print("Dataset verification passed")
except Exception as e:
    print(f"Dataset verification failed: {e}")
    print("Debugging info:")
    try:
        sample = common_voice["train"][0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input features type: {type(sample['input_features'])}")
        if isinstance(sample['input_features'], list):
            print(f"Input features length: {len(sample['input_features'])}")
            if len(sample['input_features']) > 0:
                print(f"First element type: {type(sample['input_features'][0])}")
    except Exception as debug_e:
        print(f"Debug error: {debug_e}")
    exit(1)

print("Loading model...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Set model configuration
model.generation_config.language = config.LANGUAGE
model.generation_config.task = config.TASK
model.generation_config.forced_decoder_ids = None
    
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = []
        for feature in features:
            input_feat = feature["input_features"]
            if isinstance(input_feat, list):
                input_feat = np.array(input_feat)
            if len(input_feat.shape) == 1:
                input_feat = input_feat.reshape(80, -1)
            elif len(input_feat.shape) > 2:
                input_feat = input_feat.squeeze()
            input_features.append({"input_features": input_feat})
        
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt"
        )

        # Create attention mask for input features
        if "attention_mask" not in batch:
            batch["attention_mask"] = torch.ones(
                batch["input_features"].shape[:-1], dtype=torch.long
            )

        # Tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
            padding=True
        )

        # Replace padding with -100 so they're ignored in loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if present
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        batch["decoder_attention_mask"] = labels_batch["attention_mask"]  # NEW

        return batch



data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

training_args = Seq2SeqTrainingArguments(
    output_dir=config.OUTPUT_DIR,
    per_device_train_batch_size=config.PER_DEVICE_TRAIN_BATCH_SIZE,  
    per_device_eval_batch_size=config.PER_DEVICE_EVAL_BATCH_SIZE,   
    gradient_accumulation_steps=config.GRADIENT_ACCUMULATION_STEPS,
    learning_rate=config.LEARNING_RATE,
    warmup_steps=config.WARMUP_STEPS,  
    # max_steps=config.MAX_STEPS, 
    num_train_epochs=20,
    gradient_checkpointing=config.GRADIENT_CHECKPOINTING,
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
    save_strategy="steps",
    save_steps=config.SAVE_STEPS, 
    eval_strategy="steps",
    eval_steps=config.EVAL_STEPS,
    logging_steps=config.LOGGING_STEPS,
    report_to=["wandb"],
    load_best_model_at_end=config.LOAD_BEST_MODEL_AT_END,
    metric_for_best_model=config.METRIC_FOR_BEST_MODEL,
    greater_is_better=config.GREATER_IS_BETTER,
    push_to_hub=config.PUSH_TO_HUB,
    dataloader_pin_memory=config.DATALOADER_PIN_MEMORY,
    remove_unused_columns=config.REMOVE_UNUSED_COLUMNS,
    save_total_limit=config.SAVE_TOTAL_LIMIT,
    seed=config.SEED,
    dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
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

print(f"Starting training with {len(common_voice['train'])} training samples...")
print(f"Model will be saved to: {config.OUTPUT_DIR}")

# Start training
trainer.train()

# Save final model locally
print("Saving final model locally...")
trainer.save_model("./final_model_local")
processor.save_pretrained("./final_model_local")

if HF_TOKEN:
    kwargs = {
        "dataset_tags": ["uzbek", "speech-recognition"],
        "dataset": "Local Uzbek Speech Dataset",
        "dataset_args": "metadata.json based dataset",
        "language": "uz",
        "model_name": "Whisper Large Uzbek Local",
        "finetuned_from": "openai/whisper-large-v3",
        "tasks": "automatic-speech-recognition",
        "tags": ["whisper", "uzbek", "speech-recognition", "local-dataset"],
    }

    print("Pushing model to hub...")
    try:
        trainer.push_to_hub(**kwargs)
        print("Model successfully pushed to Hugging Face Hub!")
    except Exception as e:
        print(f"Error pushing to hub: {e}")

print("Training completed!")
print(f"Final model saved in: ./final_model_local")
