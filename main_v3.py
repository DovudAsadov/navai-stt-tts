import os
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional

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

from augmentation import AudioAugmentation

load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "whisper-uzbek-stt")

# MODEL_ID = "openai/whisper-large-v3"
MODEL_ID = "openai/whisper-tiny"

if HF_TOKEN:
    login(token=HF_TOKEN)

if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
wandb.init(project=WANDB_PROJECT, name="whisper-uzbek-stt")

# Initialize audio augmentation
audio_augmenter = AudioAugmentation()

print("Loading local dataset from metadata.json...")
try:
    # Load dataset from local metadata.json file
    dataset = load_dataset("json", data_files="metadata.json", split="train")
    
    # Cast audio column to Audio feature for automatic audio loading
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    
    # Split dataset into train and test
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

# Remove any unwanted columns (keep only audio_path and text)
columns_to_remove = [col for col in common_voice["train"].column_names if col not in ["audio_path", "text"]]
if columns_to_remove:
    common_voice = common_voice.remove_columns(columns_to_remove)

# Initialize processor and model
print("Loading Whisper processor and model...")
processor = WhisperProcessor.from_pretrained(
    MODEL_ID, 
    language="uzbek",
    task="transcribe"
)

def prepare_dataset(batch, apply_augmentation=True):
    """Prepare dataset for training/evaluation"""
    audio = batch["audio_path"]  # This will be the audio data from the Audio feature
    audio_array = audio["array"]
    
    # Ensure audio is not empty and has valid shape
    if len(audio_array) == 0:
        # Pad with zeros if empty audio
        audio_array = np.zeros(16000)  # 1 second of silence
    
    # Apply augmentation during training
    if apply_augmentation and np.random.random() < 0.3: 
        try:
            audio_array = audio_augmenter.add_noise(audio_array)
        except Exception:
            # If augmentation fails, use original audio
            pass
    
    # Compute log-Mel input features
    input_features = processor.feature_extractor(
        audio_array, sampling_rate=audio["sampling_rate"]
    ).input_features
    
    # Handle the feature extractor output properly
    if isinstance(input_features, list):
        # If it's a list, take the first element
        batch["input_features"] = input_features[0]
    else:
        # If it's already an array, use it directly
        batch["input_features"] = input_features
    
    # Ensure the shape is correct (should be [n_mels, time_steps])
    if len(batch["input_features"].shape) > 2:
        # Remove extra dimensions if present
        batch["input_features"] = batch["input_features"].squeeze()
    
    # Encode target text to label ids
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    
    return batch

print("Preparing training dataset...")
train_dataset = common_voice["train"].map(
    lambda batch: prepare_dataset(batch, apply_augmentation=True),
    remove_columns=common_voice["train"].column_names,
    num_proc=1  # Reduced to avoid multiprocessing issues
)

print("Preparing test dataset...")
test_dataset = common_voice["test"].map(
    lambda batch: prepare_dataset(batch, apply_augmentation=False),
    remove_columns=common_voice["test"].column_names,
    num_proc=1  # Reduced to avoid multiprocessing issues
)

common_voice = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

# Verify the prepared dataset
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
    print("✅ Dataset verification passed")
except Exception as e:
    print(f"❌ Dataset verification failed: {e}")
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
model.generation_config.language = "uzbek"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract input features and ensure they're properly shaped
        input_features = []
        for feature in features:
            input_feat = feature["input_features"]
            # Convert to numpy if it's not already
            if isinstance(input_feat, list):
                input_feat = np.array(input_feat)
            # Ensure it's 2D [n_mels, time_steps]
            if len(input_feat.shape) == 1:
                # If 1D, reshape assuming it's flattened mel spectrogram
                input_feat = input_feat.reshape(80, -1)
            elif len(input_feat.shape) > 2:
                # If more than 2D, squeeze extra dimensions
                input_feat = input_feat.squeeze()
            input_features.append({"input_features": input_feat})
        
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

# Training arguments optimized for local dataset
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-uzbek-stt-local",
    per_device_train_batch_size=4,  # Reduced batch size
    per_device_eval_batch_size=4,   # Reduced batch size
    gradient_accumulation_steps=4,  # Increased to maintain effective batch size
    learning_rate=1e-5,
    warmup_steps=50,  # Reduced for small dataset
    max_steps=500,    # Reduced for small dataset
    gradient_checkpointing=False,  # Disabled to fix the error
    fp16=True,
    predict_with_generate=True,
    generation_max_length=225,
    save_strategy="steps",
    save_steps=100,   # More frequent saves
    eval_strategy="steps",
    eval_steps=100,   # More frequent eval
    logging_steps=10,
    report_to=["wandb"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    save_total_limit=3,
    seed=42,
    # Additional fixes for small dataset training
    dataloader_num_workers=0,  # Disable multiprocessing for data loading
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
print(f"Model will be saved to: {training_args.output_dir}")

# Start training
trainer.train()

# Save final model locally
print("Saving final model locally...")
trainer.save_model("./final_model_local")
processor.save_pretrained("./final_model_local")

# Optional: Push to hub with correct metadata
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
        print("✅ Model successfully pushed to Hugging Face Hub!")
    except Exception as e:
        print(f"❌ Error pushing to hub: {e}")

print("✅ Training completed!")
print(f"Final model saved in: ./final_model_local")
