import os
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import torch
import evaluate
import wandb
from dotenv import load_dotenv

from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Audio

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from accelerate import Accelerator
from stt.multi_gpu.augmentation import AudioAugmentation

load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "whisper-uzbek-stt")

MODEL_ID = "openai/whisper-large-v3"

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

accelerator = Accelerator()

if accelerator.is_main_process:
    print(f"Training with {accelerator.num_processes} processes")

torch.backends.cudnn.benchmark = True

if accelerator.is_main_process and HF_TOKEN:
    login(token=HF_TOKEN)

if accelerator.is_main_process and WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=WANDB_PROJECT, name="whisper-uzbek-stt-accelerate")

audio_augmenter = AudioAugmentation()

with accelerator.main_process_first():
    processor = WhisperProcessor.from_pretrained(
        MODEL_ID, 
        language="uzbek",
        task="transcribe"
    )
    
    print("Loading local dataset from metadata.json...")
    try:
        dataset = load_dataset("json", data_files="metadata.json", split="train")
        dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
        
        split_ds = dataset.train_test_split(test_size=0.1, seed=42)
        common_voice = DatasetDict({
            "train": split_ds["train"],
            "test": split_ds["test"]
        })
        
        if accelerator.is_main_process:
            print(f"Dataset loaded: {len(common_voice['train'])} train, {len(common_voice['test'])} test samples")
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

accelerator.wait_for_everyone()

def prepare_dataset(batch, apply_augmentation=True):
    audio = batch["audio_path"]
    audio_array = audio["array"]
    
    if len(audio_array) == 0:
        audio_array = np.zeros(16000)
    
    if apply_augmentation and np.random.random() < 0.3:
        try:
            audio_array = audio_augmenter.add_noise(audio_array)
        except Exception:
            pass
    
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

columns_to_remove = [col for col in common_voice["train"].column_names if col not in ["audio_path", "text"]]
if columns_to_remove:
    common_voice = common_voice.remove_columns(columns_to_remove)

print("Preparing datasets...")
train_dataset = common_voice["train"].map(
    lambda batch: prepare_dataset(batch, apply_augmentation=True),
    remove_columns=common_voice["train"].column_names,
    num_proc=1
)

test_dataset = common_voice["test"].map(
    lambda batch: prepare_dataset(batch, apply_augmentation=False),
    remove_columns=common_voice["test"].column_names,
    num_proc=1
)

common_voice = DatasetDict({
    "train": train_dataset,
    "test": test_dataset
})

model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

model.generation_config.language = "uzbek"
model.generation_config.task = "transcribe"
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
        
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
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
    output_dir="./whisper-uzbek-stt-accelerate",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=2000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=500,
    eval_steps=500,
    logging_steps=25,
    report_to=["wandb"] if accelerator.is_main_process else [],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    save_total_limit=3,
    seed=42,
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

if accelerator.is_main_process:
    print(f"Starting training with {len(common_voice['train'])} samples...")

trainer.train()

if accelerator.is_main_process:
    print("Saving final model...")
    trainer.save_model("./final_model_accelerate")
    processor.save_pretrained("./final_model_accelerate")

accelerator.wait_for_everyone()

if accelerator.is_main_process:
    print("Training completed!")