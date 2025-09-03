# train_whisper_accelerate.py
import os
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union, Optional
import multiprocessing

import torch
import evaluate
import wandb
from dotenv import load_dotenv

from huggingface_hub import login
from datasets import load_dataset, DatasetDict, Audio, concatenate_datasets, load_from_disk
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

from accelerate import Accelerator

# your augmentation module (unchanged)
from augmentation import AudioAugmentation

load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
WANDB_PROJECT = os.getenv("WANDB_PROJECT", "whisper-uzbek-stt")

MODEL_ID = "openai/whisper-large-v3"

# reduce tokenizer parallelism noisy warnings / oversubscription
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Instantiate accelerator early so we can use it to gate downloads/preprocessing
accelerator = Accelerator()  # respects accelerate launch config
num_processes = accelerator.num_processes if accelerator.use_distributed else 1
local_process_index = accelerator.process_index

# Helpful: print only on main process
if accelerator.is_main_process:
    print(f"Accelerate: num_processes={num_processes}, process_index={local_process_index}")

# Set some general PyTorch performance flags (optional but helpful for throughput)
torch.backends.cudnn.benchmark = True

# Only main process should login / init W&B to avoid duplicate logins
if accelerator.is_main_process and HF_TOKEN:
    login(token=HF_TOKEN)

if accelerator.is_main_process and WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY)
    wandb.init(project=WANDB_PROJECT, name="whisper-uzbek-stt")

# initialize audio augmentation (safe to instantiate in every process)
audio_augmenter = AudioAugmentation()

# ---------- Dataset preprocessing (done once, cached to disk) ----------
processed_cache_dir = "./processed_common_voice_cache"

# We'll create the processor on the main process first to avoid repeated downloads;
# other processes will instantiate after the main finished (main_process_first).
processor = None

with accelerator.main_process_first():  # main process does heavy downloads / preprocessing first
    # load processor (main process downloads cache); other processes will wait here until done
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="uzbek", task="transcribe")

    if os.path.exists(processed_cache_dir):
        print("Found cached processed dataset; loading from disk (main process).")
        common_voice = load_from_disk(processed_cache_dir)
    else:
        print("Loading raw datasets and preprocessing (main process). This may take a while...")

        try:
            ds_podcast = load_dataset("islomov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset", split="train")
            ds_news = load_dataset("islomov/news_youtube_uzbek_speech_dataset", split="train")
            ds_it = load_dataset("islomov/it_youtube_uzbek_speech_dataset", split="train")

            combined_ds = concatenate_datasets([ds_podcast, ds_news, ds_it])
            combined_ds = combined_ds.shuffle(seed=42)

            split_ds = combined_ds.train_test_split(test_size=0.1, seed=42)

            common_voice = DatasetDict({"train": split_ds["train"], "test": split_ds["test"]})

            print(f"Dataset loaded successfully (main): {common_voice}")

        except Exception as e:
            print(f"Error loading datasets on main process: {e}")
            raise

    # If dataset is freshly created (not loaded from cache) — do casting & mapping and save to disk
    if not os.path.exists(processed_cache_dir):
        if "id" in common_voice["train"].column_names:
            common_voice = common_voice.remove_columns(["id"])

        # ensure 16k sampling
        common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

        # define prepare function here so it captures the processor variable available in main
        def prepare_dataset(batch, apply_augmentation=True):
            audio = batch["audio"]
            audio_array = audio["array"]

            # augmentation with probability 0.3
            if apply_augmentation and np.random.random() < 0.3:
                audio_array = audio_augmenter.apply_random_augmentation(audio_array)

            # compute input features (log-Mel features)
            batch["input_features"] = processor.feature_extractor(
                audio_array, sampling_rate=audio["sampling_rate"]
            ).input_features[0]

            # encode target into token ids
            batch["labels"] = processor.tokenizer(batch["sentence"]).input_ids

            return batch

        # Choose num_proc reasonably: don't oversubscribe CPU (use half of cores or a cap)
        cpu_count = multiprocessing.cpu_count() or 1
        map_num_proc = max(1, min(8, cpu_count // 2))

        print(f"Mapping train dataset with num_proc={map_num_proc} (main process).")
        train_dataset = common_voice["train"].map(
            lambda b: prepare_dataset(b, apply_augmentation=True),
            remove_columns=common_voice["train"].column_names,
            num_proc=map_num_proc,
        )

        print(f"Mapping test dataset with num_proc={map_num_proc} (main process).")
        test_dataset = common_voice["test"].map(
            lambda b: prepare_dataset(b, apply_augmentation=False),
            remove_columns=common_voice["test"].column_names,
            num_proc=map_num_proc,
        )

        common_voice = DatasetDict({"train": train_dataset, "test": test_dataset})

        # save processed dataset to disk so other processes (or later runs) reuse it
        print("Saving processed dataset to disk (main process)...")
        common_voice.save_to_disk(processed_cache_dir)

# Wait for everyone to reach this point (main saved the cached processed data)
accelerator.wait_for_everyone()

# If this process didn't create 'common_voice' above, load the cached processed dataset
if "common_voice" not in globals() or not isinstance(common_voice, DatasetDict):
    print("Loading processed dataset from disk (worker process).")
    common_voice = load_from_disk(processed_cache_dir)

# Make sure the processor variable exists in all processes
if processor is None:
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="uzbek", task="transcribe")

# Remove any stray 'id' column if present (safe no-op)
if "id" in common_voice["train"].column_names:
    common_voice = common_voice.remove_columns(["id"])

# ---------- Model ----------
# Load model on each process (Trainer + accelerate will handle DDP)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)

# Set generation config
model.generation_config.language = "uzbek"
model.generation_config.task = "transcribe"
model.generation_config.forced_decoder_ids = None

# ---------- Data collator (unchanged but same class) ----------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # mask padding tokens with -100 for loss
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # Remove BOS token (whisper decoder_start_token_id) if all rows start with it
        try:
            if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
        except Exception:
            # fallback: if shape mismatch or empty, ignore
            pass

        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

# ---------- Metric ----------
metric = evaluate.load("wer")

def compute_metrics(pred):
    # For predict_with_generate=True Seq2SeqTrainer returns generated token IDs in predictions
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad_token_id
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# ---------- TrainingArguments ----------
# Tune dataloader workers according to CPU and number of processes
cpu_count = multiprocessing.cpu_count() or 1
dataloader_num_workers = max(2, min(16, cpu_count // max(1, num_processes)))

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-uzbek-stt",
    per_device_train_batch_size=16,           # per-GPU batch size
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=5000,
    gradient_checkpointing=True,
    fp16=True,                                # will work with accelerate mixed precision as well
    evaluation_strategy="steps",
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["wandb"] if accelerator.is_main_process else [],  # only main reports to wandb
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=False,
    dataloader_num_workers=dataloader_num_workers,
    eval_accumulation_steps=4,
)

# ---------- Trainer ----------
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor,  # newer Trainer API uses processing_class for multimodal processors
)

print("Starting training...")
trainer.train()

# Only main process should save and push
if accelerator.is_main_process:
    print("Saving final model locally (main process)...")
    trainer.save_model("./final_model")
    processor.save_pretrained("./final_model")

    # push to hub only from main process (if you want)
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

    try:
        print("Pushing model to hub (main process)...")
        trainer.push_to_hub(**kwargs)
    except Exception as e:
        print(f"Push to hub failed (main): {e}")

# Make sure all processes finish before exiting
accelerator.wait_for_everyone()
print("Training script finished (process {})".format(accelerator.process_index))
