#!/bin/bash

echo "🚀 Starting Multi-GPU Whisper Training with Accelerate"
echo "================================================"

if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "⚠️  Virtual environment not detected."
    echo "Attempting to activate virtual environment..."
    if [[ -f ".venv/bin/activate" ]]; then
        source .venv/bin/activate
        echo "✅ Virtual environment activated"
    else
        echo "❌ Virtual environment not found. Please create and activate it first."
        exit 1
    fi
fi

if [[ ! -f "metadata.json" ]]; then
    echo "❌ metadata.json not found!"
    echo "Please ensure metadata.json exists in the root directory."
    exit 1
fi

if [[ ! -d "audio_files" ]]; then
    echo "❌ audio_files directory not found!"
    echo "Please ensure audio_files directory exists."
    exit 1
fi

echo "🔧 Configuring Accelerate for multi-GPU training..."
accelerate config --config_file distributed_training/accelerate_config.yaml

echo "🏃 Starting multi-GPU training..."
cd distributed_training
accelerate launch --config_file accelerate_config.yaml train_accelerate.py

echo "🎉 Training completed!"