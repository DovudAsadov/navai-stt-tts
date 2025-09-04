#!/bin/bash

# Training script for Whisper with local dataset
# Run this script to start training with your local metadata.json

echo "🚀 Starting Whisper Training with Local Dataset"
echo "================================================"

# Check if virtual environment is activated
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

# Check if dataset exists
if [[ ! -f "metadata.json" ]]; then
    echo "❌ metadata.json not found!"
    echo "Please run the dataset preparation script first:"
    echo "   python prepare_dataset.py"
    exit 1
fi

# Check if audio files directory exists
if [[ ! -d "audio_files" ]]; then
    echo "❌ audio_files directory not found!"
    echo "Please run the dataset preparation script first:"
    echo "   python prepare_dataset.py"
    exit 1
fi

# Test dataset loading
echo "🔍 Testing dataset loading..."
python distributed_training/test_dataset.py

if [[ $? -ne 0 ]]; then
    echo "❌ Dataset test failed. Please check your data."
    exit 1
fi

echo "✅ Dataset test passed!"

# Ask for confirmation
read -p "Do you want to start training? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled."
    exit 0
fi

# Start training
echo "🏃 Starting training..."
echo "This may take a while depending on your dataset size and GPU..."

# Set CUDA device if available
if command -v nvidia-smi &> /dev/null; then
    echo "🔧 CUDA detected. Setting up GPU training..."
    export CUDA_VISIBLE_DEVICES=0
fi

# Run training
python distributed_training/main_v3.py

echo "🎉 Training script completed!"
echo "Check the output above for any errors or the final model location."
