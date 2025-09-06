#!/bin/bash

# 🚀 Whisper Multi-GPU Training Setup and Launch Script
# This script installs UV, creates virtual environment, syncs dependencies,
# configures Accelerate, and launches multi-GPU training

set -e  # Exit on any error

echo "🚀 Whisper Multi-GPU Training Setup and Launch"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Load configuration from Python config file
CONFIG_FILE="config.py"
if [ -f "$CONFIG_FILE" ]; then
    print_info "Loading configuration from $CONFIG_FILE..."
    # Use Python to validate config and extract key values
    python3 -c "
from config import TrainingConfig
import sys

config = TrainingConfig()

# Validate paths
validation = config.validate_paths()
dataset_info = config.get_dataset_info()

print(f'TRAINING_SCRIPT={config.TRAINING_SCRIPT}')
print(f'METADATA_FILE={config.METADATA_FILE}')
print(f'DATA_DIR={config.DATA_DIR}')
print(f'OUTPUT_DIR={config.OUTPUT_DIR}')
print(f'MODEL_ID={config.MODEL_ID}')

# Check if critical files exist
if not config.TRAINING_SCRIPT.exists():
    print('ERROR_TRAINING_SCRIPT_NOT_FOUND=true')
    sys.exit(1)

if not dataset_info['metadata_exists']:
    print('WARNING_NO_METADATA=true')
else:
    print(f'SAMPLE_COUNT={dataset_info[\"sample_count\"]}')
" > /tmp/config_vars.sh
    
    if [ $? -eq 0 ]; then
        source /tmp/config_vars.sh
        print_status "Configuration loaded successfully"
        rm -f /tmp/config_vars.sh
    else
        print_error "Failed to load configuration from $CONFIG_FILE"
        exit 1
    fi
else
    print_error "Configuration file not found: $CONFIG_FILE"
    print_info "Please ensure config.py exists in the project root"
    exit 1
fi

# Check if we're in the correct directory
if [ ! -f "pyproject.toml" ]; then
    print_error "pyproject.toml not found. Please run this script from the project root directory."
    exit 1
fi

print_info "Current directory: $(pwd)"

# Step 1: Install UV if not already installed
print_info "Step 1: Installing UV package manager..."
if command -v uv &> /dev/null; then
    print_status "UV is already installed: $(uv --version)"
else
    print_info "Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
    source $HOME/.cargo/env
    print_status "UV installed successfully: $(uv --version)"
fi

# Step 2: Create virtual environment with UV
print_info "Step 2: Creating virtual environment with UV..."
if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists"
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing existing virtual environment..."
        rm -rf .venv
        uv venv
        print_status "Virtual environment recreated"
    else
        print_status "Using existing virtual environment"
    fi
else
    uv venv
    print_status "Virtual environment created"
fi

# Step 3: Sync dependencies with UV
print_info "Step 3: Installing dependencies with UV..."
uv sync
print_status "Dependencies installed successfully"

# Step 4: Activate virtual environment
print_info "Step 4: Activating virtual environment..."
source .venv/bin/activate
print_status "Virtual environment activated"

# Step 5: Configure Accelerate
print_info "Step 5: Configuring Accelerate for multi-GPU training..."

# Check if accelerate config already exists
if [ -f "accelerate_config.yaml" ]; then
    print_warning "Accelerate config already exists"
    read -p "Do you want to reconfigure it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Running accelerate config..."
        accelerate config
    else
        print_status "Using existing accelerate config"
    fi
else
    print_info "Creating default accelerate config for multi-GPU..."
    # Get GPU count for configuration
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT_CONFIG=$(nvidia-smi -L | wc -l)
    else
        GPU_COUNT_CONFIG=1
    fi
    
    # Create a default config for multi-GPU training
    cat > accelerate_config.yaml << EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config: {}
distributed_type: MULTI_GPU
downcast_bf16: 'no'
enable_cpu_affinity: false
fsdp_config: {}
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: $GPU_COUNT_CONFIG
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF
    print_status "Default accelerate config created for $GPU_COUNT_CONFIG GPU(s)"
    print_info "Generated config:"
    cat accelerate_config.yaml
fi

# Step 6: Check GPU availability
print_info "Step 6: Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi -L | wc -l)
    print_status "Found $GPU_COUNT GPU(s):"
    nvidia-smi -L
else
    print_warning "nvidia-smi not found. Training will run on CPU."
    GPU_COUNT=0
fi

# Step 7: Check if training script exists
print_info "Step 7: Checking training script..."
TRAINING_SCRIPT="stt/multi_gpu/model.py"
if [ ! -f "$TRAINING_SCRIPT" ]; then
    print_error "Training script not found: $TRAINING_SCRIPT"
    print_info "Available files in stt/multi_gpu/:"
    ls -la stt/multi_gpu/ || echo "Directory not found"
    exit 1
fi
print_status "Training script found: $TRAINING_SCRIPT"

# Step 8: Check for required data files
print_info "Step 8: Checking required data files..."
if [ "$WARNING_NO_METADATA" = "true" ]; then
    print_warning "metadata.json not found at: $METADATA_FILE"
    print_info "You may need to prepare your dataset first"
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Exiting. Please prepare your dataset first."
        exit 1
    fi
else
    print_status "Found metadata.json with $SAMPLE_COUNT samples at: $METADATA_FILE"
fi

# Step 9: Launch training
print_info "Step 9: Launching multi-GPU training..."
echo
print_warning "Starting training in 5 seconds... Press Ctrl+C to cancel"
sleep 5

echo
print_status "🚀 Launching Accelerate training..."
echo

# Launch with accelerate
if [ "$GPU_COUNT" -gt 1 ]; then
    print_info "Launching multi-GPU training on $GPU_COUNT GPUs"
    accelerate launch --config_file accelerate_config.yaml $TRAINING_SCRIPT
elif [ "$GPU_COUNT" -eq 1 ]; then
    print_info "Launching single-GPU training"
    accelerate launch --config_file accelerate_config.yaml $TRAINING_SCRIPT
else
    print_info "Launching CPU training"
    python $TRAINING_SCRIPT
fi

# Step 10: Post-training info
echo
if [ $? -eq 0 ]; then
    print_status "🎉 Training completed successfully!"
    echo
    print_info "Check the following directories for outputs:"
    echo "  - Model checkpoints: ./checkpoints/ or ./output/"
    echo "  - Logs: ./logs/"
    echo "  - WandB logs: ./wandb/"
else
    print_error "Training failed with exit code $?"
    print_info "Check the error messages above for troubleshooting"
fi

echo
print_info "Script completed at $(date)"
