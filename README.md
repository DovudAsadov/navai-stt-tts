# 🎙️ NAVAI STT-TTS

Speech-to-Text and Text-to-Speech training framework using Whisper and Hugging Face Transformers.

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- CUDA (optional, for GPU training)
- Hugging Face Token
- WandB API Key (optional)

---

### 📦 1. Clone the Repository

```bash
git clone https://github.com/DovudAsadov/navai-stt-tts.git
cd navai-stt-tts
```

---

### ⚙️ 2. Install [`uv`](https://github.com/astral-sh/uv) (Python package & environment manager)

If you haven't installed `uv` yet:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

---

### 🧪 3. Set Up Python Environment

```bash
uv venv
source .venv/bin/activate
uv sync
```

---

### 🔐 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
echo "HUGGING_FACE_TOKEN=your_hf_token" > .env
echo "WANDB_API_KEY=your_wandb_key" >> .env
echo "WANDB_PROJECT=whisper-uzbek-stt" >> .env
```

> ⚠️ Make sure to use `>>` on subsequent lines to **append** instead of overwrite.

---

### 🚀 5. Run Training

#### Quick Setup & Training
```bash
./setup_and_train.sh
```

#### Manual Training
```bash
# Configure accelerate for multi-GPU
accelerate config

# Start training
accelerate launch --config_file accelerate_config.yaml stt/multi_gpu/model.py
```

---

### 📁 Project Structure

```
├── stt/                    # Speech-to-Text components
├── tts/                    # Text-to-Speech components  
├── audio_files/            # Training audio data
├── config.py              # Main configuration
├── setup_and_train.sh     # Automated setup script
└── pyproject.toml         # Dependencies
```