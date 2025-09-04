import os
from pathlib import Path

class TrainingConfig:
    """Main configuration class for Whisper training"""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    STT_DIR = PROJECT_ROOT / "stt"
    DATA_DIR = PROJECT_ROOT  # Data is in project root, not stt/data
    MULTI_GPU_DIR = STT_DIR / "multi_gpu"
    
    # Data files
    METADATA_FILE = STT_DIR / "metadata.json"
    AUDIO_FILES_DIR = PROJECT_ROOT / "audio_files"  # Audio files are in root/audio_files
    
    # Training script
    TRAINING_SCRIPT = MULTI_GPU_DIR / "model.py"
    
    # Model configuration
    MODEL_ID = "openai/whisper-large-v3-turbo"  # or "openai/whisper-tiny" for testing
    LANGUAGE = "uzbek"
    TASK = "transcribe"
    
    # Training parameters
    OUTPUT_DIR = "./whisper-uzbek-stt-multi-gpu"
    PER_DEVICE_TRAIN_BATCH_SIZE = 8
    PER_DEVICE_EVAL_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 2
    LEARNING_RATE = 1e-5
    WARMUP_STEPS = 500
    MAX_STEPS = 5000
    SAVE_STEPS = 1000
    EVAL_STEPS = 1000
    LOGGING_STEPS = 25
    
    # Accelerate configuration
    ACCELERATE_CONFIG_FILE = PROJECT_ROOT / "accelerate_config.yaml"
    
    # Environment variables
    HF_TOKEN_ENV = "HUGGING_FACE_TOKEN"
    WANDB_API_KEY_ENV = "WANDB_API_KEY"
    WANDB_PROJECT_ENV = "WANDB_PROJECT"
    WANDB_PROJECT_DEFAULT = "whisper-uzbek-stt"
    
    # GPU settings
    MIXED_PRECISION = "fp16"
    GRADIENT_CHECKPOINTING = True
    DATALOADER_PIN_MEMORY = False
    DATALOADER_NUM_WORKERS = 0
    
    # Dataset settings
    TEST_SIZE = 0.1
    SEED = 42
    REMOVE_UNUSED_COLUMNS = False
    
    # Augmentation settings
    AUGMENTATION_PROBABILITY = 0.3
    
    # Model saving
    SAVE_TOTAL_LIMIT = 3
    LOAD_BEST_MODEL_AT_END = True
    METRIC_FOR_BEST_MODEL = "wer"
    GREATER_IS_BETTER = False
    PUSH_TO_HUB = False
    
    @classmethod
    def get_accelerate_config(cls, num_gpus: int = None) -> dict:
        """Generate accelerate configuration based on available GPUs"""
        if num_gpus is None:
            try:
                import subprocess
                result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
                num_gpus = len(result.stdout.strip().split('\n')) if result.returncode == 0 else 1
            except:
                num_gpus = 1
        
        if num_gpus > 1:
            distributed_type = "MULTI_GPU"
        else:
            distributed_type = "NO"
        
        return {
            "compute_environment": "LOCAL_MACHINE",
            "debug": False,
            "deepspeed_config": {},
            "distributed_type": distributed_type,
            "downcast_bf16": "no",
            "enable_cpu_affinity": False,
            "fsdp_config": {},
            "gpu_ids": "all",
            "machine_rank": 0,
            "main_training_function": "main",
            "mixed_precision": cls.MIXED_PRECISION,
            "num_machines": 1,
            "num_processes": num_gpus,
            "rdzv_backend": "static",
            "same_network": True,
            "tpu_config": {},
            "tpu_env": [],
            "tpu_use_cluster": False,
            "tpu_use_sudo": False,
            "use_cpu": False
        }
    
    @classmethod
    def validate_paths(cls) -> dict:
        """Validate that all required paths exist"""
        validation_results = {}
        
        paths_to_check = {
            "Project root": cls.PROJECT_ROOT,
            "STT directory": cls.STT_DIR,
            "Data directory": cls.DATA_DIR,
            "Multi-GPU directory": cls.MULTI_GPU_DIR,
            "Training script": cls.TRAINING_SCRIPT,
            "Metadata file": cls.METADATA_FILE,
        }
        
        for name, path in paths_to_check.items():
            validation_results[name] = {
                "path": str(path),
                "exists": path.exists(),
                "is_file": path.is_file() if path.exists() else None,
                "is_dir": path.is_dir() if path.exists() else None
            }
        
        return validation_results
    
    @classmethod
    def get_dataset_info(cls) -> dict:
        """Get information about the dataset"""
        info = {
            "metadata_exists": cls.METADATA_FILE.exists(),
            "sample_count": 0,
            "audio_dir_exists": cls.AUDIO_FILES_DIR.exists(),
            "audio_file_count": 0
        }
        
        if info["metadata_exists"]:
            try:
                with open(cls.METADATA_FILE, 'r', encoding='utf-8') as f:
                    info["sample_count"] = sum(1 for _ in f)
            except Exception as e:
                info["metadata_error"] = str(e)
        
        if info["audio_dir_exists"]:
            try:
                audio_files = list(cls.AUDIO_FILES_DIR.glob("*.flac")) + \
                            list(cls.AUDIO_FILES_DIR.glob("*.wav")) + \
                            list(cls.AUDIO_FILES_DIR.glob("*.mp3"))
                info["audio_file_count"] = len(audio_files)
            except Exception as e:
                info["audio_error"] = str(e)
        
        return info
    
    @classmethod
    def get_env_vars(cls) -> dict:
        """Get environment variables"""
        return {
            "hf_token": os.getenv(cls.HF_TOKEN_ENV),
            "wandb_api_key": os.getenv(cls.WANDB_API_KEY_ENV),
            "wandb_project": os.getenv(cls.WANDB_PROJECT_ENV, cls.WANDB_PROJECT_DEFAULT)
        }

# Example usage and validation
if __name__ == "__main__":
    config = TrainingConfig()
    
    print("=== Training Configuration ===")
    print(f"Project Root: {config.PROJECT_ROOT}")
    print(f"Training Script: {config.TRAINING_SCRIPT}")
    print(f"Metadata File: {config.METADATA_FILE}")
    print(f"Model ID: {config.MODEL_ID}")
    print(f"Output Directory: {config.OUTPUT_DIR}")
    
    print("\n=== Path Validation ===")
    validation = config.validate_paths()
    for name, result in validation.items():
        status = "✅" if result["exists"] else "❌"
        print(f"{status} {name}: {result['path']}")
    
    print("\n=== Dataset Information ===")
    dataset_info = config.get_dataset_info()
    for key, value in dataset_info.items():
        print(f"{key}: {value}")
    
    print("\n=== Environment Variables ===")
    env_vars = config.get_env_vars()
    for key, value in env_vars.items():
        masked_value = "***" if value and "token" in key.lower() else value
        print(f"{key}: {masked_value}")
    
    print("\n=== Accelerate Configuration ===")
    accelerate_config = config.get_accelerate_config()
    for key, value in accelerate_config.items():
        print(f"{key}: {value}")
