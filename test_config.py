#!/usr/bin/env python3
"""
Configuration Test Script
Run this to validate your configuration and check if everything is set up correctly.
"""

from config import TrainingConfig
import sys

def main():
    print("🔍 Testing Training Configuration")
    print("=" * 50)
    
    config = TrainingConfig()
    all_good = True
    
    # Test 1: Path validation
    print("\n1️⃣  Path Validation:")
    validation = config.validate_paths()
    for name, result in validation.items():
        status = "✅" if result["exists"] else "❌"
        print(f"   {status} {name}: {result['path']}")
        if not result["exists"] and "script" in name.lower():
            all_good = False
    
    # Test 2: Dataset information
    print("\n2️⃣  Dataset Information:")
    dataset_info = config.get_dataset_info()
    
    if dataset_info["metadata_exists"]:
        print(f"   ✅ Metadata file: {config.METADATA_FILE}")
        print(f"   📊 Sample count: {dataset_info['sample_count']}")
    else:
        print(f"   ❌ Metadata file not found: {config.METADATA_FILE}")
        all_good = False
    
    if dataset_info["audio_dir_exists"]:
        print(f"   ✅ Audio directory: {config.AUDIO_FILES_DIR}")
        print(f"   🎵 Audio files: {dataset_info['audio_file_count']}")
    else:
        print(f"   ⚠️  Audio directory not found: {config.AUDIO_FILES_DIR}")
    
    # Test 3: Environment variables
    print("\n3️⃣  Environment Variables:")
    env_vars = config.get_env_vars()
    
    for key, value in env_vars.items():
        if value:
            masked_value = "***" if "token" in key.lower() else value
            print(f"   ✅ {key}: {masked_value}")
        else:
            print(f"   ⚠️  {key}: Not set")
    
    # Test 4: Accelerate configuration
    print("\n4️⃣  Accelerate Configuration:")
    try:
        accelerate_config = config.get_accelerate_config()
        print(f"   ✅ Distributed type: {accelerate_config['distributed_type']}")
        print(f"   ✅ Number of processes: {accelerate_config['num_processes']}")
        print(f"   ✅ Mixed precision: {accelerate_config['mixed_precision']}")
    except Exception as e:
        print(f"   ❌ Error generating accelerate config: {e}")
        all_good = False
    
    # Test 5: Model configuration
    print("\n5️⃣  Model Configuration:")
    print(f"   📝 Model ID: {config.MODEL_ID}")
    print(f"   🌍 Language: {config.LANGUAGE}")
    print(f"   🎯 Task: {config.TASK}")
    print(f"   📁 Output directory: {config.OUTPUT_DIR}")
    
    # Summary
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 All checks passed! Ready for training.")
        sys.exit(0)
    else:
        print("❌ Some issues found. Please fix them before training.")
        sys.exit(1)

if __name__ == "__main__":
    main()
