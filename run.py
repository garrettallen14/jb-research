#!/usr/bin/env python3
"""
GRPO Training Runner with YAML Configuration
Load config from run.yaml and start training.
"""

import yaml
import os
import torch
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

from model import LocalModel
from client import OpenAIClient
from reward import PRBOReward
from data import load_behaviors, get_batch
from trainer import MinimalGRPOTrainer


def load_config(config_path: str = "run.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_models(config: Dict[str, Any]) -> tuple:
    """Initialize models based on config."""
    model_config = config['model']
    
    # Policy model (local HuggingFace model)
    print(f"Loading policy model: {model_config['policy_model_name']}")
    policy_model = LocalModel(
        model_config['policy_model_name'],
        load_in_4bit=model_config.get('load_in_4bit', True),
        torch_dtype=torch.float16 if config['hardware']['torch_dtype'] == 'float16' else torch.float32,
        device_map=config['hardware']['device_map']
    )
    
    # Target model (OpenRouter endpoint)
    target_url = os.getenv('TARGET_BASE_URL')
    target_name = os.getenv('TARGET_MODEL_NAME')
    target_key = os.getenv('TARGET_API_KEY')
    print(f"Connecting to target model: {target_name} at {target_url}")
    
    target_model = OpenAIClient(
        base_url=target_url,
        model_name=target_name,
        api_key=target_key
    )
    
    # Judge model (OpenRouter endpoint)
    judge_url = os.getenv('JUDGE_BASE_URL')
    judge_name = os.getenv('JUDGE_MODEL_NAME')
    judge_key = os.getenv('JUDGE_API_KEY')
    print(f"Connecting to judge model: {judge_name} at {judge_url}")
    
    judge_model = OpenAIClient(
        base_url=judge_url,
        model_name=judge_name,
        api_key=judge_key
    )
    
    return policy_model, target_model, judge_model


# Removed: create_grpo_config and setup_reward_function
# Trainer is now self-contained and handles config internally


def main():
    """Main training entry point."""
    # Load configuration
    print("Loading configuration from run.yaml...")
    config = load_config()
    
    # Set random seeds
    torch.manual_seed(config['experiment']['seed'])
    
    # Create output directory
    output_dir = Path(config['training']['output_dir'])
    output_dir.mkdir(exist_ok=True)
    
    # Setup models
    print("Setting up models...")
    policy_model, target_model, judge_model = setup_models(config)
    
    # Load data
    print("Loading behaviors dataset...")
    behaviors = load_behaviors(config['data']['behaviors_file'])
    if config['data'].get('max_behaviors'):
        behaviors = behaviors[:config['data']['max_behaviors']]
    print(f"Loaded {len(behaviors)} behaviors")
    
    # Initialize minimal GRPO trainer with config
    print("Initializing MinimalGRPOTrainer...")
    training_config = config.get('training', {})
    trainer = MinimalGRPOTrainer(
        policy_model=policy_model,
        target_model=target_model, 
        judge_model=judge_model,
        config=training_config  # Pass config to trainer
    )
    
    # Print training info
    print("\n" + "="*50)
    print("GRPO TRAINING CONFIGURATION")
    print("="*50)
    print(f"Policy Model: {config['model']['policy_model_name']}")
    print(f"Target Model: {os.getenv('TARGET_MODEL_NAME')}")
    print(f"Judge Model: {os.getenv('JUDGE_MODEL_NAME')}")
    print(f"Batch Size: {trainer.batch_size}")
    print(f"Group Size: {trainer.group_size}")
    print(f"Learning Rate: {trainer.learning_rate}")
    print(f"Max Epochs: {trainer.max_epochs}")
    print(f"Max Length: {trainer.max_length}")
    print(f"Temperature: {trainer.temperature}")
    print(f"GPU Memory Limit: {config['hardware']['max_memory_per_gpu']}")
    print(f"Behaviors: {len(behaviors)}")
    print("="*50 + "\n")
    
    # Start training
    print("Starting GRPO training...")
    try:
        trainer.train(behaviors)
        print("Training completed successfully!")
        
        # Save final model
        final_model_path = output_dir / "final_model"
        policy_model.save_model(str(final_model_path))
        print(f"Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        # Save checkpoint
        checkpoint_path = output_dir / "interrupted_checkpoint"
        policy_model.save_model(str(checkpoint_path))
        print(f"Checkpoint saved to {checkpoint_path}")
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        # Save emergency checkpoint
        emergency_path = output_dir / "emergency_checkpoint"
        try:
            policy_model.save_model(str(emergency_path))
            print(f"Emergency checkpoint saved to {emergency_path}")
        except:
            print("Could not save emergency checkpoint")
        raise


if __name__ == "__main__":
    main()
