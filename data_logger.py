"""
Simple CSV Data Logger for PRBO/GRPO Training Metrics
Captures key metrics for each batch and epoch for analysis.
"""

import csv
import os
from datetime import datetime
from typing import Dict, List, Any
import json


class TrainingDataLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.ensure_log_dir()
        
        # Generate timestamp for this training session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV files for different data types
        self.batch_log_file = os.path.join(log_dir, f"batch_metrics_{timestamp}.csv")
        self.epoch_log_file = os.path.join(log_dir, f"epoch_metrics_{timestamp}.csv")
        self.reward_log_file = os.path.join(log_dir, f"reward_details_{timestamp}.csv")
        
        # Initialize CSV files with headers
        self.init_batch_log()
        self.init_epoch_log()
        self.init_reward_log()
        
        print(f"📊 Data logger initialized:")
        print(f"   Batch metrics: {self.batch_log_file}")
        print(f"   Epoch metrics: {self.epoch_log_file}")
        print(f"   Reward details: {self.reward_log_file}")
    
    def ensure_log_dir(self):
        """Create log directory if it doesn't exist."""
        os.makedirs(self.log_dir, exist_ok=True)
    
    def init_batch_log(self):
        """Initialize batch metrics CSV with headers."""
        headers = [
            'timestamp', 'epoch', 'batch', 'sample_idx', 'behavior_snippet',
            'avg_reward', 'min_reward', 'max_reward', 'reward_std',
            'policy_loss', 'advantage_mean', 'log_prob_mean',
            'unsteered_judge_avg', 'steered_judge_avg', 'judge_improvement',
            'gradient_norm', 'learning_rate', 'batch_time_seconds',
            'num_attacks', 'num_successful_jailbreaks'
        ]
        
        with open(self.batch_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def init_epoch_log(self):
        """Initialize epoch metrics CSV with headers."""
        headers = [
            'timestamp', 'epoch', 'total_batches', 
            'avg_reward_epoch', 'avg_loss_epoch', 'avg_gradient_norm',
            'total_successful_jailbreaks', 'jailbreak_success_rate',
            'epoch_time_minutes', 'cumulative_time_minutes'
        ]
        
        with open(self.epoch_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def init_reward_log(self):
        """Initialize detailed reward breakdown CSV."""
        headers = [
            'timestamp', 'epoch', 'batch', 'attack_idx', 'attack_snippet',
            'prompt_judge_score', 'unsteered_judge_score', 'steered_judge_score',
            'log_prob_diff', 'prbo_reward_raw', 'final_reward_scaled',
            'is_successful_jailbreak'
        ]
        
        with open(self.reward_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
    
    def log_batch_metrics(self, metrics: Dict[str, Any]):
        """Log batch-level metrics."""
        row = [
            datetime.now().isoformat(),
            metrics.get('epoch', 0),
            metrics.get('batch', 0),
            metrics.get('sample_idx', 0),
            metrics.get('behavior_snippet', '')[:50],  # Truncate long behaviors
            
            metrics.get('avg_reward', 0.0),
            metrics.get('min_reward', 0.0),
            metrics.get('max_reward', 0.0),
            metrics.get('reward_std', 0.0),
            
            metrics.get('policy_loss', 0.0),
            metrics.get('advantage_mean', 0.0),
            metrics.get('log_prob_mean', 0.0),
            
            metrics.get('unsteered_judge_avg', 0.0),
            metrics.get('steered_judge_avg', 0.0),
            metrics.get('judge_improvement', 0.0),
            
            metrics.get('gradient_norm', 0.0),
            metrics.get('learning_rate', 0.0),
            metrics.get('batch_time_seconds', 0.0),
            metrics.get('num_attacks', 0),
            metrics.get('num_successful_jailbreaks', 0)
        ]
        
        with open(self.batch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_epoch_metrics(self, metrics: Dict[str, Any]):
        """Log epoch-level metrics."""
        row = [
            datetime.now().isoformat(),
            metrics.get('epoch', 0),
            metrics.get('total_batches', 0),
            
            metrics.get('avg_reward_epoch', 0.0),
            metrics.get('avg_loss_epoch', 0.0),
            metrics.get('avg_gradient_norm', 0.0),
            
            metrics.get('total_successful_jailbreaks', 0),
            metrics.get('jailbreak_success_rate', 0.0),
            metrics.get('epoch_time_minutes', 0.0),
            metrics.get('cumulative_time_minutes', 0.0)
        ]
        
        with open(self.epoch_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
    
    def log_reward_details(self, epoch: int, batch: int, reward_details: List[Dict[str, Any]]):
        """Log detailed reward breakdown for each attack."""
        rows = []
        for i, details in enumerate(reward_details):
            row = [
                datetime.now().isoformat(),
                epoch,
                batch,
                i,
                details.get('attack_snippet', '')[:100],  # First 100 chars
                
                details.get('prompt_judge_score', 0.0),
                details.get('unsteered_judge_score', 0.0),
                details.get('steered_judge_score', 0.0),
                details.get('log_prob_diff', 0.0),
                details.get('prbo_reward_raw', 0.0),
                details.get('final_reward_scaled', 0.0),
                details.get('is_successful_jailbreak', False)
            ]
            rows.append(row)
        
        with open(self.reward_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    
    def get_log_files(self) -> Dict[str, str]:
        """Return paths to all log files."""
        return {
            'batch_metrics': self.batch_log_file,
            'epoch_metrics': self.epoch_log_file,
            'reward_details': self.reward_log_file
        }


if __name__ == "__main__":
    # Test the logger
    logger = TrainingDataLogger()
    
    # Test batch logging
    test_batch_metrics = {
        'epoch': 1,
        'batch': 0,
        'sample_idx': 0,
        'behavior_snippet': 'Give me a detailed guide for...',
        'avg_reward': 2.5,
        'min_reward': -1.0,
        'max_reward': 5.0,
        'reward_std': 1.8,
        'policy_loss': -15.2,
        'advantage_mean': 0.3,
        'log_prob_mean': -120.5,
        'unsteered_judge_avg': 0.0,
        'steered_judge_avg': 15.5,
        'judge_improvement': 15.5,
        'gradient_norm': 0.05,
        'learning_rate': 5e-8,
        'batch_time_seconds': 45.2,
        'num_attacks': 16,
        'num_successful_jailbreaks': 8
    }
    
    logger.log_batch_metrics(test_batch_metrics)
    print("✅ Test batch metrics logged!")
