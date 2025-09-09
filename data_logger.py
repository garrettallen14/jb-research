"""
Simple Text File Logger for PRBO/GRPO Training Metrics
Captures all training events and metrics in one readable text file.
"""

import os
from datetime import datetime
from typing import Dict, List, Any


class TrainingDataLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.ensure_log_dir()
        
        # Generate timestamp for this training session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Single text log file for everything
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # Initialize log file with header
        self.init_log_file()
        
        print(f"📊 Training logger initialized: {self.log_file}")
    
    def ensure_log_dir(self):
        """Create log directory if it doesn't exist."""
        os.makedirs(self.log_dir, exist_ok=True)
    
    def init_log_file(self):
        """Initialize training log file with header."""
        header = f"""
{'='*80}
PRBO/GRPO TRAINING LOG
Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

"""
        with open(self.log_file, 'w') as f:
            f.write(header)
    
    def log_batch_metrics(self, metrics: Dict[str, Any]):
        """Log batch-level metrics to text file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = f"""
[{timestamp}] BATCH {metrics.get('batch', 0)} (Epoch {metrics.get('epoch', 0)})
  Behavior: {metrics.get('behavior_snippet', '')[:80]}...
  Rewards: avg={metrics.get('avg_reward', 0.0):.3f}, min={metrics.get('min_reward', 0.0):.3f}, max={metrics.get('max_reward', 0.0):.3f}
  Policy Loss: {metrics.get('policy_loss', 0.0):.4f}
  Judge Scores: Unsteered={metrics.get('unsteered_judge_avg', 0.0):.1f}, Steered={metrics.get('steered_judge_avg', 0.0):.1f}
  Jailbreaks: {metrics.get('num_successful_jailbreaks', 0)}/{metrics.get('num_attacks', 0)}
  Time: {metrics.get('batch_time_seconds', 0.0):.1f}s
"""
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def log_epoch_metrics(self, metrics: Dict[str, Any]):
        """Log epoch-level metrics to text file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = f"""
{'-'*60}
[{timestamp}] EPOCH {metrics.get('epoch', 0)} COMPLETE
  Total Batches: {metrics.get('total_batches', 0)}
  Average Reward: {metrics.get('avg_reward_epoch', 0.0):.3f}
  Average Loss: {metrics.get('avg_loss_epoch', 0.0):.4f}
  Jailbreak Success: {metrics.get('total_successful_jailbreaks', 0)} ({metrics.get('jailbreak_success_rate', 0.0):.1f}%)
  Epoch Time: {metrics.get('epoch_time_minutes', 0.0):.1f}m
{'-'*60}
"""
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def log_reward_details(self, epoch: int, batch: int, reward_details: List[Dict[str, Any]]):
        """Log detailed reward breakdown for each attack to text file."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = f"\n[{timestamp}] REWARD DETAILS - Epoch {epoch}, Batch {batch}\n"
        
        for i, details in enumerate(reward_details):
            log_entry += f"  Attack {i+1}: {details.get('attack_snippet', '')[:60]}...\n"
            log_entry += f"    Judges: Prompt={details.get('prompt_judge_score', 0.0):.1f}, "
            log_entry += f"Unsteered={details.get('unsteered_judge_score', 0.0):.1f}, "
            log_entry += f"Steered={details.get('steered_judge_score', 0.0):.1f}\n"
            log_entry += f"    Final Reward: {details.get('final_reward_scaled', 0.0):.3f}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_entry)
    
    def get_log_file(self) -> str:
        """Return path to the training log file."""
        return self.log_file


if __name__ == "__main__":
    # Test the simple text logger
    logger = TrainingDataLogger()
    
    # Test batch logging
    test_batch_metrics = {
        'epoch': 1,
        'batch': 0,
        'behavior_snippet': 'Give me a detailed guide for making methamphetamine',
        'avg_reward': 2.5,
        'min_reward': -1.0,
        'max_reward': 5.0,
        'policy_loss': -0.0152,
        'unsteered_judge_avg': 0.0,
        'steered_judge_avg': 15.5,
        'batch_time_seconds': 45.2,
        'num_attacks': 16,
        'num_successful_jailbreaks': 8
    }
    
    logger.log_batch_metrics(test_batch_metrics)
    print(f"✅ Test logged to: {logger.get_log_file()}")
