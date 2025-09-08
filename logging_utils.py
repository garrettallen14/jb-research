"""
Simple, robust logging utilities for GRPO training.
Logs to both console and files for easy debugging.
"""

import os
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class SimpleLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create timestamped log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.training_log = self.log_dir / f"training_{timestamp}.log"
        self.metrics_log = self.log_dir / f"metrics_{timestamp}.jsonl"
        self.error_log = self.log_dir / f"errors_{timestamp}.log"
        
        self.start_time = time.time()
        
        # Log session start
        self.log("🚀 Training session started", "INFO")
    
    def log(self, message: str, level: str = "INFO"):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {level}: {message}"
        
        # Console output
        print(log_line)
        
        # File output
        with open(self.training_log, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")
    
    def log_metrics(self, epoch: int, batch: int, metrics: Dict[str, Any]):
        """Log training metrics to JSONL file."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "epoch": epoch,
            "batch": batch,
            "elapsed_time": time.time() - self.start_time,
            **metrics
        }
        
        # Console summary
        loss = metrics.get("loss", 0)
        reward = metrics.get("avg_reward", 0)
        self.log(f"📊 Epoch {epoch}, Batch {batch}: Loss={loss:.4f}, Reward={reward:.4f}")
        
        # JSONL file
        with open(self.metrics_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    
    def log_error(self, error: str, context: str = ""):
        """Log errors to both console and error file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        error_line = f"[{timestamp}] ERROR in {context}: {error}"
        
        # Console
        print(f"❌ {error_line}")
        
        # Error file
        with open(self.error_log, "a", encoding="utf-8") as f:
            f.write(error_line + "\n")
    
    def log_api_call(self, endpoint: str, success: bool, response_time: float = 0):
        """Log API call results."""
        status = "✅ SUCCESS" if success else "❌ FAILED"
        self.log(f"🌐 API {endpoint}: {status} ({response_time:.2f}s)")
    
    def close(self):
        """Close logging session."""
        total_time = time.time() - self.start_time
        self.log(f"🏁 Training session ended. Total time: {total_time:.2f}s", "INFO")


# Global logger instance
logger = None

def get_logger(log_dir: str = "logs") -> SimpleLogger:
    """Get or create global logger instance."""
    global logger
    if logger is None:
        logger = SimpleLogger(log_dir)
    return logger
