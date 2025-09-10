"""Ultra-simple logging utilities."""
import time
from datetime import datetime
from pathlib import Path

class SimpleLogger:
    """Dead simple logger - just print with timestamps."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create simple log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"training_{timestamp}.log"
        
        self.start_time = time.time()
        self.log("🚀 Simple training session started")
    
    def log(self, message: str):
        """Log message to console and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        
        # Console
        print(log_line)
        
        # File
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(log_line + "\n")
        except:
            pass  # Don't fail if file logging fails
    
    def log_metrics(self, epoch: int, batch: int, metrics: dict):
        """Log training metrics."""
        loss = metrics.get("loss", 0)
        reward = metrics.get("avg_reward", 0)
        self.log(f"📊 Epoch {epoch}, Batch {batch}: Loss={loss:.4f}, Reward={reward:.4f}")
    
    def close(self):
        """Close logging session."""
        elapsed = time.time() - self.start_time
        self.log(f"🏁 Training session ended. Total time: {elapsed:.2f}s")

# Global logger
_logger = None

def get_logger() -> SimpleLogger:
    """Get global logger instance."""
    global _logger
    if _logger is None:
        _logger = SimpleLogger()
    return _logger
