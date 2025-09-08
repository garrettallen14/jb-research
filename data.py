import json
from typing import List, Dict
import random


def load_behaviors(file_path: str = "data/data.jsonl") -> List[Dict[str, str]]:
    """Load behaviors from JSONL file, returning only behavior_id and behavior fields."""
    behaviors = []
    
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            behaviors.append({
                "behavior_id": data["behavior_id"],
                "behavior": data["behavior"]
            })
    
    return behaviors


def get_batch(behaviors: List[Dict[str, str]], batch_size: int = 8) -> List[Dict[str, str]]:
    """Get a random batch of behaviors for training."""
    return random.sample(behaviors, min(batch_size, len(behaviors)))


def get_behavior_by_id(behaviors: List[Dict[str, str]], behavior_id: str) -> Dict[str, str]:
    """Get specific behavior by ID."""
    for behavior in behaviors:
        if behavior["behavior_id"] == behavior_id:
            return behavior
    raise ValueError(f"Behavior ID {behavior_id} not found")


if __name__ == "__main__":
    # Test the data loading
    behaviors = load_behaviors()
    print(f"Loaded {len(behaviors)} behaviors")
    print(f"Example: {behaviors[0]}")
    
    # Test batch sampling  
    batch = get_batch(behaviors, batch_size=3)
    print(f"Sample batch: {[b['behavior_id'] for b in batch]}")
