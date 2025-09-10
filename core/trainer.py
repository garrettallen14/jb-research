"""Ultra-simple GRPO trainer implementation."""
import torch
import torch.nn.functional as F
from torch.optim import AdamW
import asyncio
from typing import Dict, List, Any
from pathlib import Path

class SimpleTrainer:
    """Minimal GRPO trainer - crystal clear and simple."""
    
    def __init__(self, policy_model, reward_computer, config):
        self.policy = policy_model
        self.reward = reward_computer
        self.config = config
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.policy.model.parameters(), 
            lr=config.get('learning_rate', 5e-8)
        )
        
        print(f"🚀 Simple Trainer initialized!")
        print(f"   Learning rate: {config.get('learning_rate', 5e-8)}")
        print(f"   Group size: {config.get('group_size', 16)}")
        
        # Checkpointing setup
        self.checkpoint_dir = Path(config.get('output_dir', 'checkpoints'))
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.save_steps = config.get('save_steps', 25)
        
    async def train_step(self, behavior: Dict[str, Any]) -> Dict[str, float]:
        """Single training step - generate attacks, compute rewards, update policy."""
        
        # Extract behavior text
        if isinstance(behavior, dict):
            behavior_text = behavior.get('behavior', '')
            behavior_id = behavior.get('behavior_id', 'unknown')
        else:
            behavior_text = str(behavior)
            behavior_id = 'unknown'
            
        print(f"\n🎯 TRAINING STEP - Behavior: {behavior_id}")
        print(f"   Target: {behavior_text[:100]}...")
        
        # Step 1: Generate attack prompts
        group_size = self.config.get('group_size', 16)
        attacks = await self.policy.generate_attacks(behavior_text, group_size)
        print(f"✅ Generated {len(attacks)} attack prompts")
        
        # Step 2: Compute rewards for all attacks
        print("🎯 Computing PRBO rewards...")
        rewards = await self.reward.compute_batch_rewards(attacks, [behavior_text] * len(attacks))
        print(f"✅ Computed {len(rewards)} rewards")
        
        # Step 3: Compute relative advantages (GRPO core idea)
        advantages = self.compute_relative_advantages(rewards)
        print(f"📊 Advantages: min={min(advantages):.3f}, max={max(advantages):.3f}, avg={sum(advantages)/len(advantages):.3f}")
        
        # Step 4: Update policy using REINFORCE with advantages
        loss = self.compute_policy_loss(attacks, advantages)
        
        # Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(), 
            max_norm=self.config.get('max_grad_norm', 0.001)
        )
        
        self.optimizer.step()
        
        # Return metrics
        metrics = {
            "loss": loss.item(),
            "avg_reward": sum(rewards) / len(rewards),
            "min_reward": min(rewards),
            "max_reward": max(rewards),
            "avg_advantage": sum(advantages) / len(advantages)
        }
        
        print(f"📉 Loss: {metrics['loss']:.4f}, Avg Reward: {metrics['avg_reward']:.3f}")
        
        return metrics
    
    def compute_relative_advantages(self, rewards: List[float]) -> List[float]:
        """GRPO: Compare each reward to group average."""
        if not rewards:
            return []
            
        avg_reward = sum(rewards) / len(rewards)
        advantages = [reward - avg_reward for reward in rewards]
        return advantages
    
    def compute_policy_loss(self, attacks: List[str], advantages: List[float]) -> torch.Tensor:
        """Compute REINFORCE loss with advantages."""
        total_loss = 0.0
        valid_samples = 0
        
        for attack, advantage in zip(attacks, advantages):
            try:
                # Tokenize attack prompt
                inputs = self.policy.tokenizer(
                    attack,
                    return_tensors='pt',
                    truncation=True,
                    max_length=256,
                    padding=True
                )
                
                # Move to device
                device = next(self.policy.model.parameters()).device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Forward pass (with gradients!)
                outputs = self.policy.model(**inputs)
                logits = outputs.logits[0]  # [seq_len, vocab_size]
                
                # Get tokens and logits (shift for next-token prediction)
                tokens = inputs['input_ids'][0][:-1]
                shifted_logits = logits[:-1]
                
                # Compute log probabilities
                log_probs = F.log_softmax(shifted_logits, dim=-1)
                token_log_probs = log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1)
                
                # REINFORCE loss: -advantage * sum(log_probs)
                attack_log_prob = token_log_probs.sum()
                loss = -advantage * attack_log_prob
                
                total_loss = total_loss + loss
                valid_samples += 1
                
            except Exception as e:
                print(f"⚠️ Skipping sample due to error: {e}")
                continue
        
        if valid_samples == 0:
            print("❌ No valid samples for loss computation!")
            return torch.tensor(0.0, requires_grad=True)
        
        avg_loss = total_loss / valid_samples
        return avg_loss
    
    async def train_epoch(self, behaviors: List[Dict[str, Any]]) -> Dict[str, float]:
        """Train for one epoch on all behaviors."""
        epoch_metrics = []
        
        for i, behavior in enumerate(behaviors):
            print(f"\n{'='*60}")
            print(f"BATCH {i+1}/{len(behaviors)}")
            
            batch_metrics = await self.train_step(behavior)
            epoch_metrics.append(batch_metrics)
        
        # Aggregate epoch metrics
        if epoch_metrics:
            epoch_summary = {
                "epoch_loss": sum(m["loss"] for m in epoch_metrics) / len(epoch_metrics),
                "epoch_avg_reward": sum(m["avg_reward"] for m in epoch_metrics) / len(epoch_metrics),
                "epoch_batches": len(epoch_metrics)
            }
        else:
            epoch_summary = {"epoch_loss": 0.0, "epoch_avg_reward": 0.0, "epoch_batches": 0}
        
        print(f"\n🎉 EPOCH COMPLETE!")
        print(f"   Average Loss: {epoch_summary['epoch_loss']:.4f}")
        print(f"   Average Reward: {epoch_summary['epoch_avg_reward']:.3f}")
        
        return epoch_summary
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]) -> str:
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.policy.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"💾 Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.policy.device)
        
        self.policy.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        print(f"📂 Checkpoint loaded: {checkpoint_path}")
        return checkpoint
