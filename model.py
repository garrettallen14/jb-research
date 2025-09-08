import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional, Dict
import os


class LocalModel:
    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **model_kwargs
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_load_kwargs = {
            "torch_dtype": self.torch_dtype,
            "device_map": "auto" if self.device == "cuda" else None,
            **model_kwargs
        }
        
        if load_in_8bit:
            model_load_kwargs["load_in_8bit"] = True
        elif load_in_4bit:
            model_load_kwargs["load_in_4bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **model_load_kwargs)
        
        if not (load_in_8bit or load_in_4bit) and self.device != "cuda":
            self.model = self.model.to(self.device)
    
    def generate(self, prompt: str, max_length: int = 512, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text[len(prompt):].strip()
    
    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def load_weights(self, weights_path: str):
        if os.path.isdir(weights_path):
            self.model = AutoModelForCausalLM.from_pretrained(weights_path)
        else:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
    
    def update_weights(self, state_dict: Dict[str, torch.Tensor]):
        self.model.load_state_dict(state_dict)
    
    def get_weights(self) -> Dict[str, torch.Tensor]:
        return self.model.state_dict()
