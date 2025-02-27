"""
Utilities for model loading and activation extraction.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any, Tuple, List, Optional, Union
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ModelWrapper:
    """Wrapper for loading and extracting activations from language models."""
    
    def __init__(self, model_name: str, device: Optional[str] = None):
        """
        Initialize the model wrapper.
        
        Args:
            model_name: Name or path of the model to load
            device: Device to load the model on. If None, will determine automatically.
        """
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                       "mps" if torch.backends.mps.is_available() else 
                                       "cpu")
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        # Load model and tokenizer
        logger.info(f"Loading model and tokenizer from {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        
        # Init activation storage
        self.hooked_activations = {
            "early_rep": None,
            "late_rep": None,
        }
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations at specific layers."""
        
        def early_hook_fn(module, input, output):
            """Capture embeddings at the beginning."""
            self.hooked_activations["early_rep"] = output.clone().detach()
        
        def late_hook_fn(module, input, output):
            """Capture activations near the end of processing."""
            self.hooked_activations["late_rep"] = output.clone().detach()
        
        # Register hooks - adjust these based on model architecture if needed
        try:
            # Try common model architectures
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                # For many modern transformer models
                self.early_hook = self.model.model.embed_tokens.register_forward_hook(early_hook_fn)
                
                # For the late hook, try to find the MLP in the second-to-last layer
                if hasattr(self.model.model, 'layers'):
                    self.late_hook = self.model.model.layers[-2].mlp.register_forward_hook(late_hook_fn)
                elif hasattr(self.model.model, 'h'):
                    # For older GPT-2 style models
                    self.late_hook = self.model.model.h[-2].mlp.register_forward_hook(late_hook_fn)
                else:
                    logger.warning("Could not attach late hook - unknown model architecture")
            else:
                logger.warning("Could not attach hooks - unknown model architecture")
        except Exception as e:
            logger.error(f"Error registering hooks: {e}")
    
    def generate(self, prompt: str, max_new_tokens: int = 200, 
                temperature: float = 0.7, do_sample: bool = True) -> Tuple[str, Dict[str, np.ndarray]]:
        """
        Generate text and capture activations.
        
        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Tuple of (generated_text, activation_dict)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Reset activations
        for k in self.hooked_activations:
            self.hooked_activations[k] = None
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Get generated text (excluding prompt)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response_text = generated_text[len(self.tokenizer.decode(inputs.input_ids[0], skip_special_tokens=True)):]
        
        # Process captured activations
        activations = {}
        for k, v in self.hooked_activations.items():
            if v is not None:
                # Convert to numpy array and move to CPU
                activations[k] = v.cpu().numpy()
        
        return response_text, activations
    
    def cleanup(self):
        """Remove hooks when done."""
        if hasattr(self, 'early_hook'):
            self.early_hook.remove()
        if hasattr(self, 'late_hook'):
            self.late_hook.remove()
        
    def __del__(self):
        """Clean up hooks when the object is deleted."""
        self.cleanup()