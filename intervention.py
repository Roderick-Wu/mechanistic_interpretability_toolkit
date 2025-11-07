"""
Intervention tools for mechanistic interpretability.

This module provides utilities for:
- Activation patching (causal interventions)
- Steering vectors (controlling model behavior)
- Attention intervention
- Hook-based model manipulation

These tools allow you to modify the forward pass during inference to understand
causal relationships and control model outputs.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np


@dataclass
class ActivationPatch:
    """
    Specification for an activation patch.
    
    Attributes:
        layer_name: Name of the layer to patch
        position: Token position(s) to patch (int, list, or slice)
        value: New activation value (tensor or callable)
        mode: 'replace', 'add', or 'subtract'
    """
    layer_name: str
    position: Any  # int, List[int], or slice
    value: Any  # torch.Tensor or Callable
    mode: str = 'replace'  # 'replace', 'add', 'subtract'


@dataclass
class SteeringVector:
    """
    A steering vector for controlling model behavior.
    
    Attributes:
        vector: The steering vector (shape: [hidden_dim])
        layer_name: Name of layer to apply steering
        coefficient: Strength of steering
        positions: Which token positions to affect (None = all)
    """
    vector: torch.Tensor
    layer_name: str
    coefficient: float = 1.0
    positions: Optional[List[int]] = None


class InterventionHandler:
    """
    Manages interventions (patches, steering) during model forward pass.
    
    This class registers hooks on the model and applies interventions
    during inference without modifying model weights.
    """
    
    def __init__(self, model, tokenizer=None):
        """
        Initialize intervention handler.
        
        Args:
            model: The transformer model
            tokenizer: Optional tokenizer for debugging
        """
        self.model = model
        self.tokenizer = tokenizer
        self.hooks = []
        self.activations = {}
        self.interventions = defaultdict(list)
        
    def clear_interventions(self):
        """Remove all registered interventions."""
        self.interventions.clear()
        
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        
    def register_activation_patch(self, patch: ActivationPatch):
        """
        Register an activation patch to apply during forward pass.
        
        Args:
            patch: ActivationPatch specification
        """
        self.interventions[patch.layer_name].append(patch)
        
    def register_steering_vector(self, steering: SteeringVector):
        """
        Register a steering vector to apply during forward pass.
        
        Args:
            steering: SteeringVector specification
        """
        self.interventions[steering.layer_name].append(steering)
        
    def _apply_interventions(self, layer_name: str, activation: torch.Tensor) -> torch.Tensor:
        """
        Apply all interventions for a given layer.
        
        Args:
            layer_name: Name of the layer
            activation: Current activation tensor
            
        Returns:
            Modified activation tensor
        """
        if layer_name not in self.interventions:
            return activation
        
        modified = activation.clone()
        
        for intervention in self.interventions[layer_name]:
            if isinstance(intervention, ActivationPatch):
                modified = self._apply_patch(modified, intervention)
            elif isinstance(intervention, SteeringVector):
                modified = self._apply_steering(modified, intervention)
                
        return modified
    
    def _apply_patch(self, activation: torch.Tensor, patch: ActivationPatch) -> torch.Tensor:
        """Apply a single activation patch."""
        # Get the value to patch with
        if callable(patch.value):
            value = patch.value(activation)
        else:
            value = patch.value
            
        # Apply to specified positions
        if patch.mode == 'replace':
            activation[:, patch.position] = value
        elif patch.mode == 'add':
            activation[:, patch.position] += value
        elif patch.mode == 'subtract':
            activation[:, patch.position] -= value
        else:
            raise ValueError(f"Unknown patch mode: {patch.mode}")
            
        return activation
    
    def _apply_steering(self, activation: torch.Tensor, steering: SteeringVector) -> torch.Tensor:
        """Apply a steering vector."""
        vector = steering.vector.to(activation.device)
        
        if steering.positions is None:
            # Apply to all positions
            activation = activation + steering.coefficient * vector
        else:
            # Apply to specific positions
            for pos in steering.positions:
                activation[:, pos] += steering.coefficient * vector
                
        return activation
    
    def _create_hook(self, layer_name: str) -> Callable:
        """Create a hook function for a specific layer."""
        def hook_fn(module, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                # For layers that return (hidden_states, *other)
                modified = self._apply_interventions(layer_name, output[0])
                return (modified,) + output[1:]
            else:
                return self._apply_interventions(layer_name, output)
        return hook_fn
    
    def register_hooks(self):
        """Register hooks on all layers with interventions."""
        self.clear_hooks()
        
        for layer_name in self.interventions.keys():
            # Find the module by name
            module = self._get_module_by_name(layer_name)
            if module is not None:
                hook = module.register_forward_hook(self._create_hook(layer_name))
                self.hooks.append(hook)
            else:
                print(f"Warning: Could not find layer '{layer_name}'")
    
    def _get_module_by_name(self, name: str):
        """Get a module by its name."""
        parts = name.split('.')
        module = self.model
        
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module
    
    def __enter__(self):
        """Context manager entry - register hooks."""
        self.register_hooks()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clear hooks."""
        self.clear_hooks()


def create_steering_vector(model, tokenizer, 
                          positive_prompts: List[str],
                          negative_prompts: List[str],
                          layer_name: str,
                          normalize: bool = True) -> torch.Tensor:
    """
    Create a steering vector from contrastive prompts.
    
    This computes the difference in activations between positive and negative
    examples to create a vector that steers the model toward desired behavior.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer
        positive_prompts: List of prompts representing desired behavior
        negative_prompts: List of prompts representing undesired behavior
        layer_name: Which layer to extract activations from
        normalize: Whether to normalize the steering vector
        
    Returns:
        Steering vector (tensor of shape [hidden_dim])
    """
    model.eval()
    
    def get_layer_activations(prompts: List[str]) -> torch.Tensor:
        """Get mean activations for a list of prompts."""
        activations = []
        
        with torch.no_grad():
            for prompt in prompts:
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model(**inputs, output_hidden_states=True)
                
                # Get activations from specified layer
                hidden_states = outputs.hidden_states
                layer_idx = int(layer_name.split('.')[-1]) if '.' in layer_name else 0
                
                # Take mean over sequence length
                act = hidden_states[layer_idx].mean(dim=1).squeeze(0)
                activations.append(act)
        
        return torch.stack(activations).mean(dim=0)
    
    # Get mean activations for positive and negative prompts
    pos_activations = get_layer_activations(positive_prompts)
    neg_activations = get_layer_activations(negative_prompts)
    
    # Steering vector is the difference
    steering_vector = pos_activations - neg_activations
    
    if normalize:
        steering_vector = steering_vector / steering_vector.norm()
    
    return steering_vector


def activation_patch_experiment(model, tokenizer,
                                clean_prompt: str,
                                corrupted_prompt: str,
                                layer_name: str,
                                position: int = -1) -> Dict[str, Any]:
    """
    Perform activation patching to identify causal effects.
    
    This patches activations from a corrupted run into a clean run
    to see which activations are causally important.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer
        clean_prompt: The "clean" input
        corrupted_prompt: The "corrupted" input
        layer_name: Which layer to patch
        position: Which position to patch (-1 = last token)
        
    Returns:
        Dictionary with results
    """
    model.eval()
    
    # Get clean and corrupted activations
    with torch.no_grad():
        # Clean run
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
        clean_outputs = model(**clean_inputs, output_hidden_states=True)
        clean_logits = clean_outputs.logits
        
        # Corrupted run to get activation to patch
        corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt").to(model.device)
        corrupted_outputs = model(**corrupted_inputs, output_hidden_states=True)
        
        # Extract the activation to patch
        layer_idx = int(layer_name.split('.')[-1]) if '.' in layer_name else 0
        patch_value = corrupted_outputs.hidden_states[layer_idx][:, position, :]
    
    # Patched run
    handler = InterventionHandler(model, tokenizer)
    patch = ActivationPatch(
        layer_name=f"transformer.h.{layer_idx}",
        position=position,
        value=patch_value,
        mode='replace'
    )
    handler.register_activation_patch(patch)
    
    with handler:
        patched_outputs = model(**clean_inputs, output_hidden_states=True)
        patched_logits = patched_outputs.logits
    
    return {
        'clean_logits': clean_logits,
        'corrupted_logits': corrupted_outputs.logits,
        'patched_logits': patched_logits,
        'clean_prompt': clean_prompt,
        'corrupted_prompt': corrupted_prompt,
        'layer': layer_name,
        'position': position
    }


def path_patching(model, tokenizer,
                 clean_prompt: str,
                 corrupted_prompt: str,
                 layers_to_test: Optional[List[int]] = None) -> Dict[int, float]:
    """
    Perform path patching across multiple layers.
    
    Tests which layers are most causally important by patching
    each layer and measuring the effect on the output.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer
        clean_prompt: The "clean" input
        corrupted_prompt: The "corrupted" input
        layers_to_test: Which layers to test (None = all)
        
    Returns:
        Dictionary mapping layer index to causal effect score
    """
    model.eval()
    
    # Get total number of layers
    num_layers = model.config.n_layer if hasattr(model.config, 'n_layer') else \
                 model.config.num_hidden_layers
    
    if layers_to_test is None:
        layers_to_test = list(range(num_layers))
    
    # Get baseline outputs
    with torch.no_grad():
        clean_inputs = tokenizer(clean_prompt, return_tensors="pt").to(model.device)
        clean_outputs = model(**clean_inputs, output_hidden_states=True)
        clean_logits = clean_outputs.logits[0, -1, :]
        
        corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt").to(model.device)
        corrupted_outputs = model(**corrupted_inputs, output_hidden_states=True)
        corrupted_logits = corrupted_outputs.logits[0, -1, :]
    
    # Baseline difference
    baseline_diff = (clean_logits - corrupted_logits).norm().item()
    
    # Test each layer
    layer_effects = {}
    
    for layer_idx in layers_to_test:
        # Get activation to patch
        patch_value = corrupted_outputs.hidden_states[layer_idx][:, -1, :]
        
        # Apply patch
        handler = InterventionHandler(model, tokenizer)
        patch = ActivationPatch(
            layer_name=f"transformer.h.{layer_idx}",
            position=-1,
            value=patch_value,
            mode='replace'
        )
        handler.register_activation_patch(patch)
        
        with torch.no_grad(), handler:
            patched_outputs = model(**clean_inputs, output_hidden_states=True)
            patched_logits = patched_outputs.logits[0, -1, :]
        
        # Measure effect
        effect = (clean_logits - patched_logits).norm().item() / baseline_diff
        layer_effects[layer_idx] = effect
    
    return layer_effects


def steer_generation(model, tokenizer,
                    prompt: str,
                    steering_vector: torch.Tensor,
                    layer_name: str,
                    coefficient: float = 1.0,
                    max_length: int = 50,
                    positions: Optional[List[int]] = None) -> str:
    """
    Generate text with steering vector applied.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer
        prompt: Input prompt
        steering_vector: Steering vector to apply
        layer_name: Which layer to apply steering
        coefficient: Strength of steering
        max_length: Maximum generation length
        positions: Which positions to steer (None = all)
        
    Returns:
        Generated text
    """
    handler = InterventionHandler(model, tokenizer)
    
    steering = SteeringVector(
        vector=steering_vector,
        layer_name=layer_name,
        coefficient=coefficient,
        positions=positions
    )
    handler.register_steering_vector(steering)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with handler:
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            do_sample=True,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_feature_attribution(model, tokenizer,
                                prompt: str,
                                layer_name: str,
                                position: int = -1,
                                n_features: int = 10) -> List[Tuple[int, float]]:
    """
    Compute which features (neurons) are most important via ablation.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer
        prompt: Input prompt
        layer_name: Which layer to analyze
        position: Which position to analyze
        n_features: Number of top features to return
        
    Returns:
        List of (feature_index, importance_score) tuples
    """
    model.eval()
    
    # Get baseline output
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        baseline_outputs = model(**inputs, output_hidden_states=True)
        baseline_logits = baseline_outputs.logits[0, -1, :]
        
        # Get activations at target layer
        layer_idx = int(layer_name.split('.')[-1]) if '.' in layer_name else 0
        activations = baseline_outputs.hidden_states[layer_idx][0, position, :]
        hidden_dim = activations.shape[0]
    
    # Test ablating each feature
    feature_scores = []
    
    for feature_idx in range(hidden_dim):
        # Create ablation patch
        def ablate_feature(act):
            act_copy = act.clone()
            act_copy[:, :, feature_idx] = 0
            return act_copy
        
        handler = InterventionHandler(model, tokenizer)
        patch = ActivationPatch(
            layer_name=f"transformer.h.{layer_idx}",
            position=position,
            value=ablate_feature,
            mode='replace'
        )
        handler.register_activation_patch(patch)
        
        with torch.no_grad(), handler:
            ablated_outputs = model(**inputs, output_hidden_states=True)
            ablated_logits = ablated_outputs.logits[0, -1, :]
        
        # Compute importance as change in output
        importance = (baseline_logits - ablated_logits).norm().item()
        feature_scores.append((feature_idx, importance))
    
    # Sort by importance and return top N
    feature_scores.sort(key=lambda x: x[1], reverse=True)
    return feature_scores[:n_features]


class CausalTracer:
    """
    Trace causal pathways through the model using activation patching.
    
    This helps identify which components are responsible for specific behaviors.
    """
    
    def __init__(self, model, tokenizer):
        """
        Initialize causal tracer.
        
        Args:
            model: The transformer model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.handler = InterventionHandler(model, tokenizer)
        
    def trace_attention_heads(self, clean_prompt: str, corrupted_prompt: str,
                             metric_fn: Optional[Callable] = None) -> np.ndarray:
        """
        Trace which attention heads are causally important.
        
        Args:
            clean_prompt: Clean input
            corrupted_prompt: Corrupted input
            metric_fn: Function to compute metric from logits (default: prob of next token)
            
        Returns:
            Array of shape (num_layers, num_heads) with causal effects
        """
        num_layers = self.model.config.n_layer if hasattr(self.model.config, 'n_layer') else \
                    self.model.config.num_hidden_layers
        num_heads = self.model.config.n_head if hasattr(self.model.config, 'n_head') else \
                   self.model.config.num_attention_heads
        
        effects = np.zeros((num_layers, num_heads))
        
        # Get baseline
        with torch.no_grad():
            clean_inputs = self.tokenizer(clean_prompt, return_tensors="pt").to(self.model.device)
            clean_outputs = self.model(**clean_inputs, output_hidden_states=True)
            
            corrupted_inputs = self.tokenizer(corrupted_prompt, return_tensors="pt").to(self.model.device)
            corrupted_outputs = self.model(**corrupted_inputs, output_hidden_states=True)
        
        if metric_fn is None:
            # Default: KL divergence
            clean_probs = torch.softmax(clean_outputs.logits[0, -1], dim=-1)
            corrupted_probs = torch.softmax(corrupted_outputs.logits[0, -1], dim=-1)
            baseline_metric = torch.nn.functional.kl_div(
                corrupted_probs.log(), clean_probs, reduction='sum'
            ).item()
        else:
            baseline_metric = metric_fn(clean_outputs.logits, corrupted_outputs.logits)
        
        # Test each head (simplified - would need proper attention patching)
        # This is a placeholder for the actual implementation
        print("Note: Full attention head tracing requires deeper hook integration")
        
        return effects
    
    def trace_mlp_layers(self, clean_prompt: str, corrupted_prompt: str) -> Dict[int, float]:
        """
        Trace which MLP layers are causally important.
        
        Args:
            clean_prompt: Clean input
            corrupted_prompt: Corrupted input
            
        Returns:
            Dictionary mapping layer index to causal effect
        """
        return path_patching(self.model, self.tokenizer, 
                           clean_prompt, corrupted_prompt)


# Convenience function
def patch_and_run(model, tokenizer, prompt: str, 
                 patches: List[ActivationPatch]) -> torch.Tensor:
    """
    Convenience function to apply patches and run model.
    
    Args:
        model: The transformer model
        tokenizer: Tokenizer
        prompt: Input prompt
        patches: List of activation patches to apply
        
    Returns:
        Model output logits
    """
    handler = InterventionHandler(model, tokenizer)
    
    for patch in patches:
        handler.register_activation_patch(patch)
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with handler:
        outputs = model(**inputs)
    
    return outputs.logits
