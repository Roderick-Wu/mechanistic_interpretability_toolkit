"""
Model Loader for Mechanistic Interpretability Research

This module provides utilities for loading transformer models locally and 
extracting architectural information useful for mechanistic interpretability research.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig
from safetensors.torch import load_file


class ModelInspector:
    """
    A class for loading and inspecting transformer models for mechanistic interpretability.
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the ModelInspector.
        
        Args:
            model_path: Path to the local model directory
            device: Device to load the model on ('cpu', 'cuda', etc.)
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.tokenizer = None
        self.config = None
        
    def load_model(self, load_tokenizer: bool = True) -> Tuple[Any, Optional[Any]]:
        """
        Load the model and optionally the tokenizer from local path.
        
        Args:
            load_tokenizer: Whether to also load the tokenizer
            
        Returns:
            Tuple of (model, tokenizer) where tokenizer is None if load_tokenizer=False
        """
        print(f"Loading model from {self.model_path}...")
        
        # Load config
        self.config = AutoConfig.from_pretrained(str(self.model_path))
        
        # Load model
        self.model = AutoModel.from_pretrained(
            str(self.model_path),
            config=self.config,
            local_files_only=True
        ).to(self.device)
        
        # Load tokenizer if requested
        if load_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True
            )
        
        print(f"Model loaded successfully on {self.device}")
        return self.model, self.tokenizer
    
    def get_num_layers(self) -> int:
        """
        Get the number of transformer layers in the model.
        
        Returns:
            Number of layers
        """
        if self.config is None:
            self.config = AutoConfig.from_pretrained(str(self.model_path))
        
        # Different model types use different config keys
        layer_keys = ['n_layer', 'num_hidden_layers', 'num_layers', 'n_layers']
        for key in layer_keys:
            if hasattr(self.config, key):
                return getattr(self.config, key)
        
        raise ValueError("Could not determine number of layers from config")
    
    def get_layer_names(self) -> List[str]:
        """
        Get the names of all layers in the model.
        
        Returns:
            List of layer names as strings
        """
        if self.model is None:
            self.load_model(load_tokenizer=False)
        
        return [name for name, _ in self.model.named_modules()]
    
    def get_attention_layer_names(self) -> List[str]:
        """
        Get the names of attention layers specifically.
        
        Returns:
            List of attention layer names
        """
        if self.model is None:
            self.load_model(load_tokenizer=False)
        
        attn_layers = []
        for name, module in self.model.named_modules():
            module_type = type(module).__name__.lower()
            if 'attention' in module_type or 'attn' in name.lower():
                attn_layers.append(name)
        
        return attn_layers
    
    def get_mlp_layer_names(self) -> List[str]:
        """
        Get the names of MLP/feedforward layers.
        
        Returns:
            List of MLP layer names
        """
        if self.model is None:
            self.load_model(load_tokenizer=False)
        
        mlp_layers = []
        for name, module in self.model.named_modules():
            if 'mlp' in name.lower() or 'feed_forward' in name.lower():
                mlp_layers.append(name)
        
        return mlp_layers
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the full model configuration as a dictionary.
        
        Returns:
            Dictionary containing model configuration
        """
        if self.config is None:
            self.config = AutoConfig.from_pretrained(str(self.model_path))
        
        return self.config.to_dict()
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the model architecture.
        
        Returns:
            Dictionary containing architectural information
        """
        if self.config is None:
            self.config = AutoConfig.from_pretrained(str(self.model_path))
        
        config_dict = self.config.to_dict()
        
        # Extract key architectural parameters
        summary = {
            "model_type": config_dict.get("model_type", "unknown"),
            "num_layers": self.get_num_layers(),
            "hidden_size": config_dict.get("n_embd") or config_dict.get("hidden_size"),
            "num_attention_heads": config_dict.get("n_head") or config_dict.get("num_attention_heads"),
            "intermediate_size": config_dict.get("intermediate_size"),
            "vocab_size": config_dict.get("vocab_size"),
            "max_position_embeddings": config_dict.get("n_positions") or config_dict.get("max_position_embeddings"),
            "activation_function": config_dict.get("activation_function"),
        }
        
        # Add head dimension if we can calculate it
        if summary["hidden_size"] and summary["num_attention_heads"]:
            summary["head_dim"] = summary["hidden_size"] // summary["num_attention_heads"]
        
        return summary
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get the number of parameters in the model (total and trainable).
        
        Returns:
            Dictionary with 'total' and 'trainable' parameter counts
        """
        if self.model is None:
            self.load_model(load_tokenizer=False)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params
        }
    
    def get_layer_by_name(self, layer_name: str):
        """
        Get a specific layer module by its name.
        
        Args:
            layer_name: Name of the layer to retrieve
            
        Returns:
            The layer module
        """
        if self.model is None:
            self.load_model(load_tokenizer=False)
        
        for name, module in self.model.named_modules():
            if name == layer_name:
                return module
        
        raise ValueError(f"Layer '{layer_name}' not found in model")
    
    def get_weights_by_layer(self, layer_name: str) -> Dict[str, torch.Tensor]:
        """
        Get all weight tensors for a specific layer.
        
        Args:
            layer_name: Name of the layer
            
        Returns:
            Dictionary mapping parameter names to tensors
        """
        layer = self.get_layer_by_name(layer_name)
        return {name: param.data for name, param in layer.named_parameters()}
    
    def print_architecture_summary(self):
        """
        Print a formatted summary of the model architecture.
        """
        summary = self.get_architecture_summary()
        params = self.get_parameter_count()
        
        print("\n" + "="*60)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*60)
        print(f"Model Type: {summary['model_type']}")
        print(f"Number of Layers: {summary['num_layers']}")
        print(f"Hidden Size: {summary['hidden_size']}")
        print(f"Number of Attention Heads: {summary['num_attention_heads']}")
        print(f"Head Dimension: {summary.get('head_dim', 'N/A')}")
        print(f"Intermediate Size: {summary.get('intermediate_size', 'N/A')}")
        print(f"Vocabulary Size: {summary['vocab_size']}")
        print(f"Max Position Embeddings: {summary['max_position_embeddings']}")
        print(f"Activation Function: {summary['activation_function']}")
        print(f"\nTotal Parameters: {params['total']:,}")
        print(f"Trainable Parameters: {params['trainable']:,}")
        print("="*60 + "\n")


def load_model_from_path(model_path: str, device: str = "cpu") -> ModelInspector:
    """
    Convenience function to create a ModelInspector instance.
    
    Args:
        model_path: Path to the local model directory
        device: Device to load the model on
        
    Returns:
        ModelInspector instance
    """
    return ModelInspector(model_path, device)


def compare_architectures(model_paths: List[str]) -> None:
    """
    Compare architectures of multiple models side by side.
    
    Args:
        model_paths: List of paths to model directories
    """
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE COMPARISON")
    print("="*80)
    
    inspectors = [ModelInspector(path) for path in model_paths]
    summaries = [inspector.get_architecture_summary() for inspector in inspectors]
    
    # Print comparison table
    keys = ["model_type", "num_layers", "hidden_size", "num_attention_heads", 
            "vocab_size", "max_position_embeddings"]
    
    for key in keys:
        print(f"\n{key.replace('_', ' ').title()}:")
        for path, summary in zip(model_paths, summaries):
            model_name = Path(path).name
            value = summary.get(key, "N/A")
            print(f"  {model_name}: {value}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    model_path = "../models/gpt2_model"
    
    # Create inspector
    inspector = ModelInspector(model_path)
    
    # Load model
    model, tokenizer = inspector.load_model()
    
    # Print architecture summary
    inspector.print_architecture_summary()
    
    # Get specific information
    print(f"Number of layers: {inspector.get_num_layers()}")
    print(f"\nAttention layers: {len(inspector.get_attention_layer_names())}")
    print(f"MLP layers: {len(inspector.get_mlp_layer_names())}")
