"""
Activation Utilities Module

This module provides data structures and utilities for working with model activations:
  - ActivationRecord: Dataclass for storing activations with metadata
  - Save/load functions for multiple formats (pickle, PyTorch, NumPy, JSON)
  - Comparison and statistics utilities

Note: Activation extraction is now integrated directly into ModelAnalyzer.
This module only contains the data structures and utility functions.
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
import pickle


@dataclass
class ActivationRecord:
    """Container for storing activation data with metadata."""
    prompt: str
    tokens: List[str]
    token_ids: List[int]
    layer_activations: Dict[str, torch.Tensor]  # layer_name -> activations
    attention_weights: Optional[Dict[str, torch.Tensor]] = None  # layer_name -> attention
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self, include_tensors: bool = False) -> Dict:
        """
        Convert to dictionary for serialization.
        
        Args:
            include_tensors: Whether to include tensor data (converted to lists)
        """
        result = {
            'prompt': self.prompt,
            'tokens': self.tokens,
            'token_ids': self.token_ids,
            'metadata': self.metadata or {}
        }
        
        if include_tensors:
            # Convert tensors to nested lists for JSON serialization
            result['layer_activations'] = {
                k: v.cpu().detach().numpy().tolist() 
                for k, v in self.layer_activations.items()
            }
            if self.attention_weights:
                result['attention_weights'] = {
                    k: v.cpu().detach().numpy().tolist()
                    for k, v in self.attention_weights.items()
                }
        else:
            # Just include shapes for metadata
            result['layer_activation_shapes'] = {
                k: list(v.shape) for k, v in self.layer_activations.items()
            }
            if self.attention_weights:
                result['attention_weight_shapes'] = {
                    k: list(v.shape) for k, v in self.attention_weights.items()
                }
        
        return result
    
# ===================================================================
# SAVE/LOAD UTILITIES
# ===================================================================

def save_activations(
    record: Union[ActivationRecord, List[ActivationRecord]],
    output_path: str,
    format: str = 'pickle',
    include_tensors: bool = True,
    compress: bool = False
) -> None:
    """
    Save activation record(s) to disk in various formats.
    
    Args:
        record: Single ActivationRecord or list of records
        output_path: Path to save the file
        format: Save format - 'json', 'pickle', 'pt' (PyTorch), or 'npz' (NumPy)
        include_tensors: Whether to save full tensor data (vs just metadata)
        compress: Whether to compress the output (for pickle/npz)
        
    Formats:
        - 'json': Human-readable, no tensor data (just shapes), largest file
        - 'pickle': Full Python object with tensors, good for Python-only use
        - 'pt': PyTorch format, good for loading back into PyTorch
        - 'npz': NumPy compressed format, good for numpy/scientific computing
        
    Example:
        >>> record = analyzer.extract_activations("Test prompt")
        >>> save_activations(record, "activations.pkl", format='pickle')
        >>> save_activations(record, "activations.json", format='json')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    records = [record] if isinstance(record, ActivationRecord) else record
    
    if format == 'json':
        # JSON format - metadata only or with tensor lists
        data = [r.to_dict(include_tensors=include_tensors) for r in records]
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(records)} record(s) to {output_path} (JSON format)")
    
    elif format == 'pickle':
        # Pickle format - full Python objects
        with open(output_path, 'wb') as f:
            if compress:
                import gzip
                with gzip.open(str(output_path) + '.gz', 'wb') as gz:
                    pickle.dump(records, gz)
                output_path = Path(str(output_path) + '.gz')
            else:
                pickle.dump(records, f)
        print(f"Saved {len(records)} record(s) to {output_path} (Pickle format)")
    
    elif format == 'pt':
        # PyTorch format
        data = {
            'prompts': [r.prompt for r in records],
            'tokens': [r.tokens for r in records],
            'token_ids': [r.token_ids for r in records],
            'layer_activations': [r.layer_activations for r in records],
            'attention_weights': [r.attention_weights for r in records],
            'metadata': [r.metadata for r in records]
        }
        torch.save(data, output_path)
        print(f"Saved {len(records)} record(s) to {output_path} (PyTorch format)")
    
    elif format == 'npz':
        # NumPy format - convert tensors to numpy arrays
        data = {}
        for i, record in enumerate(records):
            prefix = f'record_{i}_'
            data[f'{prefix}prompt'] = record.prompt
            data[f'{prefix}tokens'] = np.array(record.tokens, dtype=object)
            data[f'{prefix}token_ids'] = np.array(record.token_ids)
            
            for layer_name, activation in record.layer_activations.items():
                safe_name = layer_name.replace('.', '_')
                data[f'{prefix}layer_{safe_name}'] = activation.cpu().numpy()
            
            if record.attention_weights:
                for layer_name, attn in record.attention_weights.items():
                    safe_name = layer_name.replace('.', '_')
                    data[f'{prefix}attn_{safe_name}'] = attn.cpu().numpy()
        
        if compress:
            np.savez_compressed(output_path, **data)
        else:
            np.savez(output_path, **data)
        print(f"Saved {len(records)} record(s) to {output_path} (NumPy format)")
    
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'json', 'pickle', 'pt', or 'npz'")


def load_activations(
    input_path: str,
    format: Optional[str] = None
) -> Union[ActivationRecord, List[ActivationRecord]]:
    """
    Load activation records from disk.
    
    Args:
        input_path: Path to the saved file
        format: Load format (auto-detected from extension if None)
        
    Returns:
        Single ActivationRecord (if only one was saved) or list of ActivationRecords.
        Always returns ActivationRecord objects for easy use with logit_lens and other analysis functions.
        
    Example:
        >>> records = load_activations("activations.pkl")
        >>> # Use directly with logit lens
        >>> analyzer.logit_lens_on_activation(records, layer_indices=[0, 6, 11])
    """
    input_path = Path(input_path)
    
    if format is None:
        # Auto-detect format from extension
        ext = input_path.suffix.lower()
        if ext == '.json':
            format = 'json'
        elif ext in ['.pkl', '.pickle', '.gz']:
            format = 'pickle'
        elif ext == '.pt':
            format = 'pt'
        elif ext == '.npz':
            format = 'npz'
        else:
            raise ValueError(f"Cannot auto-detect format from extension: {ext}")
    
    if format == 'json':
        with open(input_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} record(s) from {input_path} (JSON format)")
        
        # Convert dict to ActivationRecord (note: JSON format doesn't include tensors by default)
        records = []
        for item in data:
            # Reconstruct tensors if they were saved
            layer_activations = {}
            if 'layer_activations' in item:
                for layer_name, tensor_list in item['layer_activations'].items():
                    layer_activations[layer_name] = torch.tensor(tensor_list)
            
            attention_weights = None
            if 'attention_weights' in item:
                attention_weights = {}
                for layer_name, tensor_list in item['attention_weights'].items():
                    attention_weights[layer_name] = torch.tensor(tensor_list)
            
            records.append(ActivationRecord(
                prompt=item['prompt'],
                tokens=item['tokens'],
                token_ids=item['token_ids'],
                layer_activations=layer_activations,
                attention_weights=attention_weights,
                metadata=item.get('metadata')
            ))
        
        return records[0] if len(records) == 1 else records
    
    elif format == 'pickle':
        if str(input_path).endswith('.gz'):
            import gzip
            with gzip.open(input_path, 'rb') as f:
                records = pickle.load(f)
        else:
            with open(input_path, 'rb') as f:
                records = pickle.load(f)
        print(f"Loaded {len(records)} record(s) from {input_path} (Pickle format)")
        
        # Pickle already stores ActivationRecord objects
        return records[0] if len(records) == 1 else records
    
    elif format == 'pt':
        data = torch.load(input_path)
        print(f"Loaded {len(data['prompts'])} record(s) from {input_path} (PyTorch format)")
        
        # Convert dict structure to ActivationRecord objects
        records = []
        for i in range(len(data['prompts'])):
            records.append(ActivationRecord(
                prompt=data['prompts'][i],
                tokens=data['tokens'][i],
                token_ids=data['token_ids'][i],
                layer_activations=data['layer_activations'][i],
                attention_weights=data['attention_weights'][i],
                metadata=data['metadata'][i]
            ))
        
        return records[0] if len(records) == 1 else records
    
    elif format == 'npz':
        data = np.load(input_path, allow_pickle=True)
        print(f"Loaded activations from {input_path} (NumPy format)")
        
        # Parse NumPy format back into ActivationRecord objects
        # Group by record prefix
        records = []
        record_indices = set()
        for key in data.keys():
            if key.startswith('record_'):
                idx = int(key.split('_')[1])
                record_indices.add(idx)
        
        for i in sorted(record_indices):
            prefix = f'record_{i}_'
            
            # Extract layer activations
            layer_activations = {}
            for key in data.keys():
                if key.startswith(f'{prefix}layer_'):
                    layer_name = key.replace(f'{prefix}layer_', '').replace('_', '.')
                    layer_activations[layer_name] = torch.from_numpy(data[key])
            
            # Extract attention weights if present
            attention_weights = None
            attn_keys = [k for k in data.keys() if k.startswith(f'{prefix}attn_')]
            if attn_keys:
                attention_weights = {}
                for key in attn_keys:
                    layer_name = key.replace(f'{prefix}attn_', '').replace('_', '.')
                    attention_weights[layer_name] = torch.from_numpy(data[key])
            
            records.append(ActivationRecord(
                prompt=str(data[f'{prefix}prompt']),
                tokens=data[f'{prefix}tokens'].tolist(),
                token_ids=data[f'{prefix}token_ids'].tolist(),
                layer_activations=layer_activations,
                attention_weights=attention_weights,
                metadata=None
            ))
        
        return records[0] if len(records) == 1 else records
    
    else:
        raise ValueError(f"Unsupported format: {format}")


#===================================================================
# COMPARISON AND STATISTICS UTILITIES
# ===================================================================

def compare_activations(
    record1: ActivationRecord,
    record2: ActivationRecord,
    layer_name: str,
    metric: str = 'cosine',
    position: int = -1
) -> float:
    """
    Compare activations between two records for a specific layer.
    
    Note: Compares activations at a specific token position (default: last token)
    to handle different sequence lengths.
    
    Args:
        record1: First activation record
        record2: Second activation record
        layer_name: Name of layer to compare
        metric: Similarity metric ('cosine', 'l2', 'l1')
        position: Token position to compare (-1 for last token, 0 for first, etc.)
        
    Returns:
        Similarity/distance score
        
    Example:
        >>> rec1 = extractor.extract_activations("The cat")
        >>> rec2 = extractor.extract_activations("The dog")
        >>> sim = compare_activations(rec1, rec2, 'h.0', metric='cosine')
        >>> print(f"Cosine similarity: {sim:.4f}")
    """
    if layer_name not in record1.layer_activations:
        raise ValueError(f"Layer {layer_name} not found in record1")
    if layer_name not in record2.layer_activations:
        raise ValueError(f"Layer {layer_name} not found in record2")
    
    # Get activations at specific position [batch, seq_len, hidden_dim]
    act1 = record1.layer_activations[layer_name][0, position, :]
    act2 = record2.layer_activations[layer_name][0, position, :]
    
    if metric == 'cosine':
        # Cosine similarity
        sim = torch.nn.functional.cosine_similarity(
            act1.unsqueeze(0), 
            act2.unsqueeze(0)
        )
        return sim.item()
    elif metric == 'l2':
        # L2 distance
        return torch.norm(act1 - act2, p=2).item()
    elif metric == 'l1':
        # L1 distance
        return torch.norm(act1 - act2, p=1).item()
    else:
        raise ValueError(f"Unknown metric: {metric}")


def get_activation_statistics(
    record: ActivationRecord,
    layer_name: str
) -> Dict[str, float]:
    """
    Compute statistics for activations in a specific layer.
    
    Args:
        record: Activation record
        layer_name: Name of layer to analyze
        
    Returns:
        Dictionary of statistics (mean, std, min, max, norm)
        
    Example:
        >>> record = extractor.extract_activations("Test")
        >>> stats = get_activation_statistics(record, 'transformer.h.5')
        >>> print(f"Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
    """
    if layer_name not in record.layer_activations:
        raise ValueError(f"Layer {layer_name} not found in record")
    
    act = record.layer_activations[layer_name]
    
    return {
        'mean': act.mean().item(),
        'std': act.std().item(),
        'min': act.min().item(),
        'max': act.max().item(),
        'norm': torch.norm(act).item(),
        'shape': list(act.shape)
    }
