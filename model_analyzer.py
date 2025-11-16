"""
Unified Model Analyzer for Mechanistic Interpretability

This module provides a comprehensive ModelAnalyzer class that combines:
- Model inspection and architecture analysis
- Activation extraction during forward passes
- Logit lens analysis
- Causal interventions (activation patching and steering)
- Text generation with optional recording and intervention

All operations share a single loaded model for efficiency.
"""

import torch
from typing import Dict, List, Optional, Union, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np

from transformers import (
    AutoModel, 
    AutoModelForCausalLM,
    AutoTokenizer, 
    AutoConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)

# Import components from existing modules
from activation_extraction import (
    ActivationRecord,
    save_activations,
    load_activations,
    compare_activations,
    get_activation_statistics
)
from intervention import (
    InterventionHandler,
    ActivationPatch,
    SteeringVector,
    create_steering_vector as _create_steering_vector
)


class ModelAnalyzer:
    """
    Unified interface for mechanistic interpretability analysis.
    
    This class loads a model once and provides methods for:
    - Text generation (with optional activation recording)
    - Activation extraction and analysis
    - Logit lens analysis
    - Causal interventions (patching and steering)
    - Model architecture inspection
    
    Example:
        >>> analyzer = ModelAnalyzer("../models/gpt2_model")
        >>> 
        >>> # Generate text
        >>> text = analyzer.generate("The capital of France is")
        >>> 
        >>> # Generate with activation recording
        >>> text, activations = analyzer.generate("Hello", return_activations=True)
        >>> 
        >>> # Apply logit lens
        >>> predictions = analyzer.logit_lens("The quick brown")
        >>> 
        >>> # Generate with intervention
        >>> text = analyzer.generate_with_steering(
        ...     prompt="Today I feel",
        ...     positive_examples=["I am happy"],
        ...     negative_examples=["I am sad"]
        ... )
    """
    
    def __init__(
        self, 
        model_path: str,
        device: Optional[str] = None,
        load_for_generation: bool = True
    ):
        """
        Initialize the ModelAnalyzer with a single model instance.
        
        Args:
            model_path: Path to the local model directory
            device: Device to load model on (None = auto-detect)
            load_for_generation: If True, loads AutoModelForCausalLM for text generation
                                If False, loads AutoModel (no generation capability)
        """
        self.model_path = Path(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_for_generation = load_for_generation
        
        print(f"Initializing ModelAnalyzer for {self.model_path}")
        print(f"Device: {self.device}")
        
        self.config = None
        self.tokenizer = None
        self.model = None
        self.layer_prefix = None
        # Load model and tokenizer
        #self.load_model()
        #self._detect_layer_naming()
        
        # Initialize intervention handler (created on-demand)
        self._intervention_handler = None
        
        # Cache for architecture info
        self._layer_names_cache = None
        self._num_layers_cache = None
        
        print(f"[OK] ModelAnalyzer ready")
    
    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading model from {self.model_path}...")
        
        # Load config
        self.config = AutoConfig.from_pretrained(str(self.model_path))

        outputting_attns = False
        if hasattr(self.config, "attn_implementation"):
            #self.config.attn_implementation = "eager"
            setattr(self.config, "attn_implementation", "eager")
            print("Set attn_implementation to eager in config")
            outputting_attns = True
        elif hasattr(self.config, "_attn_implementation"):
            #self.config._attn_implementation = "eager"
            setattr(self.config, "_attn_implementation", "eager")
            print("Set _attn_implementation to eager in config")
            outputting_attns = True

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model (for generation or not)
        if self.load_for_generation:
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True
            ).to(self.device)
        else:
            self.model = AutoModel.from_pretrained(
                str(self.model_path),
                local_files_only=True
            ).to(self.device)

        #if not outputting_attns:
        try:
            self.model.set_attn_implementation("eager")
            print("Set attn_implementation to eager in model")
        except:
            print("Could not set attn_implementation in model")

        self._detect_layer_naming()

        self.model.eval()
        print(f"[OK] Model loaded on {self.device}")
    
    def _detect_layer_naming(self):
        """Detect the layer naming convention used by this model."""
        # Common patterns: transformer.h.X, model.layers.X, encoder.layer.X, etc.
        for name, _ in self.model.named_modules():
            if 'transformer.h.' in name:
                self.layer_prefix = 'transformer.h.'
                return
            elif 'model.layers.' in name:
                self.layer_prefix = 'model.layers.'
                return
            elif 'encoder.layer.' in name:
                self.layer_prefix = 'encoder.layer.'
                return
            elif 'decoder.layers.' in name:
                self.layer_prefix = 'decoder.layers.'
                return
        
        # Default fallback
        self.layer_prefix = 'layer_'
    
    def _get_layer_names_from_indices(self, num_layers: int) -> List[str]:
        """Generate layer names based on detected architecture."""
        return [f"{self.layer_prefix}{i}" for i in range(num_layers)]
    
    # ===================================================================
    # TEXT GENERATION
    # ===================================================================
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 50,
        **generation_kwargs
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            **generation_kwargs: Additional arguments for model.generate()
        
        Returns:
            Generated text string
        
        Note:
            To analyze activations during generation, use extract_activations() 
            on the prompt or generated text, or use analyze_generation_token() 
            for high-level token analysis.
        
        Example:
            >>> text = analyzer.generate("Once upon a time")
            >>> # For activation analysis:
            >>> activations = analyzer.extract_activations("Once upon a time")
            >>> # Or for token-specific analysis:
            >>> result = analyzer.analyze_generation_token("Once upon a time", target_token="upon")
        """
        if not self.load_for_generation:
            raise RuntimeError("Model was not loaded for generation. Initialize with load_for_generation=True")
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
                **generation_kwargs
            )
        
        # Decode and return generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

        
    
    # ===================================================================
    # ACTIVATION EXTRACTION
    # ===================================================================
    
    def extract_activations(
        self,
        text: str,
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        include_attention: bool = False,
        return_logits: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ActivationRecord:
        """
        Extract activations from a forward pass using HuggingFace's output_hidden_states.
        
        Args:
            prompt: Input text
            layer_names: Specific layer names to extract (for compatibility)
            layer_indices: Specific layer indices to extract (None = all layers)
            include_attention: Whether to extract attention weights
            return_logits: Whether to include final logits in metadata
            metadata: Additional metadata to store
        
        Returns:
            ActivationRecord with layer activations and metadata
        
        Example:
            >>> record = analyzer.extract_activations("The cat sat")
            >>> activation = record.layer_activations['transformer.h.5']
        """
        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        token_ids = inputs['input_ids'][0].tolist()
        
        # Use HuggingFace's built-in hidden states extraction!
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                output_attentions=include_attention
            )
        
        # Extract hidden states (tuple of tensors, one per layer including embedding)
        hidden_states = outputs.hidden_states
        
        # Convert to dict with layer names
        layer_activations = {}
        if layer_indices is not None:
            # Extract specific layers
            for idx in layer_indices:
                if 0 <= idx < len(hidden_states):
                    layer_name = f"{self.layer_prefix}{idx}"
                    layer_activations[layer_name] = hidden_states[idx].detach()
        elif layer_names is not None:
            # For backward compatibility: map layer names to indices
            for name in layer_names:
                # Extract layer number from name
                parts = name.split('.')
                if parts[-1].isdigit():
                    idx = int(parts[-1])
                    if 0 <= idx < len(hidden_states):
                        layer_activations[name] = hidden_states[idx].detach()
        else:
            # Extract all layers
            for idx, hidden_state in enumerate(hidden_states):
                layer_name = f"{self.layer_prefix}{idx}"
                layer_activations[layer_name] = hidden_state.detach()
        
        # Prepare metadata
        record_metadata = metadata or {}
        if return_logits and hasattr(outputs, 'logits'):
            record_metadata['logits_shape'] = list(outputs.logits.shape)
            record_metadata['final_logits'] = outputs.logits[0, -1].cpu().numpy().tolist()
        
        record_metadata['num_tokens'] = len(tokens)
        record_metadata['num_layers'] = len(hidden_states)
        record_metadata['model_name'] = self.model.config.model_type if hasattr(self.model.config, 'model_type') else 'unknown'
        
        # Handle attention weights
        attention_dict = None
        if include_attention:
            attention_dict = {
                f"attention_layer_{i}": attn.detach()
                for i, attn in enumerate(outputs.attentions)
            }
        
        # Create record
        record = ActivationRecord(
            prompt=text,
            tokens=tokens,
            token_ids=token_ids,
            layer_activations=layer_activations,
            attention_weights=attention_dict,
            metadata=record_metadata
        )
        
        return record
    
    def extract_batch_activations(
        self,
        prompts: List[str],
        layer_names: Optional[List[str]] = None,
        layer_indices: Optional[List[int]] = None,
        include_attention: bool = False,
        show_progress: bool = True
    ) -> List[ActivationRecord]:
        """
        Extract activations for multiple prompts.
        
        Args:
            prompts: List of input texts
            layer_names: Specific layer names to extract
            layer_indices: Specific layer indices to extract
            include_attention: Whether to extract attention weights
            show_progress: Whether to print progress
        
        Returns:
            List of ActivationRecord objects
        """
        records = []
        for i, prompt in enumerate(prompts):
            if show_progress:
                print(f"Processing prompt {i+1}/{len(prompts)}: {prompt[:50]}...")
            
            record = self.extract_activations(
                prompt=prompt,
                layer_names=layer_names,
                layer_indices=layer_indices,
                include_attention=include_attention
            )
            records.append(record)
        
        return records
    
    def save_activations(
        self,
        record: Union[ActivationRecord, List[ActivationRecord]],
        output_path: str,
        format: str = 'pickle',
        **kwargs
    ):
        """
        Save activation records to disk.
        
        Args:
            record: Single record or list of records
            output_path: Path to save file
            format: 'pickle', 'pt', 'npz', or 'json'
            **kwargs: Additional arguments for save_activations
        """
        save_activations(
            record=record,
            output_path=output_path,
            format=format,
            **kwargs
        )
    
    # ===================================================================
    # LOGIT LENS
    # ===================================================================
    
    def logit_lens(
        self,
        text: str,
        layer_indices: Optional[List[int]] = None,
        top_k: int = 10,
        apply_ln: bool = True
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Apply logit lens to see predictions at each layer.
        
        Args:
            text: Input text to analyze
            layer_indices: Specific layer indices (None = all layers)
            top_k: Number of top predictions to return per layer
            apply_ln: Whether to apply layer normalization
        
        Returns:
            Dictionary mapping layer_idx -> list of (token, probability) tuples
        
        Example:
            >>> predictions = analyzer.logit_lens("The capital of France is")
            >>> for layer_idx, preds in predictions.items():
            ...     print(f"Layer {layer_idx}: {preds[0]}")
        """
        # Get model components
        if hasattr(self.model, 'transformer'):
            transformer = self.model.transformer
            lm_head = self.model.lm_head if hasattr(self.model, 'lm_head') else None
        elif hasattr(self.model, 'model'):
            transformer = self.model.model
            lm_head = self.model.lm_head if hasattr(self.model, 'lm_head') else None
        else:
            raise ValueError("Cannot find transformer layers in model")
        
        # Get layer norm
        ln_f = transformer.ln_f if hasattr(transformer, 'ln_f') else None
        
        # Get embedding matrix for unembedding
        if lm_head is not None:
            unembed_matrix = lm_head.weight
        elif hasattr(self.model, 'get_output_embeddings'):
            unembed_matrix = self.model.get_output_embeddings().weight
        else:
            # Fallback: use input embeddings transposed
            unembed_matrix = self.model.get_input_embeddings().weight
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        # Determine layers to analyze
        num_layers = self.get_num_layers()
        if layer_indices is None:
            layer_indices = list(range(num_layers))
        
        results = {}
        
        with torch.no_grad():
            # Get hidden states at each layer
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
            hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden_dim)
            
            for layer_idx in layer_indices:
                if layer_idx >= len(hidden_states):
                    continue
                
                # Get hidden state at this layer
                hidden = hidden_states[layer_idx]
                
                # Take last token
                last_hidden = hidden[0, -1, :]
                
                # Apply layer norm if requested
                if apply_ln and ln_f is not None:
                    last_hidden = ln_f(last_hidden)
                
                # Project to vocabulary
                logits = torch.matmul(last_hidden, unembed_matrix.T)
                probs = torch.softmax(logits, dim=-1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probs, top_k)
                
                predictions = [
                    (self.tokenizer.decode([idx.item()]), prob.item())
                    for idx, prob in zip(top_indices, top_probs)
                ]
                
                results[layer_idx] = predictions
        
        return results
    
    def print_logit_lens(
        self,
        text: str,
        layer_step: int = 1,
        top_k: int = 5
    ):
        """
        Print logit lens predictions in a readable format.
        
        Args:
            text: Input text
            layer_step: Step between layers to print
            top_k: Number of top predictions per layer
        
        Example:
            >>> analyzer.print_logit_lens("The capital of France is", layer_step=2)
        """
        num_layers = self.get_num_layers()
        layer_indices = list(range(0, num_layers, layer_step))
        
        predictions = self.logit_lens(text, layer_indices=layer_indices, top_k=top_k)
        
        print(f"\nLogit Lens Analysis for: '{text}'")
        print("="*70)

        for layer_idx in sorted(predictions.keys()):
            print(f"\nLayer {layer_idx}:")
            for i, (token, prob) in enumerate(predictions[layer_idx][:top_k], 1):
                print(f"  {i}. '{token}' ({prob:.4f})")
    
    def logit_lens_on_activation(
        self,
        activation: torch.Tensor,
        layer_indices: Optional[List[int]] = None,
        top_k: int = 10,
        apply_ln: bool = True,
        token_position: int = -1
    ) -> Dict[str, Any]:
        """
        Apply logit lens to a specific activation tensor (from generation or extraction).
        
        This allows you to analyze what the model "thinks" a specific token is at each layer,
        useful for analyzing tokens in the middle of a generation.
        
        Args:
            activation: ActivationRecord from extract_activations or generation
            layer_indices: Specific layer indices to analyze (None = all layers)
            top_k: Number of top predictions to return per layer
            apply_ln: Whether to apply layer normalization before unembedding
            token_position: Which token position to analyze (-1 for last, 0 for first, etc.)
        
        Returns:
            Dictionary with:
                - 'layers': Dict mapping layer_idx -> {'top_k_tokens', 'top_k_probs', 'logits'}
                - 'metadata': Information about the analysis
        
        Example:
            >>> # Generate and capture activations
            >>> output, acts = analyzer.generate("What is the capital of France?", 
            ...                                   max_new_tokens=20, return_activations=True)
            >>> # Find the token you're interested in
            >>> tokens = analyzer.tokenizer.encode(output)
            >>> paris_pos = tokens.index(analyzer.tokenizer.encode("Paris")[0])
            >>> # Analyze that specific token
            >>> results = analyzer.logit_lens_on_activation(acts, token_position=paris_pos)
        """
        # Import here to avoid circular dependency
        from activation_extraction import ActivationRecord
        
        if isinstance(activation, ActivationRecord):
            layer_activations = activation.layer_activations
            num_tokens = len(activation.tokens)
        else:
            raise ValueError("activation must be an ActivationRecord")
        
        # Get model components for unembedding
        if hasattr(self.model, 'transformer'):
            transformer = self.model.transformer
            lm_head = self.model.lm_head if hasattr(self.model, 'lm_head') else None
        elif hasattr(self.model, 'model'):
            transformer = self.model.model
            lm_head = self.model.lm_head if hasattr(self.model, 'lm_head') else None
        else:
            raise ValueError("Cannot find transformer layers in model")
        
        # Get layer norm
        ln_f = transformer.ln_f if hasattr(transformer, 'ln_f') else None
        
        # Get embedding matrix for unembedding
        if lm_head is not None:
            unembed_matrix = lm_head.weight
        elif hasattr(self.model, 'get_output_embeddings'):
            unembed_matrix = self.model.get_output_embeddings().weight
        else:
            unembed_matrix = self.model.get_input_embeddings().weight
        
        # Determine layers to analyze
        if layer_indices is None:
            layer_indices = list(range(len(layer_activations)))
        
        results = {'layers': {}, 'metadata': {}}
        
        with torch.no_grad():
            for layer_name, hidden_state in layer_activations.items():
                # Extract layer index from name
                parts = layer_name.split('.')
                if parts[-1].isdigit():
                    layer_idx = int(parts[-1])
                else:
                    continue
                
                if layer_indices is not None and layer_idx not in layer_indices:
                    continue
                
                # Get activation at specific token position
                # Shape is typically [batch, seq_len, hidden_dim]
                if len(hidden_state.shape) == 3:
                    token_hidden = hidden_state[0, token_position, :]
                elif len(hidden_state.shape) == 2:
                    token_hidden = hidden_state[token_position, :]
                else:
                    continue
                
                # Apply layer norm if requested
                if apply_ln and ln_f is not None:
                    token_hidden = ln_f(token_hidden)
                
                # Project to vocabulary
                logits = torch.matmul(token_hidden, unembed_matrix.T)
                probs = torch.softmax(logits, dim=-1)
                
                # Get top-k predictions
                top_probs, top_indices = torch.topk(probs, top_k)
                
                top_tokens = [self.tokenizer.decode([idx.item()]) for idx in top_indices]
                top_probs_list = [prob.item() for prob in top_probs]
                
                results['layers'][layer_idx] = {
                    'top_k_tokens': top_tokens,
                    'top_k_probs': top_probs_list,
                    'logits': logits.cpu()
                }
        
        results['metadata'] = {
            'token_position': token_position,
            'num_tokens': num_tokens,
            'num_layers_analyzed': len(results['layers'])
        }
        
        return results
    
    def analyze_generation_token(
        self,
        prompt: str,
        target_token: Optional[str] = None,
        token_position: Optional[int] = None,
        max_new_tokens: int = 50,
        layer_step: int = 1,
        top_k: int = 5,
        **generation_kwargs
    ) -> Dict[str, Any]:
        """
        Generate text and analyze a specific token using logit lens.
        
        This is a high-level convenience method that:
        1. Generates text from the prompt
        2. Finds the target token (or uses token_position)
        3. Extracts activations during generation
        4. Applies logit lens to that specific token
        
        Args:
            prompt: Input prompt
            target_token: Token to search for (e.g., "Paris"). If None, uses token_position
            token_position: Absolute position of token to analyze (0-indexed). 
                          If None, searches for target_token
            max_new_tokens: Maximum tokens to generate
            layer_step: Step between layers for analysis
            top_k: Number of top predictions per layer
            **generation_kwargs: Additional arguments for generation
        
        Returns:
            Dictionary with:
                - 'generated_text': The full generated text
                - 'target_token': The token being analyzed
                - 'token_position': Position of the analyzed token
                - 'analysis': Logit lens results for that token
        
        Example:
            >>> # Analyze the "Paris" token in generation
            >>> result = analyzer.analyze_generation_token(
            ...     "What is the capital of France?",
            ...     target_token="Paris",
            ...     max_new_tokens=20
            ... )
            >>> print(result['generated_text'])
            >>> print(f"Analyzing token: {result['target_token']} at position {result['token_position']}")
            >>> for layer_idx, data in result['analysis']['layers'].items():
            ...     print(f"Layer {layer_idx}: {data['top_k_tokens'][0]}")
        """
        if target_token is None and token_position is None:
            raise ValueError("Must provide either target_token or token_position")
        
        # Generate text
        generated_text = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            **generation_kwargs
        )
        
        # Extract activations for the full generated sequence
        activations_record = self.extract_activations(generated_text)
        
        # If we need to find the target token
        if token_position is None:
            # Tokenize the full output to find the target
            full_tokens = self.tokenizer.encode(generated_text)
            token_strs = [self.tokenizer.decode([t]) for t in full_tokens]
            
            # Search for target token
            target_encoding = self.tokenizer.encode(target_token, add_special_tokens=False)
            
            found_position = None
            for i, token_id in enumerate(full_tokens):
                if token_id in target_encoding:
                    found_position = i
                    break
            
            if found_position is None:
                # Try string matching
                for i, tok_str in enumerate(token_strs):
                    if target_token.lower() in tok_str.lower():
                        found_position = i
                        break
            
            if found_position is None:
                raise ValueError(f"Could not find token '{target_token}' in generated text: {generated_text}")
            
            token_position = found_position
            actual_token = token_strs[token_position]
        else:
            # Use provided position
            full_tokens = self.tokenizer.encode(generated_text)
            token_strs = [self.tokenizer.decode([t]) for t in full_tokens]
            if token_position >= len(token_strs):
                raise ValueError(f"token_position {token_position} out of range (only {len(token_strs)} tokens)")
            actual_token = token_strs[token_position]
        
        # Analyze this specific token
        num_layers = self.get_num_layers()
        layer_indices = list(range(0, num_layers, layer_step))
        
        analysis = self.logit_lens_on_activation(
            activations_record,
            layer_indices=layer_indices,
            top_k=top_k,
            token_position=token_position
        )
        
        return {
            'generated_text': generated_text,
            'target_token': actual_token,
            'token_position': token_position,
            'all_tokens': token_strs,
            'analysis': analysis
        }
    
    # ===================================================================
    # INTERVENTIONS
    # ===================================================================
    
    def generate_with_patching(
        self,
        prompt: str,
        patches: List[ActivationPatch],
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> str:
        """
        Generate text with activation patches applied.
        
        Args:
            prompt: Input prompt
            patches: List of ActivationPatch objects to apply
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation arguments
        
        Returns:
            Generated text with patches applied
        
        Example:
            >>> patch = ActivationPatch('h.5', -1, some_activation, 'add')
            >>> text = analyzer.generate_with_patching("Test", [patch])
        """
        if not self.load_for_generation:
            raise RuntimeError("Model not loaded for generation")
        
        # Create intervention handler
        handler = InterventionHandler(self.model, self.tokenizer)
        
        # Register patches
        for patch in patches:
            handler.register_activation_patch(patch)
        
        # Generate with patches
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with handler:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generation_kwargs
                )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def generate_with_steering(
        self,
        prompt: str,
        positive_examples: List[str],
        negative_examples: List[str],
        layer_name: Optional[str] = None,
        coefficient: float = 2.0,
        max_new_tokens: int = 50,
        **generation_kwargs
    ) -> str:
        """
        Generate text with steering vector applied.
        
        Creates a steering vector from contrastive examples and applies it during generation.
        
        Args:
            prompt: Input prompt
            positive_examples: Examples of desired behavior
            negative_examples: Examples of undesired behavior
            layer_name: Layer to apply steering (None = middle layer)
            coefficient: Steering strength
            max_new_tokens: Maximum tokens to generate
            **generation_kwargs: Additional generation arguments
        
        Returns:
            Generated text with steering applied
        
        Example:
            >>> text = analyzer.generate_with_steering(
            ...     prompt="Today I feel",
            ...     positive_examples=["I am happy", "This is great"],
            ...     negative_examples=["I am sad", "This is terrible"]
            ... )
        """
        if not self.load_for_generation:
            raise RuntimeError("Model not loaded for generation")
        
        # Determine layer if not specified
        if layer_name is None:
            num_layers = self.get_num_layers()
            middle_layer = num_layers // 2
            layer_names = self.get_layer_names()
            layer_name = layer_names[middle_layer] if middle_layer < len(layer_names) else layer_names[0]
        
        # Create steering vector
        steering_vec = _create_steering_vector(
            self.model,
            self.tokenizer,
            positive_prompts=positive_examples,
            negative_prompts=negative_examples,
            layer_name=layer_name
        )
        
        # Generate with steering
        from intervention import steer_generation
        
        return steer_generation(
            self.model,
            self.tokenizer,
            prompt=prompt,
            steering_vector=steering_vec,
            layer_name=layer_name,
            coefficient=coefficient,
            max_length=max_new_tokens,
            **generation_kwargs
        )
    
    def path_patching(
        self,
        clean_prompt: str,
        corrupted_prompt: str,
    ) -> Dict[int, float]:
        """
        Test causal importance of each layer via path patching.
        
        Args:
            clean_prompt: Prompt that produces desired behavior
            corrupted_prompt: Prompt that produces undesired behavior
            metric: How to measure effect ('logit_diff', 'prob_diff')
        
        Returns:
            Dictionary mapping layer_idx -> causal_effect
        
        Example:
            >>> effects = analyzer.path_patching(
            ...     "The Eiffel Tower is in",
            ...     "The Colosseum is in"
            ... )
        """
        from intervention import path_patching as _path_patching
        
        return _path_patching(
            self.model,
            self.tokenizer,
            clean_prompt=clean_prompt,
            corrupted_prompt=corrupted_prompt,
        )
    
    # ===================================================================
    # MODEL INSPECTION
    # ===================================================================
    
    def get_layer_names(self) -> List[str]:
        """Get names of all transformer layers."""
        if self._layer_names_cache is not None:
            return self._layer_names_cache
        
        layer_names = []
        for name, module in self.model.named_modules():
            if ('h.' in name or 'layer.' in name or 'block.' in name or 'layers.' in name):
                parts = name.split('.')
                if len(parts) >= 2 and parts[-1].isdigit():
                    if len(parts) == 2 or (len(parts) == 3 and parts[0] in ['transformer', 'model']):
                        layer_names.append(name)
        
        self._layer_names_cache = sorted(set(layer_names), key=lambda x: int(x.split('.')[-1]))
        return self._layer_names_cache
    
    def get_attention_layer_names(self) -> List[str]:
        """
        Get names of attention layers specifically.
        
        Returns:
            List of attention layer names
        
        Example:
            >>> attn_layers = analyzer.get_attention_layer_names()
            >>> print(f"Found {len(attn_layers)} attention layers")
        """
        attn_layers = []
        for name, module in self.model.named_modules():
            module_type = type(module).__name__.lower()
            if 'attention' in module_type or 'attn' in name.lower():
                attn_layers.append(name)
        return attn_layers
    
    def get_mlp_layer_names(self) -> List[str]:
        """
        Get names of MLP/feedforward layers.
        
        Returns:
            List of MLP layer names
        
        Example:
            >>> mlp_layers = analyzer.get_mlp_layer_names()
            >>> print(f"Found {len(mlp_layers)} MLP layers")
        """
        mlp_layers = []
        for name, module in self.model.named_modules():
            if 'mlp' in name.lower() or 'feed_forward' in name.lower():
                mlp_layers.append(name)
        return mlp_layers
    
    def get_layer_by_name(self, layer_name: str):
        """
        Get a specific layer module by its name.
        
        Args:
            layer_name: Name of the layer to retrieve
            
        Returns:
            The layer module
        
        Example:
            >>> layer = analyzer.get_layer_by_name('h.0')
            >>> print(type(layer))
        """
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
        
        Example:
            >>> weights = analyzer.get_weights_by_layer('h.0')
            >>> for name, tensor in weights.items():
            ...     print(f"{name}: {tensor.shape}")
        """
        layer = self.get_layer_by_name(layer_name)
        return {name: param.data for name, param in layer.named_parameters()}
    
    def get_num_layers(self) -> int:
        """Get names of all transformer layers."""
        if self._layer_names_cache is not None:
            return self._layer_names_cache
        
        layer_names = []
        for name, module in self.model.named_modules():
            if ('h.' in name or 'layer.' in name or 'block.' in name or 'layers.' in name):
                parts = name.split('.')
                if len(parts) >= 2 and parts[-1].isdigit():
                    if len(parts) == 2 or (len(parts) == 3 and parts[0] in ['transformer', 'model']):
                        layer_names.append(name)
        
        self._layer_names_cache = sorted(set(layer_names), key=lambda x: int(x.split('.')[-1]))
        return self._layer_names_cache
    
    def get_num_layers(self) -> int:
        """Get number of transformer layers."""
        if self._num_layers_cache is not None:
            return self._num_layers_cache
        
        self._num_layers_cache = len(self.get_layer_names())
        return self._num_layers_cache
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the full model configuration as a dictionary.
        
        Returns:
            Dictionary containing model configuration
            
        Example:
            >>> analyzer = ModelAnalyzer("../models/gpt2_model")
            >>> config = analyzer.get_model_config()
            >>> print(f"Vocab size: {config['vocab_size']}")
        """
        return self.config.to_dict()
    
    def get_parameter_count(self) -> Dict[str, int]:
        """
        Get detailed parameter counts for the model.
        
        Returns:
            Dictionary with 'total', 'trainable', and 'non_trainable' parameter counts
            
        Example:
            >>> analyzer = ModelAnalyzer("../models/gpt2_model")
            >>> counts = analyzer.get_parameter_count()
            >>> print(f"Total: {counts['total']:,}")
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "total": total_params,
            "trainable": trainable_params,
            "non_trainable": total_params - trainable_params
        }
    
    def get_architecture_summary(self) -> Dict[str, Any]:
        """
        Get summary of model architecture.
        
        Returns:
            Dictionary with architecture information
        """
        summary = {
            'model_type': self.config.model_type if hasattr(self.config, 'model_type') else 'unknown',
            'num_layers': self.get_num_layers(),
            'hidden_size': self.config.hidden_size if hasattr(self.config, 'hidden_size') else None,
            'num_attention_heads': self.config.num_attention_heads if hasattr(self.config, 'num_attention_heads') else None,
            'vocab_size': self.config.vocab_size if hasattr(self.config, 'vocab_size') else None,
            'max_position_embeddings': self.config.max_position_embeddings if hasattr(self.config, 'max_position_embeddings') else None,
        }
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary['total_parameters'] = total_params
        summary['trainable_parameters'] = trainable_params
        
        return summary
    
    def print_architecture_summary(self):
        """Print a formatted architecture summary."""
        summary = self.get_architecture_summary()
        
        print("\n" + "="*70)
        print("MODEL ARCHITECTURE SUMMARY")
        print("="*70)
        
        print(f"\nModel Type: {summary['model_type']}")
        print(f"Number of Layers: {summary['num_layers']}")
        print(f"Hidden Size: {summary['hidden_size']}")
        print(f"Attention Heads: {summary['num_attention_heads']}")
        print(f"Vocabulary Size: {summary['vocab_size']}")
        print(f"Max Position Embeddings: {summary['max_position_embeddings']}")
        
        print(f"\nTotal Parameters: {summary['total_parameters']:,}")
        print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
        
        print("\nLayer Names:")
        for i, layer_name in enumerate(self.get_layer_names()):
            print(f"  {i}: {layer_name}")
        
        print("="*70 + "\n")
    
    # ===================================================================
    # UTILITY METHODS
    # ===================================================================
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text and return tensor dict."""
        return self.tokenizer(text, return_tensors="pt").to(self.device)
    
    def decode(self, token_ids: torch.Tensor) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    def get_tokens(self, text: str) -> List[str]:
        """Get list of token strings for text."""
        token_ids = self.tokenizer.encode(text)
        return self.tokenizer.convert_ids_to_tokens(token_ids)
