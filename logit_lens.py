"""
Logit Lens Implementation for Mechanistic Interpretability

The logit lens technique projects intermediate layer activations through the 
unembedding matrix to see what the model "believes" at each layer.

Based on: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


class LogitLens:
    """
    Apply the logit lens technique to analyze transformer model internals.
    
    The logit lens projects intermediate activations back to vocabulary space
    to understand what tokens the model is "thinking about" at each layer.
    """
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize LogitLens with a model.
        
        Args:
            model_path: Path to the local model directory
            device: Device to load model on
        """
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            output_hidden_states=True
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )
        self.model.eval()
        
        # Get the unembedding matrix (final layer that projects to vocab)
        # In GPT-2, this is tied to the embedding matrix
        self.unembedding = self._get_unembedding_matrix()
        
        # Get layer norm for proper projection
        self.final_layer_norm = self._get_final_layer_norm()
        
    def _get_unembedding_matrix(self) -> torch.Tensor:
        """
        Extract the unembedding matrix (projects hidden states to vocabulary).
        
        Returns:
            Unembedding weight matrix
        """
        # For GPT-2 and similar models, the LM head is tied to embeddings
        if hasattr(self.model, 'lm_head'):
            return self.model.lm_head.weight
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            # Some models tie weights, so we use the embedding matrix transposed
            return self.model.transformer.wte.weight
        else:
            raise ValueError("Could not find unembedding matrix in model")
    
    def _get_final_layer_norm(self):
        """
        Get the final layer normalization used before unembedding.
        
        Returns:
            Layer norm module
        """
        if hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'ln_f'):
            return self.model.transformer.ln_f
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'norm'):
            return self.model.model.norm
        else:
            return None  # Some models may not have this
    
    def apply_lens_to_hidden_state(self, hidden_state: torch.Tensor, 
                                   apply_ln: bool = True) -> torch.Tensor:
        """
        Apply the logit lens: project hidden state to vocabulary logits.
        
        Args:
            hidden_state: Hidden state tensor [batch, seq_len, hidden_dim]
            apply_ln: Whether to apply layer normalization before projection
            
        Returns:
            Logits over vocabulary [batch, seq_len, vocab_size]
        """
        if apply_ln and self.final_layer_norm is not None:
            hidden_state = self.final_layer_norm(hidden_state)
        
        # Project to vocabulary space
        logits = torch.matmul(hidden_state, self.unembedding.T)
        return logits
    
    @torch.no_grad()
    def get_layer_predictions(self, text: str, 
                             layer_indices: Optional[List[int]] = None) -> Dict[str, any]:
        """
        Get top predictions at each layer for given text.
        
        Args:
            text: Input text string
            layer_indices: Which layers to analyze (None = all layers)
            
        Returns:
            Dictionary containing predictions and metadata
        """
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids']
        
        # Get model outputs with hidden states
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (layer) tensors
        
        # Determine which layers to analyze
        if layer_indices is None:
            layer_indices = list(range(len(hidden_states)))
        
        # Get predictions for each layer
        results = {
            'input_text': text,
            'input_tokens': self.tokenizer.convert_ids_to_tokens(input_ids[0]),
            'input_ids': input_ids[0].cpu().numpy(),
            'layers': {},
            'final_logits': outputs.logits[0].cpu()
        }
        
        for layer_idx in layer_indices:
            hidden_state = hidden_states[layer_idx]
            
            # Apply logit lens
            logits = self.apply_lens_to_hidden_state(hidden_state)
            logits = logits[0]  # Remove batch dimension
            
            # Get top predictions for each position
            probs = F.softmax(logits, dim=-1)
            top_k = 5
            top_probs, top_indices = torch.topk(probs, k=top_k, dim=-1)
            
            results['layers'][layer_idx] = {
                'logits': logits.cpu(),
                'probs': probs.cpu(),
                'top_k_indices': top_indices.cpu(),
                'top_k_probs': top_probs.cpu(),
                'top_k_tokens': [
                    [self.tokenizer.decode([idx]) for idx in position_indices]
                    for position_indices in top_indices.cpu().numpy()
                ]
            }
        
        return results
    
    @torch.no_grad()
    def get_convergence_metrics(self, text: str) -> Dict[str, np.ndarray]:
        """
        Calculate how predictions converge to final output across layers.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with convergence metrics
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        # Get final predictions
        final_logits = outputs.logits[0]
        final_probs = F.softmax(final_logits, dim=-1)
        final_top_indices = torch.argmax(final_probs, dim=-1)
        
        num_layers = len(hidden_states)
        seq_len = final_logits.shape[0]
        
        # Initialize metrics
        kl_divergences = np.zeros((num_layers, seq_len))
        top1_matches = np.zeros((num_layers, seq_len), dtype=bool)
        top1_ranks = np.zeros((num_layers, seq_len), dtype=int)
        top1_logits = np.zeros((num_layers, seq_len))
        
        for layer_idx in range(num_layers):
            hidden_state = hidden_states[layer_idx]
            logits = self.apply_lens_to_hidden_state(hidden_state)[0]
            probs = F.softmax(logits, dim=-1)
            
            # KL divergence from final distribution
            kl_div = F.kl_div(
                F.log_softmax(logits, dim=-1),
                final_probs,
                reduction='none'
            ).sum(dim=-1)
            kl_divergences[layer_idx] = kl_div.cpu().numpy()
            
            # Top-1 predictions
            top_indices = torch.argmax(probs, dim=-1)
            top1_matches[layer_idx] = (top_indices == final_top_indices).cpu().numpy()
            
            # Rank of final prediction in current layer
            sorted_indices = torch.argsort(probs, dim=-1, descending=True)
            ranks = (sorted_indices == final_top_indices.unsqueeze(-1)).nonzero(as_tuple=True)[1]
            top1_ranks[layer_idx] = ranks.cpu().numpy()
            
            # Logit of top prediction
            top1_logits[layer_idx] = logits.gather(1, top_indices.unsqueeze(-1)).squeeze(-1).cpu().numpy()
        
        return {
            'kl_divergence': kl_divergences,
            'top1_match': top1_matches,
            'top1_rank': top1_ranks,
            'top1_logit': top1_logits,
            'input_tokens': self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        }
    
    @torch.no_grad()
    def compare_to_input(self, text: str) -> Dict[str, np.ndarray]:
        """
        Analyze how much each layer preserves input token information.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary with input preservation metrics
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs['input_ids'][0]
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        num_layers = len(hidden_states)
        seq_len = len(input_ids)
        
        # Get embedding of input tokens (layer 0)
        input_embeddings = hidden_states[0][0]
        input_logits = self.apply_lens_to_hidden_state(input_embeddings.unsqueeze(0))[0]
        input_probs = F.softmax(input_logits, dim=-1)
        
        # Initialize metrics
        input_token_ranks = np.zeros((num_layers, seq_len), dtype=int)
        input_token_probs = np.zeros((num_layers, seq_len))
        kl_from_input = np.zeros((num_layers, seq_len))
        
        for layer_idx in range(num_layers):
            hidden_state = hidden_states[layer_idx]
            logits = self.apply_lens_to_hidden_state(hidden_state)[0]
            probs = F.softmax(logits, dim=-1)
            
            # KL divergence from input distribution
            kl_div = F.kl_div(
                F.log_softmax(input_logits, dim=-1),
                probs,
                reduction='none'
            ).sum(dim=-1)
            kl_from_input[layer_idx] = kl_div.cpu().numpy()
            
            # Rank and probability of input token
            for pos in range(seq_len):
                token_id = input_ids[pos].item()
                
                # Probability of input token
                input_token_probs[layer_idx, pos] = probs[pos, token_id].item()
                
                # Rank of input token
                sorted_probs = torch.sort(probs[pos], descending=True)
                rank = (sorted_probs.indices == token_id).nonzero(as_tuple=True)[0].item()
                input_token_ranks[layer_idx, pos] = rank
        
        return {
            'input_token_rank': input_token_ranks,
            'input_token_prob': input_token_probs,
            'kl_from_input': kl_from_input,
            'input_tokens': self.tokenizer.convert_ids_to_tokens(input_ids)
        }
    
    def print_layer_predictions(self, text: str, 
                               layer_step: int = 2,
                               max_tokens: int = 10):
        """
        Print a readable summary of predictions at each layer.
        
        Args:
            text: Input text
            layer_step: Show every Nth layer (1 = all layers)
            max_tokens: Maximum number of tokens to display
        """
        results = self.get_layer_predictions(text)
        input_tokens = results['input_tokens'][:max_tokens]
        
        # Get final predictions
        final_logits = results['final_logits']
        final_probs = F.softmax(final_logits, dim=-1)
        final_preds = torch.argmax(final_probs, dim=-1)
        final_tokens = [self.tokenizer.decode([idx]) for idx in final_preds[:max_tokens]]
        
        print("\n" + "="*80)
        print("LOGIT LENS ANALYSIS")
        print("="*80)
        print(f"\nInput text: {text[:100]}...")
        print(f"\nInput tokens: {input_tokens}")
        print(f"Final predictions: {final_tokens}")
        print("\n" + "-"*80)
        
        layers_to_show = sorted([k for k in results['layers'].keys() if k % layer_step == 0])
        
        for layer_idx in layers_to_show:
            layer_data = results['layers'][layer_idx]
            print(f"\nLayer {layer_idx}:")
            
            for pos in range(min(max_tokens, len(input_tokens))):
                top_token = layer_data['top_k_tokens'][pos][0]
                top_prob = layer_data['top_k_probs'][pos][0].item()
                matches_final = (top_token.strip() == final_tokens[pos].strip())
                
                match_marker = " ✓" if matches_final else ""
                print(f"  Pos {pos} ({input_tokens[pos]:>10s}): {top_token:>10s} "
                      f"(p={top_prob:.3f}){match_marker}")
        
        print("\n" + "="*80 + "\n")
    
    def plot_convergence(self, text: str, 
                        figsize: Tuple[int, int] = (12, 8),
                        save_path: Optional[str] = None):
        """
        Plot how predictions converge across layers.
        
        Args:
            text: Input text
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        metrics = self.get_convergence_metrics(text)
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # KL Divergence
        ax = axes[0, 0]
        im = ax.imshow(metrics['kl_divergence'], aspect='auto', cmap='viridis')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title('KL Divergence from Final Output')
        plt.colorbar(im, ax=ax)
        
        # Top-1 Match
        ax = axes[0, 1]
        im = ax.imshow(metrics['top1_match'], aspect='auto', cmap='RdYlGn')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title('Top-1 Matches Final Prediction')
        plt.colorbar(im, ax=ax)
        
        # Rank of Final Prediction
        ax = axes[1, 0]
        im = ax.imshow(np.log1p(metrics['top1_rank']), aspect='auto', cmap='viridis_r')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title('Rank of Final Prediction (log scale)')
        plt.colorbar(im, ax=ax, label='log(rank + 1)')
        
        # Top-1 Logit Value
        ax = axes[1, 1]
        im = ax.imshow(metrics['top1_logit'], aspect='auto', cmap='plasma')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title('Logit of Top Prediction')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
    
    def plot_input_preservation(self, text: str,
                               figsize: Tuple[int, int] = (12, 6),
                               save_path: Optional[str] = None):
        """
        Plot how well input tokens are preserved across layers.
        
        Args:
            text: Input text
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        metrics = self.compare_to_input(text)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Rank of input token
        ax = axes[0]
        im = ax.imshow(np.log1p(metrics['input_token_rank']), 
                      aspect='auto', cmap='viridis_r')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title('Rank of Input Token (log scale)')
        plt.colorbar(im, ax=ax, label='log(rank + 1)')
        
        # KL divergence from input
        ax = axes[1]
        im = ax.imshow(metrics['kl_from_input'], aspect='auto', cmap='plasma')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title('KL Divergence from Input Distribution')
        plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()


def analyze_text(model_path: str, text: str, device: str = "cpu"):
    """
    Convenience function to perform complete logit lens analysis.
    
    Args:
        model_path: Path to model directory
        text: Text to analyze
        device: Device to use
    """
    lens = LogitLens(model_path, device)
    
    print("Analyzing with logit lens...")
    lens.print_layer_predictions(text, layer_step=3, max_tokens=8)
    
    print("\nGenerating convergence plots...")
    lens.plot_convergence(text)
    
    print("\nGenerating input preservation plots...")
    lens.plot_input_preservation(text)
    
    return lens


if __name__ == "__main__":
    # Example usage
    model_path = "../models/gpt2_model"
    
    # Create logit lens analyzer
    lens = LogitLens(model_path, device="cpu")
    
    # Example text from GPT-3 paper abstract
    text = "We train GPT-3, an autoregressive language model with 175 billion parameters"
    
    # Print layer-by-layer predictions
    lens.print_layer_predictions(text, layer_step=2, max_tokens=10)
    
    # Generate visualizations
    print("\nGenerating convergence visualization...")
    lens.plot_convergence(text)
    
    print("\nGenerating input preservation visualization...")
    lens.plot_input_preservation(text)
