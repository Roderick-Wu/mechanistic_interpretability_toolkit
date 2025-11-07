"""
Visualization tools for mechanistic interpretability analysis.

This module provides visualization functions for:
- Logit lens analysis
- Embedding visualization with dimensionality reduction (PCA, t-SNE, UMAP)
- Activation analysis across layers
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Optional, Dict, Tuple, Union
import torch

# Dimensionality reduction imports
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ============================================================================
# LOGIT LENS VISUALIZATION FUNCTIONS
# ============================================================================

def plot_logit_lens_basic(lens, text: str, model_path: str = None, 
                          layer_step: int = 3, max_tokens: int = 8):
    """
    Basic logit lens visualization with printed output.
    
    Args:
        lens: LogitLens instance (or None to create one)
        text: Text to analyze
        model_path: Path to model (required if lens is None)
        layer_step: Show every Nth layer
        max_tokens: Maximum tokens to display
    """
    from logit_lens import LogitLens
    
    if lens is None:
        if model_path is None:
            raise ValueError("Must provide either lens or model_path")
        lens = LogitLens(model_path, device="cpu")
    
    print(f"Analyzing: '{text}'")
    lens.print_layer_predictions(text, layer_step=layer_step, max_tokens=max_tokens)


def plot_convergence_analysis(lens, text: str, model_path: str = None,
                              save_path: Optional[str] = None):
    """
    Plot and analyze prediction convergence across layers.
    
    Args:
        lens: LogitLens instance (or None to create one)
        text: Text to analyze
        model_path: Path to model (required if lens is None)
        save_path: Optional path to save figure
    """
    from logit_lens import LogitLens
    
    if lens is None:
        if model_path is None:
            raise ValueError("Must provide either lens or model_path")
        lens = LogitLens(model_path, device="cpu")
    
    print(f"Analyzing: '{text[:60]}...'")
    
    # Get convergence metrics
    metrics = lens.get_convergence_metrics(text)
    
    print(f"\nConvergence Statistics:")
    print(f"  - KL divergence shape: {metrics['kl_divergence'].shape}")
    print(f"  - Average KL at layer 0: {metrics['kl_divergence'][0].mean():.3f}")
    print(f"  - Average KL at final layer: {metrics['kl_divergence'][-1].mean():.3f}")
    
    # Show when predictions match final output
    num_layers = metrics['top1_match'].shape[0]
    for pos in range(min(5, metrics['top1_match'].shape[1])):
        first_match = None
        for layer in range(num_layers):
            if metrics['top1_match'][layer, pos]:
                first_match = layer
                break
        print(f"  - Token {pos} ({metrics['input_tokens'][pos]}): "
              f"converges at layer {first_match if first_match else 'never'}")
    
    # Generate plots
    print("\nGenerating convergence visualization...")
    lens.plot_convergence(text[:50], save_path=save_path)


def plot_token_preservation(lens, text: str, model_path: str = None,
                           save_path: Optional[str] = None):
    """
    Visualize how input tokens are preserved across layers.
    
    Args:
        lens: LogitLens instance (or None to create one)
        text: Text to analyze
        model_path: Path to model (required if lens is None)
        save_path: Optional path to save figure
    """
    from logit_lens import LogitLens
    
    if lens is None:
        if model_path is None:
            raise ValueError("Must provide either lens or model_path")
        lens = LogitLens(model_path, device="cpu")
    
    print(f"Analyzing: '{text}'")
    
    # Get input preservation metrics
    metrics = lens.compare_to_input(text)
    
    print(f"\nInput Token Preservation:")
    tokens = metrics['input_tokens']
    
    # Find interesting tokens
    for pos, token in enumerate(tokens[:10]):  # Show first 10
        ranks = metrics['input_token_rank'][:, pos]
        if ranks[0] < 1000:  # Only show relatively well-preserved tokens
            print(f"\nToken '{token}' at position {pos}:")
            print(f"  - Input layer rank: {ranks[0]}")
            if len(ranks) > 6:
                print(f"  - Mid-layer rank: {ranks[len(ranks)//2]}")
            print(f"  - Final layer rank: {ranks[-1]}")
    
    print("\nGenerating preservation visualization...")
    lens.plot_input_preservation(text, save_path=save_path)


def plot_prediction_refinement(lens, text: str, model_path: str = None,
                               layers: Optional[List[int]] = None):
    """
    Show how predictions refine over specific layers.
    
    Args:
        lens: LogitLens instance (or None to create one)
        text: Text to analyze
        model_path: Path to model (required if lens is None)
        layers: List of layer indices to examine
    """
    from logit_lens import LogitLens
    
    if lens is None:
        if model_path is None:
            raise ValueError("Must provide either lens or model_path")
        lens = LogitLens(model_path, device="cpu")
    
    if layers is None:
        layers = [0, 3, 6, 9, 11]
    
    print(f"Analyzing: '{text}'")
    
    # Get detailed predictions
    results = lens.get_layer_predictions(text, layer_indices=layers)
    
    print("\nTop-3 predictions across selected layers:")
    print(f"Input tokens: {results['input_tokens']}")
    
    # Focus on the last token position
    pos = -1
    
    for layer_idx in layers:
        layer_data = results['layers'][layer_idx]
        top3_tokens = layer_data['top_k_tokens'][pos][:3]
        top3_probs = layer_data['top_k_probs'][pos][:3]
        
        print(f"\nLayer {layer_idx}:")
        for i, (token, prob) in enumerate(zip(top3_tokens, top3_probs)):
            print(f"  {i+1}. '{token}' (p={prob:.4f})")


# ============================================================================
# EMBEDDING VISUALIZATION WITH DIMENSIONALITY REDUCTION
# ============================================================================

def reduce_dimensions(embeddings: np.ndarray, 
                     method: str = 'pca',
                     n_components: int = 2,
                     **kwargs) -> np.ndarray:
    """
    Reduce dimensionality of embeddings using various methods.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        method: 'pca', 'tsne', or 'umap'
        n_components: Number of dimensions to reduce to (typically 2 or 3)
        **kwargs: Additional arguments for the reduction method
        
    Returns:
        Reduced embeddings of shape (n_samples, n_components)
    """
    method = method.lower()
    
    if method == 'pca':
        reducer = PCA(n_components=n_components, **kwargs)
        reduced = reducer.fit_transform(embeddings)
        print(f"PCA explained variance ratio: {reducer.explained_variance_ratio_}")
        return reduced
        
    elif method == 'tsne':
        # Set sensible defaults for t-SNE
        tsne_kwargs = {'perplexity': 30, 'max_iter': 1000, 'random_state': 42}
        tsne_kwargs.update(kwargs)
        reducer = TSNE(n_components=n_components, **tsne_kwargs)
        reduced = reducer.fit_transform(embeddings)
        return reduced
        
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        # Set sensible defaults for UMAP
        umap_kwargs = {'n_neighbors': 15, 'min_dist': 0.1, 'random_state': 42}
        umap_kwargs.update(kwargs)
        reducer = umap.UMAP(n_components=n_components, **umap_kwargs)
        reduced = reducer.fit_transform(embeddings)
        return reduced
        
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'pca', 'tsne', or 'umap'")


def plot_embeddings_2d(embeddings: np.ndarray,
                       labels: Optional[List[str]] = None,
                       colors: Optional[np.ndarray] = None,
                       method: str = 'pca',
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (10, 8),
                       save_path: Optional[str] = None,
                       show_labels: bool = True,
                       **reduction_kwargs):
    """
    Visualize high-dimensional embeddings in 2D using dimensionality reduction.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Optional labels for each embedding
        colors: Optional colors/categories for each point
        method: Reduction method ('pca', 'tsne', 'umap')
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        show_labels: Whether to show text labels for points
        **reduction_kwargs: Additional arguments for reduction method
    """
    # Reduce dimensions
    reduced = reduce_dimensions(embeddings, method=method, n_components=2, 
                               **reduction_kwargs)
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot points
    if colors is not None:
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], 
                           c=colors, cmap='tab10', s=100, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Category')
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], s=100, alpha=0.7)
    
    # Add labels if provided
    if labels is not None and show_labels:
        for i, label in enumerate(labels):
            ax.annotate(label, (reduced[i, 0], reduced[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
    
    # Set title and labels
    if title is None:
        title = f"Embedding Visualization ({method.upper()})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_embeddings_3d(embeddings: np.ndarray,
                       labels: Optional[List[str]] = None,
                       colors: Optional[np.ndarray] = None,
                       method: str = 'pca',
                       title: Optional[str] = None,
                       figsize: Tuple[int, int] = (12, 10),
                       save_path: Optional[str] = None,
                       **reduction_kwargs):
    """
    Visualize high-dimensional embeddings in 3D using dimensionality reduction.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Optional labels for each embedding
        colors: Optional colors/categories for each point
        method: Reduction method ('pca', 'tsne', 'umap')
        title: Plot title
        figsize: Figure size
        save_path: Optional path to save figure
        **reduction_kwargs: Additional arguments for reduction method
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Reduce dimensions
    reduced = reduce_dimensions(embeddings, method=method, n_components=3,
                               **reduction_kwargs)
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    if colors is not None:
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                           c=colors, cmap='tab10', s=100, alpha=0.7)
        plt.colorbar(scatter, ax=ax, label='Category', shrink=0.5)
    else:
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2],
                  s=100, alpha=0.7)
    
    # Add labels if provided
    if labels is not None:
        for i, label in enumerate(labels):
            ax.text(reduced[i, 0], reduced[i, 1], reduced[i, 2], label,
                   fontsize=8, alpha=0.8)
    
    # Set title and labels
    if title is None:
        title = f"Embedding Visualization 3D ({method.upper()})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f'{method.upper()} Component 1', fontsize=12)
    ax.set_ylabel(f'{method.upper()} Component 2', fontsize=12)
    ax.set_zlabel(f'{method.upper()} Component 3', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def compare_reduction_methods(embeddings: np.ndarray,
                              labels: Optional[List[str]] = None,
                              colors: Optional[np.ndarray] = None,
                              methods: List[str] = ['pca', 'tsne', 'umap'],
                              figsize: Tuple[int, int] = (18, 6),
                              save_path: Optional[str] = None):
    """
    Compare multiple dimensionality reduction methods side by side.
    
    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Optional labels for each embedding
        colors: Optional colors/categories for each point
        methods: List of methods to compare
        figsize: Figure size
        save_path: Optional path to save figure
    """
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=figsize)
    
    if n_methods == 1:
        axes = [axes]
    
    for ax, method in zip(axes, methods):
        try:
            # Reduce dimensions
            reduced = reduce_dimensions(embeddings, method=method, n_components=2)
            
            # Plot points
            if colors is not None:
                scatter = ax.scatter(reduced[:, 0], reduced[:, 1],
                                   c=colors, cmap='tab10', s=100, alpha=0.7)
            else:
                ax.scatter(reduced[:, 0], reduced[:, 1], s=100, alpha=0.7)
            
            # Add labels if provided and not too many
            if labels is not None and len(labels) <= 20:
                for i, label in enumerate(labels):
                    ax.annotate(label, (reduced[i, 0], reduced[i, 1]),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, alpha=0.8)
            
            ax.set_title(f'{method.upper()}', fontsize=14, fontweight='bold')
            ax.set_xlabel('Component 1', fontsize=10)
            ax.set_ylabel('Component 2', fontsize=10)
            ax.grid(True, alpha=0.3)
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error with {method}:\n{str(e)}',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{method.upper()} (Failed)', fontsize=14)
    
    plt.suptitle('Dimensionality Reduction Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_layer_embeddings(model, tokenizer, text: str,
                         layer_indices: Optional[List[int]] = None,
                         method: str = 'pca',
                         figsize: Tuple[int, int] = (15, 10),
                         save_path: Optional[str] = None):
    """
    Visualize how token embeddings evolve across layers.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer
        text: Input text
        layer_indices: Which layers to visualize (None = all)
        method: Reduction method
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt")
    
    # Get hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    
    # Select layers
    if layer_indices is None:
        layer_indices = [0, len(hidden_states)//4, len(hidden_states)//2, 
                        3*len(hidden_states)//4, len(hidden_states)-1]
    
    n_layers = len(layer_indices)
    fig, axes = plt.subplots(1, n_layers, figsize=figsize)
    
    if n_layers == 1:
        axes = [axes]
    
    for ax, layer_idx in zip(axes, layer_indices):
        # Get embeddings for this layer
        embeddings = hidden_states[layer_idx][0].cpu().numpy()
        
        # Reduce dimensions
        reduced = reduce_dimensions(embeddings, method=method, n_components=2)
        
        # Plot
        ax.scatter(reduced[:, 0], reduced[:, 1], s=100, alpha=0.7)
        
        # Label tokens
        for i, token in enumerate(tokens):
            ax.annotate(token, (reduced[i, 0], reduced[i, 1]),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, alpha=0.8)
        
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.set_xlabel(f'{method.upper()} 1', fontsize=10)
        ax.set_ylabel(f'{method.upper()} 2', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Token Embeddings Across Layers ({method.upper()})',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
