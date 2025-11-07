# Visualization Module Quick Reference

## Logit Lens Visualizations

### Basic Analysis
```python
from visualization import plot_logit_lens_basic
from logit_lens import LogitLens

lens = LogitLens("../models/gpt2_model")
plot_logit_lens_basic(lens, "Your text here", layer_step=3, max_tokens=8)
```

### Convergence Analysis
```python
from visualization import plot_convergence_analysis

plot_convergence_analysis(lens, "Your text here", save_path="convergence.png")
```

### Token Preservation
```python
from visualization import plot_token_preservation

plot_token_preservation(lens, "Your text here", save_path="preservation.png")
```

### Prediction Refinement
```python
from visualization import plot_prediction_refinement

plot_prediction_refinement(lens, "Your text here", layers=[0, 3, 6, 9, 11])
```

---

## Embedding Visualizations

### 2D Visualization
```python
from visualization import plot_embeddings_2d
import numpy as np

embeddings = np.random.randn(50, 768)  # Your embeddings
labels = [f"token_{i}" for i in range(50)]

# PCA
plot_embeddings_2d(embeddings, labels=labels, method='pca')

# t-SNE
plot_embeddings_2d(embeddings, labels=labels, method='tsne', perplexity=30)

# UMAP
plot_embeddings_2d(embeddings, labels=labels, method='umap', n_neighbors=15)
```

### 3D Visualization
```python
from visualization import plot_embeddings_3d

plot_embeddings_3d(embeddings, labels=labels, method='pca')
```

### Compare Methods Side-by-Side
```python
from visualization import compare_reduction_methods

compare_reduction_methods(embeddings, labels=labels, 
                         methods=['pca', 'tsne', 'umap'],
                         save_path="comparison.png")
```

### Layer Evolution
```python
from visualization import plot_layer_embeddings
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("../models/gpt2_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("../models/gpt2_model", local_files_only=True)

plot_layer_embeddings(model, tokenizer, "Your text here",
                     layer_indices=[0, 3, 6, 9, 11],
                     method='pca')
```

### Direct Dimensionality Reduction
```python
from visualization import reduce_dimensions

# PCA (fast, linear)
reduced = reduce_dimensions(embeddings, method='pca', n_components=2)

# t-SNE (slow, non-linear, preserves local structure)
reduced = reduce_dimensions(embeddings, method='tsne', n_components=2,
                           perplexity=50, max_iter=1000)

# UMAP (fast, non-linear, preserves global structure)
reduced = reduce_dimensions(embeddings, method='umap', n_components=2,
                           n_neighbors=15, min_dist=0.1)
```

---

## Method Selection Guide

### When to use PCA:
- ✓ Need fast results
- ✓ Want interpretable components
- ✓ Linear relationships are sufficient
- ✓ Need to understand variance explained

### When to use t-SNE:
- ✓ Exploring local neighborhoods
- ✓ Identifying clusters
- ✓ Don't care about global structure
- ✓ Have time for slower computation

### When to use UMAP:
- ✓ Need both local and global structure
- ✓ Want faster computation than t-SNE
- ✓ Working with large datasets
- ✓ Need consistent results across runs

---

## Parameters Guide

### PCA Parameters:
- `n_components`: Number of dimensions (2 or 3 for visualization)

### t-SNE Parameters:
- `perplexity`: Balance local vs global (5-50, default 30)
- `max_iter`: Number of iterations (1000-5000, default 1000)
- `learning_rate`: Step size (10-1000, default 200)

### UMAP Parameters:
- `n_neighbors`: Local neighborhood size (2-100, default 15)
- `min_dist`: Minimum distance between points (0.0-0.99, default 0.1)
- `metric`: Distance metric (default 'euclidean')

---

## Tips & Best Practices

1. **Always start with PCA** - It's fast and gives you a baseline
2. **Try multiple perplexity values** for t-SNE (e.g., 5, 15, 30, 50)
3. **Use compare_reduction_methods()** to see all methods at once
4. **Normalize embeddings** before visualization if needed
5. **Use colors** to highlight categories or clusters
6. **Save high-resolution plots** with `save_path` parameter and `dpi=300`
7. **For many points** (>1000), consider using UMAP instead of t-SNE

---

## Common Patterns

### Visualize Word Analogy
```python
tokens = ["king", "queen", "man", "woman", "prince", "princess"]
token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in tokens]
embeddings = model.get_input_embeddings().weight[token_ids].detach().numpy()

compare_reduction_methods(embeddings, labels=tokens, methods=['pca', 'umap'])
```

### Visualize Attention Head Outputs
```python
# Get attention outputs from specific layer/head
attention_outputs = ...  # Shape: (seq_len, hidden_dim)

plot_embeddings_2d(attention_outputs, 
                  labels=[f"pos_{i}" for i in range(len(attention_outputs))],
                  method='umap',
                  title='Attention Head 0 Outputs')
```

### Compare Layers
```python
# Get embeddings from multiple layers
layer_0_emb = ...
layer_6_emb = ...
layer_11_emb = ...

combined = np.vstack([layer_0_emb, layer_6_emb, layer_11_emb])
colors = np.array([0]*len(layer_0_emb) + [1]*len(layer_6_emb) + [2]*len(layer_11_emb))

plot_embeddings_2d(combined, colors=colors, method='umap',
                  title='Layer Comparison')
```
