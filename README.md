# Mechanistic Interpretability Toolkit

A toolkit for mechanistic interpretability research on transformer language models. Features include model inspection, **Logit Lens** analysis, **embedding visualization** with multiple dimensionality reduction techniques, and **causal interventions** (activation patching and steering vectors).

## Features

### 1. Model Loader (`model_loader.py`)
Load and inspect transformer models to extract architectural information.

**Key capabilities:**
- Load models from local paths
- Extract layer information (count, names, types)
- Get model configuration and architecture summary
- Count parameters
- Access specific layers and weights

**Example:**
```python
from model_loader import ModelInspector

inspector = ModelInspector("../models/gpt2_model")
model, tokenizer = inspector.load_model()

# Print architecture summary
inspector.print_architecture_summary()

# Get layer information
num_layers = inspector.get_num_layers()
attention_layers = inspector.get_attention_layer_names()
```

### 2. Logit Lens (`logit_lens.py`)
Analyze what the model "believes" at each layer by projecting intermediate activations to vocabulary space.

**Based on:** [Interpreting GPT: The Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) by nostalgebraist

**Key capabilities:**
- Apply logit lens to any transformer layer
- Track prediction convergence across layers
- Analyze input token preservation
- Visualize layer-by-layer predictions
- Generate convergence and preservation plots

**Example:**
```python
from logit_lens import LogitLens

lens = LogitLens("../models/gpt2_model")
text = "The quick brown fox jumps"

# Print predictions at each layer
lens.print_layer_predictions(text, layer_step=2)

# Visualize convergence
lens.plot_convergence(text)

# Analyze input preservation
lens.plot_input_preservation(text)
```

### 3. Visualization Tools (`visualization.py`)
Comprehensive visualization functions for mechanistic interpretability analysis.

**Logit Lens Visualizations:**
- `plot_logit_lens_basic()` - Basic layer-by-layer predictions
- `plot_convergence_analysis()` - Prediction convergence across layers
- `plot_token_preservation()` - Input token preservation analysis
- `plot_prediction_refinement()` - How predictions refine over layers

**Embedding Visualizations:**
- `plot_embeddings_2d()` - 2D embedding visualization with PCA/t-SNE/UMAP
- `plot_embeddings_3d()` - 3D embedding visualization
- `compare_reduction_methods()` - Side-by-side comparison of reduction methods
- `plot_layer_embeddings()` - Token evolution across layers
- `reduce_dimensions()` - Flexible dimensionality reduction utility

**Supported reduction methods:**
- **PCA** (Principal Component Analysis) - Fast, linear, interpretable
- **t-SNE** (t-Distributed Stochastic Neighbor Embedding) - Non-linear, preserves local structure
- **UMAP** (Uniform Manifold Approximation and Projection) - Fast, preserves global structure

**Example:**
```python
from visualization import plot_embeddings_2d, compare_reduction_methods
import numpy as np

# Visualize embeddings with PCA
embeddings = model.get_input_embeddings().weight.detach().numpy()
labels = ["king", "queen", "man", "woman"]
plot_embeddings_2d(embeddings, labels=labels, method='pca')

# Compare all reduction methods
compare_reduction_methods(embeddings, labels=tokens, 
                         methods=['pca', 'tsne', 'umap'])
```

### 4. Intervention Tools (`intervention.py`)
Perform causal interventions to understand model behavior.

**Activation Patching:**
- `ActivationPatch` - Specification for patching activations
- `activation_patch_experiment()` - Single-layer patching experiment
- `path_patching()` - Test causal importance across layers
- `compute_feature_attribution()` - Identify important neurons

**Steering Vectors:**
- `SteeringVector` - Specification for steering
- `create_steering_vector()` - Create steering from contrastive prompts
- `steer_generation()` - Generate text with steering applied

**Advanced Tools:**
- `InterventionHandler` - Context manager for applying interventions
- `CausalTracer` - Trace causal pathways through the model
- `patch_and_run()` - Convenience function for quick patching

**Example:**
```python
from intervention import (
    InterventionHandler,
    ActivationPatch,
    create_steering_vector,
    steer_generation
)

# Create steering vector from contrastive examples
steering_vec = create_steering_vector(
    model, tokenizer,
    positive_prompts=["I am happy", "This is great"],
    negative_prompts=["I am sad", "This is terrible"],
    layer_name="transformer.h.6"
)

# Generate with steering
steered_text = steer_generation(
    model, tokenizer,
    prompt="Today I feel",
    steering_vector=steering_vec,
    layer_name="transformer.h.6",
    coefficient=2.0
)

# Or use activation patching
handler = InterventionHandler(model, tokenizer)
patch = ActivationPatch(
    layer_name="transformer.h.5",
    position=-1,
    value=some_activation,
    mode='replace'
)
handler.register_activation_patch(patch)

with handler:
    outputs = model(**inputs)
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download a model locally (e.g., GPT-2):
```python
from transformers import AutoModel, AutoTokenizer

model_name = "gpt2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save locally
model.save_pretrained("../models/gpt2_model")
tokenizer.save_pretrained("../models/gpt2_model")
```

3. Run the main demo:
```bash
python main.py
```

## Usage Examples

### Basic Model Inspection
```python
from model_loader import ModelInspector

inspector = ModelInspector("../models/gpt2_model")
inspector.print_architecture_summary()
```

### Logit Lens Analysis
```python
from logit_lens import LogitLens
from visualization import plot_convergence_analysis, plot_token_preservation

lens = LogitLens("../models/gpt2_model", device="cpu")

# Analyze predictions
text = "Machine learning is"
plot_convergence_analysis(lens, text)
plot_token_preservation(lens, text)
```

### Embedding Visualization
```python
from visualization import plot_embeddings_2d, compare_reduction_methods
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("../models/gpt2_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("../models/gpt2_model", local_files_only=True)

# Get token embeddings
tokens = ["king", "queen", "man", "woman"]
token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in tokens]
embeddings = model.get_input_embeddings().weight[token_ids].detach().numpy()

# Visualize with different methods
compare_reduction_methods(embeddings, labels=tokens, 
                         methods=['pca', 'tsne', 'umap'])

# 2D visualization with specific method
plot_embeddings_2d(embeddings, labels=tokens, method='umap',
                  title='Word Embeddings (UMAP)')
```

### Activation Patching
```python
from intervention import activation_patch_experiment, path_patching
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("../models/gpt2_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("../models/gpt2_model", local_files_only=True)

# Patch a single layer
result = activation_patch_experiment(
    model, tokenizer,
    clean_prompt="The Eiffel Tower is in",
    corrupted_prompt="The Colosseum is in",
    layer_name="6",
    position=-1
)

# Test all layers
layer_effects = path_patching(
    model, tokenizer,
    clean_prompt="The Eiffel Tower is in",
    corrupted_prompt="The Colosseum is in"
)
print(layer_effects)  # {layer_idx: causal_effect}
```

### Steering Vectors
```python
from intervention import create_steering_vector, steer_generation

# Create steering vector
steering_vec = create_steering_vector(
    model, tokenizer,
    positive_prompts=["I am happy", "This is wonderful"],
    negative_prompts=["I am sad", "This is terrible"],
    layer_name="transformer.h.6"
)

# Generate with steering
output = steer_generation(
    model, tokenizer,
    prompt="Today I feel",
    steering_vector=steering_vec,
    layer_name="transformer.h.6",
    coefficient=2.0  # Steering strength
)
print(output)
```

## What is the Logit Lens?

The logit lens is a mechanistic interpretability technique that reveals what a transformer model "thinks" at intermediate layers.

**Key insights:**
1. **Gradual convergence**: Models form rough guesses early and refine them across layers
2. **Input discarding**: Input representations are immediately transformed into output-space predictions
3. **Rare token preservation**: Unusual tokens are preserved differently than common ones
4. **Prediction refinement**: Each layer refines the distribution, often maintaining the same top tokens

**How it works:**
- Take hidden states from any layer
- Apply final layer normalization (optional but recommended)
- Project through the unembedding matrix (vocabulary projection)
- Interpret as logits over vocabulary

This reveals what tokens the model is "thinking about" at each layer, even though only the final layer is used for actual predictions.

## Understanding the Visualizations

### Convergence Plot
Shows 4 metrics across layers and token positions:
1. **KL Divergence**: How different from final output (lower = more similar)
2. **Top-1 Match**: Whether top prediction matches final prediction
3. **Rank**: Where the final prediction ranks in current layer (lower = better)
4. **Logit Value**: Raw logit score of top prediction (higher = more confident)

**Pattern to look for:** Values should generally improve (converge) as you move down the layers.

### Input Preservation Plot
Shows 2 metrics:
1. **Input Token Rank**: How highly ranked is the *input* token at each layer
2. **KL from Input**: How different from input distribution

**Pattern to look for:** Immediate jump after first layer shows rapid transformation from input space to prediction space. Some tokens (especially rare ones) may be preserved longer.

## File Structure

```
llm_preference/
├── model_loader.py       # Model loading and inspection
├── logit_lens.py         # Logit lens implementation
├── visualization.py      # Visualization tools (logit lens + embeddings)
├── intervention.py       # Activation patching & steering vectors ⭐
├── main.py              # Main demo script (run this!)
├── requirements.txt     # Python dependencies
├── README.md           # This file
└── VISUALIZATION_GUIDE.md  # Visualization quick reference
```

## Advanced Usage

### Custom Dimensionality Reduction
```python
from visualization import reduce_dimensions
import numpy as np

# Your high-dimensional embeddings
embeddings = np.random.randn(100, 768)  # 100 samples, 768 dims

# Reduce with PCA
reduced_pca = reduce_dimensions(embeddings, method='pca', n_components=2)

# Reduce with t-SNE (with custom parameters)
reduced_tsne = reduce_dimensions(embeddings, method='tsne', n_components=2,
                                perplexity=50, n_iter=2000)

# Reduce with UMAP (with custom parameters)
reduced_umap = reduce_dimensions(embeddings, method='umap', n_components=3,
                                n_neighbors=20, min_dist=0.05)
```

### Visualizing Layer Evolution
```python
from visualization import plot_layer_embeddings
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("../models/gpt2_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("../models/gpt2_model", local_files_only=True)

# See how token embeddings change across layers
text = "The cat sat on the mat"
plot_layer_embeddings(model, tokenizer, text, 
                     layer_indices=[0, 3, 6, 9, 11],
                     method='pca')
```

### Combining Logit Lens with Embedding Visualization
```python
from logit_lens import LogitLens
from visualization import plot_convergence_analysis, plot_embeddings_2d

lens = LogitLens("../models/gpt2_model")
text = "Artificial intelligence is transforming"

# Analyze prediction convergence
plot_convergence_analysis(lens, text)

# Get hidden states and visualize
results = lens.get_layer_predictions(text, layer_indices=[0, 6, 11])
# Extract and visualize specific layer embeddings
# ... (custom analysis)
```

## Common Research Questions

The toolkit can help answer:

1. **When does the model "figure out" the answer?** 
   - Use `plot_convergence_analysis()` to see which layer first predicts correctly

2. **How are rare tokens preserved?**
   - Use `plot_token_preservation()` on text with unusual words

3. **Do predictions refine gradually or jump discontinuously?**
   - Examine KL divergence progression with `get_convergence_metrics()`

4. **What are early layers "thinking"?**
   - Use `plot_prediction_refinement()` with early layer indices

5. **How do embeddings cluster semantically?**
   - Use `plot_embeddings_2d()` or `compare_reduction_methods()` on word embeddings

6. **How do token representations change across layers?**
   - Use `plot_layer_embeddings()` to visualize evolution

7. **Which dimensionality reduction method works best for my embeddings?**
   - Use `compare_reduction_methods()` to see PCA, t-SNE, and UMAP side-by-side

8. **Which layers are causally responsible for a behavior?**
   - Use `path_patching()` to test each layer's causal effect

9. **Can I steer the model toward desired outputs?**
   - Create steering vectors with `create_steering_vector()` from contrastive prompts

10. **Which neurons are most important for a prediction?**
    - Use `compute_feature_attribution()` to identify critical features

## References

- [Interpreting GPT: The Logit Lens](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens) - nostalgebraist (2020)
- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) - Neel Nanda's interpretability library
- [Exploratory Analysis Demo](https://colab.research.google.com/drive/1KZJqWw1GsFUVzxpWEenMySNRo92CQo_W) - Original Colab notebook

## Future Additions

Planned features:
- Attention head visualization and analysis
- Circuit discovery algorithms (e.g., ACDC)
- Automated interpretability with activation clustering
- Sparse autoencoder analysis
- Gradient-based attribution methods (integrated gradients, etc.)
- Interactive dashboards for real-time exploration
- Support for more model architectures (LLaMA, Mistral, etc.)

## License

MIT License - feel free to use and modify for your research!

## Contributing

This is a research toolkit. Contributions, bug reports, and suggestions are welcome!
