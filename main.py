"""
Main script for Mechanistic Interpretability Toolkit

Run this script to see demonstrations of:
1. Model inspection and architecture analysis
2. Logit lens analysis with multiple examples
3. Embedding visualization with dimensionality reduction
4. Activation patching and steering vectors

Usage:
    python main.py

Make sure you have a model at ../models/gpt2_model
"""

import torch
from model_loader import ModelInspector, load_model_from_path
from logit_lens import LogitLens
from visualization import (
    plot_logit_lens_basic,
    plot_convergence_analysis,
    plot_token_preservation,
    plot_prediction_refinement,
    plot_embeddings_2d,
    plot_embeddings_3d,
    compare_reduction_methods,
    plot_layer_embeddings
)
from intervention import (
    InterventionHandler,
    ActivationPatch,
    SteeringVector,
    create_steering_vector,
    activation_patch_experiment,
    path_patching,
    steer_generation,
    CausalTracer
)


def inspect_model(model_path: str):
    """
    Perform comprehensive model inspection.
    
    Args:
        model_path: Path to the model directory
    """
    print("="*80)
    print("MODEL INSPECTION")
    print("="*80)
    
    inspector = load_model_from_path(model_path, device="cpu")
    model, tokenizer = inspector.load_model()
    
    # Print architecture summary
    inspector.print_architecture_summary()
    
    # Layer information
    print("\n--- Layer Information ---")
    num_layers = inspector.get_num_layers()
    print(f"Total number of transformer layers: {num_layers}")
    
    # Show first few layer names
    print("\n--- First 20 Layer Names ---")
    layer_names = inspector.get_layer_names()
    for i, name in enumerate(layer_names[:20]):
        print(f"{i+1}. {name}")
    if len(layer_names) > 20:
        print(f"... and {len(layer_names) - 20} more layers")
    
    # Attention layers
    print("\n--- Attention Layers ---")
    attn_layers = inspector.get_attention_layer_names()
    for layer in attn_layers[:5]:
        print(f"  - {layer}")
    if len(attn_layers) > 5:
        print(f"  ... and {len(attn_layers) - 5} more attention layers")
    
    # MLP layers
    print("\n--- MLP Layers ---")
    mlp_layers = inspector.get_mlp_layer_names()
    for layer in mlp_layers[:5]:
        print(f"  - {layer}")
    if len(mlp_layers) > 5:
        print(f"  ... and {len(mlp_layers) - 5} more MLP layers")
    
    # Parameter counts
    print("\n--- Parameter Information ---")
    params = inspector.get_parameter_count()
    print(f"Total parameters: {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Non-trainable parameters: {params['non_trainable']:,}")
    
    # Config keys
    print("\n--- Configuration Keys ---")
    config = inspector.get_model_config()
    print(f"Available config keys: {list(config.keys())[:10]}...")
    
    # Access specific layer
    print("\n--- Accessing Specific Layer ---")
    try:
        layer = inspector.get_layer_by_name("h.0")
        print(f"Successfully accessed layer 'h.0': {type(layer).__name__}")
        
        weights = inspector.get_weights_by_layer("h.0")
        print(f"Number of weight tensors in this layer: {len(weights)}")
        for name, tensor in list(weights.items())[:3]:
            print(f"  - {name}: shape {tensor.shape}")
    except ValueError as e:
        print(f"Could not access layer: {e}")
    
    print("\n" + "="*80 + "\n")


def run_logit_lens_examples(model_path: str):
    """
    Run all logit lens examples.
    
    Args:
        model_path: Path to the model directory
    """
    print("="*80)
    print("LOGIT LENS EXAMPLES")
    print("="*80 + "\n")
    
    lens = LogitLens(model_path, device="cpu")
    
    # Example 1: Basic Analysis
    print("="*80)
    print("EXAMPLE 1: Basic Logit Lens Analysis")
    print("="*80)
    text = "The quick brown fox jumps over the lazy"
    plot_logit_lens_basic(lens, text)
    
    # Example 2: Convergence Analysis
    print("\n" + "="*80)
    print("EXAMPLE 2: Convergence Analysis")
    print("="*80)
    text = "In the year 2023, artificial intelligence made significant progress"
    plot_convergence_analysis(lens, text)
    
    # Example 3: Token Preservation
    print("\n" + "="*80)
    print("EXAMPLE 3: Token Preservation")
    print("="*80)
    text = "Sometimes when people say quantum, they mean physics. Other times when people say quantum"
    plot_token_preservation(lens, text)
    
    # Example 4: Prediction Refinement
    print("\n" + "="*80)
    print("EXAMPLE 4: Prediction Refinement")
    print("="*80)
    text = "The capital of France is"
    plot_prediction_refinement(lens, text, layers=[0, 3, 6, 9, 11])
    
    print("\n" + "="*80)
    print("ALL LOGIT LENS EXAMPLES COMPLETED!")
    print("="*80 + "\n")


def run_intervention_examples(model_path: str):
    """
    Run intervention examples (activation patching, steering).
    
    Args:
        model_path: Path to the model directory
    """
    print("="*80)
    print("INTERVENTION EXAMPLES")
    print("="*80 + "\n")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # Load model for generation
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    
    # Example 1: Basic Activation Patching
    print("="*80)
    print("EXAMPLE 1: Basic Activation Patching")
    print("="*80)
    
    print("\nDemonstrating activation patching with simple prompts...")
    clean_prompt = "The Eiffel Tower is in"
    corrupted_prompt = "The Colosseum is in"
    
    print(f"Clean prompt: '{clean_prompt}'")
    print(f"Corrupted prompt: '{corrupted_prompt}'")
    print("\nPatching activations from corrupted into clean...")
    
    result = activation_patch_experiment(
        model, tokenizer,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        layer_name="6",
        position=-1
    )
    
    print(f"✓ Patched layer 6 at position -1")
    print(f"  Shape of logits: {result['patched_logits'].shape}")
    
    # Example 2: Path Patching
    print("\n" + "="*80)
    print("EXAMPLE 2: Path Patching Across Layers")
    print("="*80)
    
    print("\nTesting causal importance of each layer...")
    print("(This may take a moment...)")
    
    layer_effects = path_patching(
        model, tokenizer,
        clean_prompt=clean_prompt,
        corrupted_prompt=corrupted_prompt,
        layers_to_test=[0, 3, 6, 9, 11]
    )
    
    print("\nCausal effect by layer:")
    for layer_idx, effect in sorted(layer_effects.items()):
        bar = "█" * int(effect * 20)
        print(f"  Layer {layer_idx:2d}: {effect:.4f} {bar}")
    
    # Example 3: Creating a Steering Vector
    print("\n" + "="*80)
    print("EXAMPLE 3: Creating a Steering Vector")
    print("="*80)
    
    print("\nCreating steering vector from contrastive prompts...")
    
    positive_prompts = [
        "I am happy and excited",
        "This is wonderful and great",
        "I feel joyful and pleased"
    ]
    negative_prompts = [
        "I am sad and upset",
        "This is terrible and awful",
        "I feel miserable and unhappy"
    ]
    
    print("Positive prompts (3):", positive_prompts[0], "...")
    print("Negative prompts (3):", negative_prompts[0], "...")
    
    steering_vec = create_steering_vector(
        model, tokenizer,
        positive_prompts=positive_prompts,
        negative_prompts=negative_prompts,
        layer_name="transformer.h.6",
        normalize=True
    )
    
    print(f"\n✓ Created steering vector with shape: {steering_vec.shape}")
    print(f"  Norm: {steering_vec.norm().item():.4f}")
    
    # Example 4: Steering Generation
    print("\n" + "="*80)
    print("EXAMPLE 4: Steering Text Generation")
    print("="*80)
    
    prompt = "Today I feel"
    print(f"\nPrompt: '{prompt}'")
    print("\nGenerating with steering (coefficient=2.0)...")
    
    try:
        steered_text = steer_generation(
            model, tokenizer,
            prompt=prompt,
            steering_vector=steering_vec,
            layer_name="transformer.h.6",
            coefficient=2.0,
            max_length=30
        )
        print(f"Steered output: '{steered_text}'")
    except Exception as e:
        print(f"Note: Generation example requires additional setup: {e}")
    
    # Example 5: Using InterventionHandler Context Manager
    print("\n" + "="*80)
    print("EXAMPLE 5: Manual Intervention with Context Manager")
    print("="*80)
    
    print("\nUsing InterventionHandler to apply custom patches...")
    
    handler = InterventionHandler(model, tokenizer)
    
    # Create a simple patch that zeros out a position
    patch = ActivationPatch(
        layer_name="transformer.h.5",
        position=0,  # First token
        value=torch.zeros(768),  # GPT-2 hidden size
        mode='replace'
    )
    
    handler.register_activation_patch(patch)
    
    inputs = tokenizer("Hello world", return_tensors="pt")
    
    with handler:
        outputs = model(**inputs)
        print(f"✓ Applied patch and ran model")
        print(f"  Output logits shape: {outputs.logits.shape}")
    
    print("\n" + "="*80)
    print("ALL INTERVENTION EXAMPLES COMPLETED!")
    print("="*80 + "\n")
    """
    Run embedding visualization examples with dimensionality reduction.
    
    Args:
        model_path: Path to the model directory
    """
    print("="*80)
    print("EMBEDDING VISUALIZATION EXAMPLES")
    print("="*80 + "\n")
    
    import numpy as np
    from transformers import AutoModel, AutoTokenizer
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    
    # Example 1: Visualize token embeddings with different methods
    print("="*80)
    print("EXAMPLE 1: Token Embeddings with Different Reduction Methods")
    print("="*80)
    
    # Get some token embeddings
    tokens = ["king", "queen", "man", "woman", "prince", "princess", 
              "boy", "girl", "father", "mother"]
    token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in tokens]
    
    # Get embeddings from embedding layer
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight[token_ids].cpu().numpy()
    
    print(f"Comparing reduction methods for {len(tokens)} tokens...")
    compare_reduction_methods(embeddings, labels=tokens, 
                             methods=['pca', 'tsne', 'umap'])
    
    # Example 2: Visualize how embeddings change across layers
    print("\n" + "="*80)
    print("EXAMPLE 2: Token Evolution Across Layers")
    print("="*80)
    
    text = "The cat sat on the mat"
    print(f"Visualizing token evolution for: '{text}'")
    plot_layer_embeddings(model, tokenizer, text, 
                         layer_indices=[0, 3, 6, 9, 11],
                         method='pca')
    
    # Example 3: 3D visualization
    print("\n" + "="*80)
    print("EXAMPLE 3: 3D Embedding Visualization")
    print("="*80)
    
    print("Creating 3D visualization of token embeddings...")
    plot_embeddings_3d(embeddings, labels=tokens, method='pca')
    
    print("\n" + "="*80)
    print("ALL INTERVENTION EXAMPLES COMPLETED!")
    print("="*80 + "\n")


def run_embedding_visualization_examples(model_path: str):
    """
    Run embedding visualization examples with dimensionality reduction.
    
    Args:
        model_path: Path to the model directory
    """
    print("="*80)
    print("EMBEDDING VISUALIZATION EXAMPLES")
    print("="*80 + "\n")
    
    import numpy as np
    from transformers import AutoModel, AutoTokenizer
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    
    # Example 1: Visualize token embeddings with different methods
    print("="*80)
    print("EXAMPLE 1: Token Embeddings with Different Reduction Methods")
    print("="*80)
    
    # Get some token embeddings
    tokens = ["king", "queen", "man", "woman", "prince", "princess", 
              "boy", "girl", "father", "mother"]
    token_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in tokens]
    
    # Get embeddings from embedding layer
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight[token_ids].cpu().numpy()
    
    print(f"Comparing reduction methods for {len(tokens)} tokens...")
    compare_reduction_methods(embeddings, labels=tokens, 
                             methods=['pca', 'tsne', 'umap'])
    
    # Example 2: Visualize how embeddings change across layers
    print("\n" + "="*80)
    print("EXAMPLE 2: Token Evolution Across Layers")
    print("="*80)
    
    text = "The cat sat on the mat"
    print(f"Visualizing token evolution for: '{text}'")
    plot_layer_embeddings(model, tokenizer, text, 
                         layer_indices=[0, 3, 6, 9, 11],
                         method='pca')
    
    # Example 3: 3D visualization
    print("\n" + "="*80)
    print("EXAMPLE 3: 3D Embedding Visualization")
    print("="*80)
    
    print("Creating 3D visualization of token embeddings...")
    plot_embeddings_3d(embeddings, labels=tokens, method='pca')
    
    print("\n" + "="*80)
    print("ALL EMBEDDING VISUALIZATION EXAMPLES COMPLETED!")
    print("="*80 + "\n")


def main():
    """
    Main entry point - run all demonstrations.
    """
    model_path = "../models/gpt2_model"
    
    print("\n" + "="*80)
    print("MECHANISTIC INTERPRETABILITY TOOLKIT")
    print("="*80 + "\n")
    
    try:
        # Part 1: Model Inspection
        inspect_model(model_path)
        
        # Part 2: Logit Lens Examples
        run_logit_lens_examples(model_path)
        
        # Part 3: Embedding Visualization Examples
        run_embedding_visualization_examples(model_path)
        
        # Part 4: Intervention Examples
        run_intervention_examples(model_path)
        
        print("\n" + "="*80)
        print("ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY!")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have:")
        print("  1. Installed all requirements: pip install -r requirements.txt")
        print("  2. Downloaded a model to ../models/gpt2_model")
        print("\nTo download GPT-2:")
        print("  from transformers import AutoModel, AutoTokenizer")
        print("  model = AutoModel.from_pretrained('gpt2')")
        print("  model.save_pretrained('../models/gpt2_model')")
        print("  tokenizer = AutoTokenizer.from_pretrained('gpt2')")
        print("  tokenizer.save_pretrained('../models/gpt2_model')")


if __name__ == "__main__":
    main()
