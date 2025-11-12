"""
Example usage of the unified ModelAnalyzer class.

This demonstrates how to use a single model instance for multiple analysis tasks.
"""

import torch
from model_analyzer import ModelAnalyzer


def main():
    print("\n" + "="*80)
    print("UNIFIED MODEL ANALYZER DEMONSTRATION")
    print("="*80 + "\n")
    
    # Initialize analyzer (loads model once)
    print("Initializing ModelAnalyzer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    analyzer = ModelAnalyzer("../models/gpt2_model", device=device)
    
    # ===================================================================
    # 1. MODEL INSPECTION
    # ===================================================================
    print("\n" + "="*80)
    print("1. MODEL INSPECTION")
    print("="*80)
    
    analyzer.print_architecture_summary()
    
    # ===================================================================
    # 2. TEXT GENERATION
    # ===================================================================
    print("\n" + "="*80)
    print("2. TEXT GENERATION")
    print("="*80)
    
    prompt = "The capital of France is"
    print(f"\nPrompt: '{prompt}'")
    
    generated = analyzer.generate(prompt, max_new_tokens=10, temperature=0.7)
    print(f"Generated: {generated}")
    
    # ===================================================================
    # 3. GENERATION WITH ACTIVATION RECORDING
    # ===================================================================
    print("\n" + "="*80)
    print("3. GENERATION WITH ACTIVATION RECORDING")
    print("="*80)
    
    prompt = "Once upon a time"
    print(f"\nPrompt: '{prompt}'")
    
    # Use extract_activations for getting activations - cleaner and more explicit
    activations = analyzer.extract_activations(prompt)
    
    # Generate text separately
    generated = analyzer.generate(
        prompt,
        max_new_tokens=15
    )
    
    print(f"Generated: {generated}")
    print(f"Recorded activations from {len(activations.layer_activations)} layers")
    print(f"Tokens: {activations.tokens}")
    
    # ===================================================================
    # 4. ACTIVATION EXTRACTION
    # ===================================================================
    print("\n" + "="*80)
    print("4. ACTIVATION EXTRACTION")
    print("="*80)
    
    prompt = "The quick brown fox"
    print(f"\nPrompt: '{prompt}'")
    
    record = analyzer.extract_activations(prompt, include_attention=False)
    
    print(f"Extracted activations from {len(record.layer_activations)} layers")
    print(f"First layer shape: {list(record.layer_activations.values())[0].shape}")
    
    # Save activations
    analyzer.save_activations(record, "test_output.pkl", format='pickle')
    print("✓ Saved to test_output.pkl")
    
    # ===================================================================
    # 5. LOGIT LENS ANALYSIS
    # ===================================================================
    print("\n" + "="*80)
    print("5. LOGIT LENS ANALYSIS")
    print("="*80)
    
    prompt = "The Eiffel Tower is in"
    analyzer.print_logit_lens(prompt, layer_step=3, top_k=5)
    
    # ===================================================================
    # 6. BATCH ACTIVATION EXTRACTION
    # ===================================================================
    print("\n" + "="*80)
    print("6. BATCH ACTIVATION EXTRACTION")
    print("="*80)
    
    prompts = [
        "Paris is the capital of",
        "London is the capital of",
        "Berlin is the capital of"
    ]
    
    print(f"\nExtracting activations for {len(prompts)} prompts...")
    records = analyzer.extract_batch_activations(prompts, show_progress=True)
    
    print(f"✓ Extracted {len(records)} activation records")
    
    # ===================================================================
    # 7. STEERING VECTOR GENERATION
    # ===================================================================
    print("\n" + "="*80)
    print("7. STEERING VECTOR GENERATION")
    print("="*80)
    
    prompt = "Today I feel"
    print(f"\nPrompt: '{prompt}'")
    print("Steering toward positive sentiment...")
    
    steered = analyzer.generate_with_steering(
        prompt=prompt,
        positive_examples=["I am happy", "This is wonderful"],
        negative_examples=["I am sad", "This is terrible"],
        coefficient=2.0,
        max_new_tokens=20
    )
    
    print(f"Steered generation: {steered}")
    
    # ===================================================================
    # 8. PATH PATCHING
    # ===================================================================
    print("\n" + "="*80)
    print("8. PATH PATCHING (Causal Analysis)")
    print("="*80)
    
    print("\nTesting which layers are causally important...")
    print("Clean: 'The Eiffel Tower is in'")
    print("Corrupted: 'The Colosseum is in'")
    
    effects = analyzer.path_patching(
        clean_prompt="The Eiffel Tower is in",
        corrupted_prompt="The Colosseum is in"
    )
    
    print("\nCausal effects by layer:")
    for layer_idx, effect in sorted(effects.items())[:5]:
        print(f"  Layer {layer_idx}: {effect:.4f}")
    print(f"  ... and {len(effects) - 5} more layers")
    
    # ===================================================================
    # 9. UTILITY METHODS
    # ===================================================================
    print("\n" + "="*80)
    print("9. UTILITY METHODS")
    print("="*80)
    
    text = "Hello world!"
    tokens = analyzer.get_tokens(text)
    print(f"\nText: '{text}'")
    print(f"Tokens: {tokens}")
    
    print("\n" + "="*80)
    print("✓ ALL DEMONSTRATIONS COMPLETED!")
    print("="*80 + "\n")
    
    print("Summary:")
    print("  - Loaded model once")
    print("  - Performed 9 different analysis tasks")
    print("  - Generated text with and without recording")
    print("  - Applied logit lens")
    print("  - Used steering vectors")
    print("  - Analyzed causal structure")
    print("\nAll using the same ModelAnalyzer instance! 🎉")


if __name__ == "__main__":
    main()

