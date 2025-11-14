import torch
from model_analyzer import ModelAnalyzer
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

def main():
    """
    Main entry point - run all demonstrations.
    """
    print("Initializing ModelAnalyzer...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = "../models/Llama-3.2-1B"
    analyzer = ModelAnalyzer(model_path, device=device)
    analyzer.load_model()

    analyzer.print_architecture_summary()

    test_prompts = []
    test_prompts += [f"{i}+{j}=" for i in range(1, 6) for j in range(1, 6)]
    test_prompts += [f"{i}00+{j}00=" for i in range(1, 6) for j in range(1, 6)]
    test_prompts += [f"336+639=", "958+501=", "698+316="]

    records = []
    for prompt in test_prompts:
        print(f"Generating and recording activations for prompt: '{prompt}'")
        generated_text = analyzer.generate(
            prompt=prompt,
            max_new_tokens=20
        )

        record = analyzer.extract_activations(text=generated_text, 
            layer_names=None,
            layer_indices=None,
            include_attention=False,
            return_logits=False,
            metadata=None
        )

        records.append(record)

    # Save records for later use
    analyzer.save_activations(record=records, output_path="activations/activation_records.pt", format="pt")




    
    


if __name__ == "__main__":
    main()