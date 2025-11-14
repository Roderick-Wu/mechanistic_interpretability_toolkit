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

    analyzer.print_architecture_summary()

    generated_text = analyzer.generate(
        prompt="4+5=",
        max_new_tokens=10
    )

    record = analyzer.extract_activations(text=generated_text, 
        layer_names=None,
        layer_indices=None,
        include_attention=False,
        return_logits=False,
        metadata=None
    )




    
    


if __name__ == "__main__":
    main()