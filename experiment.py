import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    model_path = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Llama-3.1-8B"
    analyzer = ModelAnalyzer(model_path, device=device)
    
    # Load model before using it
    analyzer.load_model()
    analyzer.print_architecture_summary()

    model_num_layers = analyzer.get_num_layers()


    generated_text = analyzer.generate(
        prompt="0+0=",
        max_new_tokens=10
    )

    record = analyzer.extract_activations(text=generated_text, 
        layer_names=None,
        layer_indices=None,
        include_attention=True,
        return_logits=False,
        metadata=None
    )

    base_activations = record.layer_activations
    base_attn_weights = record.attention_weights


    test_prompts = []
    test_prompts += [f"{i}+{j}=" for i in range(1, 100) for j in range(1, 100)]
    #test_prompts += [f"{i}00+{j}00=" for i in range(1, 6) for j in range(1, 6)]
    #test_prompts += [f"336+639=", "958+501=", "698+316="]

    # Get list of layer names from base_activations
    layer_names_list = list(base_activations.keys())
    print(f"Found {len(layer_names_list)} layers: {layer_names_list[:3]}...{layer_names_list[-3:]}")
    
    dotted_with_base = {
        layer_name: np.zeros((99, 99))
        for layer_name in layer_names_list
    }
    
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt: {prompt}")
        
        generated_text = analyzer.generate(
            prompt=prompt,
            max_new_tokens=10
        )

        record = analyzer.extract_activations(text=generated_text, 
            layer_names=None,
            layer_indices=None,
            include_attention=True,
            return_logits=False,
            metadata=None
        )

        y, x = prompt.split("+")
        y = int(y)
        x = int(x.replace("=", ""))

        for layer_name, layer_activations in record.layer_activations.items():
            print(f"Layer: {layer_name}, Activations shape: {layer_activations.shape}")

            dotted_with_base[layer_name][y-1, x-1] = np.dot(base_activations[layer_name][0, 4, :].cpu().numpy(), layer_activations[0, 4, :].cpu().numpy())

        del record
        del generated_text
        torch.cuda.empty_cache()

    # Save records for later use
    # analyzer.save_activations(record=records, output_path="/home/wuroderi/scratch/logits/activation_records.pt", format="pt")


    print("Plotting")
    for layer in layer_names_list:
        plt.figure(figsize=(16, 14))
        # Use colorbar to show values instead
        sns.heatmap(dotted_with_base[layer], annot=False, cmap="viridis", 
                    cbar_kws={'label': 'Dot Product Value'})
        plt.title("Dot Product of Activations with Base Activation (Prompt 'X+Y=')", 
                  fontsize=14, pad=20)
        plt.xlabel("X values", fontsize=12)
        plt.ylabel("Y values", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"PLOTS/activation_dot_product_layer_{layer}.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory



    
    


if __name__ == "__main__":
    main()