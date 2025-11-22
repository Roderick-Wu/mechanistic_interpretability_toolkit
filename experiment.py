import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
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

np.random.seed(0)
torch.manual_seed(0)

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
        prompt="0 - 0 = ",
        max_new_tokens=10
    )

    record = analyzer.extract_activations(text=generated_text, 
        layer_names=None,
        layer_indices=None,
        include_attention=True,
        return_logits=False,
        metadata=None
    )

    base_token_inds = record.tokens.index("Ġ=")

    base_activations = record.layer_activations
    base_activations = {k: v.cpu() for k, v in base_activations.items()}
    base_attn_weights = record.attention_weights
    base_attn_weights = {k: v.cpu() for k, v in base_attn_weights.items()}


    test_prompts = []
    test_prompts += [f"{i} + {j} = " for i in range(0, 100) for j in range(0, 100)]
    #test_prompts += [f"{i}00+{j}00=" for i in range(1, 6) for j in range(1, 6)]
    #test_prompts += [f"336+639=", "958+501=", "698+316="]

    # Get list of layer names from base_activations
    layer_names_list = list(base_activations.keys())
    attention_names_list = list(base_attn_weights.keys())
    print(f"Found {len(layer_names_list)} layers: {layer_names_list[:3]}...{layer_names_list[-3:]}")
    
    dotted_with_base = {
        layer_name: np.zeros((100, 100)) for layer_name in layer_names_list
    }
    max_attn_weights = {
        attention_layer: np.zeros((100, 100)) for attention_layer in attention_names_list
    }

    save_dir = "/home/wuroderi/scratch/SAVED/"
    info_dict = {}
    
    for i, prompt in enumerate(test_prompts):
        print(f"Prompt: {prompt}")
        prompt_info = {}
        
        generated_text = analyzer.generate(
            prompt=prompt,
            max_new_tokens=10
        )

        prompt_info['generated_text'] = generated_text

        record = analyzer.extract_activations(text=generated_text, 
            layer_names=None,
            layer_indices=None,
            include_attention=True,
            return_logits=False,
            metadata=None
        )

        prompt_info['tokens'] = record.tokens

        print(record.tokens)

        y, x = prompt.split("+")
        y = int(y.strip())
        x = int(x.replace("=", "").strip())

        token_ind = record.tokens.index("Ġ=")

        ######
        if token_ind != 5:
            print(prompt)
            print(f"Token index of '=': {token_ind}")
            print(record.tokens)
        ######

        for layer_name, layer_activations in record.layer_activations.items():
            #print(f"Layer: {layer_name}, Activations shape: {layer_activations.shape}")

            layer_activations_np = layer_activations.cpu().numpy()
            dotted_with_base[layer_name][y, x] = np.dot(base_activations[layer_name][0, base_token_inds, :], layer_activations_np[0, token_ind, :])
            np.save(f"{save_dir}/activations_{layer_name}_{y}_{x}.npy", layer_activations_np)

        for layer_name, layer_attn_weights in record.attention_weights.items():
            #print(f"Layer: {layer_name}, Attention Weights shape: {layer_attn_weights.shape}")

            layer_attn_weights_np = layer_attn_weights.cpu().numpy()
            # (Batch, Heads, Query Len, Key Len)
            max_attn_weights[layer_name][y, x] = np.argmax(layer_attn_weights_np[0, :, token_ind, token_ind])
            np.save(f"{save_dir}/attn_weights_{layer_name}_{y}_{x}.npy", layer_attn_weights_np)


        info_dict[prompt] = prompt_info
        del record
        del generated_text
        torch.cuda.empty_cache()
        
    json.dump(info_dict, open(f"{save_dir}/misc_info_dict.json", "w"), indent=4)
    # Save records for later use
    # analyzer.save_activations(record=records, output_path="/home/wuroderi/scratch/logits/activation_records.pt", format="pt")

    plot_dir = "/home/wuroderi/scratch/PLOTS"
    np.save(f"{save_dir}/misc_dotted_with_base.npy", dotted_with_base)
    np.save(f"{save_dir}/misc_max_attn_weights.npy", max_attn_weights)

    print("Plotting")
    for layer in layer_names_list:
        plt.figure(figsize=(16, 14))
        # Use colorbar to show values instead
        sns.heatmap(dotted_with_base[layer], annot=False, cmap="viridis", 
                    cbar_kws={'label': 'Dot Product Value'})
        plt.title("Dot Product of Activations with Base Activation (Prompt 'X + Y = ')", 
                  fontsize=14, pad=20)
        plt.xlabel("X values", fontsize=12)
        plt.ylabel("Y values", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/activation_dot_product_{layer}.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory

    for attention_layer in attention_names_list:
        plt.figure(figsize=(16, 14))
        sns.heatmap(max_attn_weights[attention_layer], annot=False, cmap="magma", 
                    cbar_kws={'label': 'Max Attention Weight'})
        plt.title("Max Attention Weights at '=' Token (Prompt 'X + Y = ')", 
                  fontsize=14, pad=20)
        plt.xlabel("X values", fontsize=12)
        plt.ylabel("Y values", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/max_attention_weights_{attention_layer}.png", 
                    dpi=150, bbox_inches='tight')
        plt.close()  # Close figure to free memory



    
    


if __name__ == "__main__":
    main()