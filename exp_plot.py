import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.manifold import TSNE
from umap import UMAP
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




save_dir = "/home/wuroderi/scratch/SAVED/"
info_dict = json.load(open(f"{save_dir}/misc_info_dict.json", "r"))

print("Initializing ModelAnalyzer...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "/home/wuroderi/projects/def-zhijing/wuroderi/models/Llama-3.1-8B"
analyzer = ModelAnalyzer(model_path, device=device)
    
# Load model before using it
#analyzer.load_model()
#analyzer.print_architecture_summary()

#model_num_layers = analyzer.get_num_layers()
#layer_names_list = analyzer.get_layer_names()
#attn_layer_names = analyzer.get_attention_layer_names()
#print(attn_layer_names)

y, x = 3, 8

token_idx, idx_check = 5, 5

if False:
    idx_1, idx_2 = 1, 4

    for idx_check in (1, 4, 5):
        head_strength = np.zeros((model_num_layers, analyzer.model.config.num_attention_heads))

        for i, attn_layer in enumerate(layer_names_list):
            attn_weights = np.load(f"{save_dir}/attn_weights_attention_layer_{i}_{y}_{x}.npy")
            # (Batch, Heads, Query Len, Key Len)
            head_strength[i, :] = attn_weights[0, :, token_idx, idx_check]
    

        plt.figure(figsize=(10, 8))
        sns.heatmap(head_strength, cmap="viridis")
        plt.title(f"Attention Weights Heatmap ({y} + {x})")
        plt.xlabel("Attention Heads")
        plt.ylabel("Layers")
        plt.savefig(f"PLOTS/attn_weights_heatmap_{y}_{x}_{idx_check}.png")
        plt.close()

        head_strength = np.zeros((model_num_layers, analyzer.model.config.num_attention_heads))


if False:
    #tests = [(3, 8), (8, 3), (2, 4), (4, 2), (3, 3)]
    tests = [(10, 10), (19, 78), (78, 19), (30, 40), (40, 30), (48, 57), (57, 48), (67, 33), (33, 67)]


    for (y, x) in tests:
        head_strength = np.zeros((model_num_layers, analyzer.model.config.num_attention_heads))

        for i, attn_layer in enumerate(layer_names_list):
            attn_weights = np.load(f"{save_dir}/attn_weights_attention_layer_{i}_{y}_{x}.npy")
            # (Batch, Heads, Query Len, Key Len)
            head_strength[i, :] = attn_weights[0, :, token_idx, :token_idx+1].max(axis=-1).squeeze()
    

        plt.figure(figsize=(10, 8))
        sns.heatmap(head_strength, cmap="viridis")
        plt.title(f"Attention Weights Heatmap ({y} + {x})")
        plt.xlabel("Attention Heads")
        plt.ylabel("Layers")
        plt.savefig(f"PLOTS/attn_weights_heatmap_{y}_{x}.png")
        plt.close()

if True:
    #test_prompts += [f"{i} + {j} = " for i in range(0, 100) for j in range(0, 100)]
    model_num_layers = 32
    num_attention_heads = 32
    hidden_size = 4096
    layer_names_list = [f"model.layers.{i}" for i in range(model_num_layers)]

    all_prompts = [(y, x) for y in range(0, 41) for x in range(0, 41)]
    memory_prompts = [(y, x) for y in range(0, 21) for x in range(0, 21)] # single digits and teens
    memory_prompts += [(y, x) for y in range(15, 101, 5) for x in range(15, 101, 5)] # multiples of 5

    head_strength_group1 = np.zeros((model_num_layers, num_attention_heads))
    head_strength_group2 = np.zeros((model_num_layers, num_attention_heads))
    for i, layer in enumerate(layer_names_list):
        if i == 0:
            continue

        print("Processing layer:", layer)

        activations_group1 = np.empty((0, hidden_size))                
        activations_group2 = np.empty((0, hidden_size))

        for (y, x) in all_prompts: # iterate through all
            #activations = np.load(f"{save_dir}/activations_{layer}_{y}_{x}.npy")
            attn_weights = np.load(f"{save_dir}/attn_weights_attention_layer_{i}_{y}_{x}.npy")
                
            # (Batch, Heads, Query Len, Key Len)
            if (y, x) in memory_prompts:
                #activations_group1 = np.concatenate((activations_group1, activations[0, token_idx, :].reshape(1, -1)), axis=0)
                head_strength_group1[i, :] += attn_weights[0, :, token_idx, :token_idx + 1].mean(axis=-1).squeeze()
            else:
                #activations_group2 = np.concatenate((activations_group2, activations[0, token_idx, :].reshape(1, -1)), axis=0)
                head_strength_group2[i, :] += attn_weights[0, :, token_idx, :token_idx + 1].mean(axis=-1).squeeze()
            
        #print(f"Creating t-SNE visualization for layer {i}...")
        #print(f"  Group1 shape: {activations_group1.shape}, Group2 shape: {activations_group2.shape}")
        
        #all_activations = np.concatenate([activations_group1, activations_group2], axis=0)
        #labels = np.concatenate([
            #np.zeros(activations_group1.shape[0]),
            #np.ones(activations_group2.shape[0])
        #])
        
        ## Additional safety: remove features with zero variance
        #variances = np.var(all_activations, axis=0)
        #non_zero_var_mask = variances > 1e-10
        #if not np.all(non_zero_var_mask):
            #print(f"  Removing {np.sum(~non_zero_var_mask)} features with zero variance")
            #all_activations = all_activations[:, non_zero_var_mask]
        
        
        ## Adjust perplexity based on sample size
        #perplexity_value = min(30, (all_activations.shape[0] - 1) // 3)
        
        #try:
            #tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity_value, max_iter=1000)
            #embeddings_2d = tsne.fit_transform(all_activations)
        #except Exception as e:
            #print(f"  ERROR in t-SNE for layer {i}: {e}")
            #continue
        
        ## Create the plot
        #plt.figure(figsize=(10, 8))
        
        ## Plot group1 (memory prompts) in blue
        #mask_group1 = labels == 0
        #plt.scatter(embeddings_2d[mask_group1, 0], embeddings_2d[mask_group1, 1], 
                   #c='blue', label='Memory Prompts (Group 1)', alpha=0.6, s=20)
        
        ## Plot group2 (other prompts) in red
        #mask_group2 = labels == 1
        #plt.scatter(embeddings_2d[mask_group2, 0], embeddings_2d[mask_group2, 1], 
                   #c='red', label='Other Prompts (Group 2)', alpha=0.6, s=20)
        
        #plt.title(f't-SNE Visualization of Activations - Layer {i}')
        #plt.xlabel('t-SNE Dimension 1')
        #plt.ylabel('t-SNE Dimension 2')
        #plt.legend()
        #plt.grid(True, alpha=0.3)
        #plt.tight_layout()
        #plt.savefig(f'PLOTS/tsne_activations_layer_{i}.png', dpi=300)
        #plt.close()
        
        #print(f"Saved t-SNE plot for layer {i}")


    # After processing all layers, plot average head strengths
    plt.figure(figsize=(10, 8))
    sns.heatmap(head_strength_group1 / len(memory_prompts), cmap="viridis")
    plt.title(f"Average Attention Weights Heatmap - Memory Prompts (Group 1)")
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.savefig(f"PLOTS/avg_attn_weights_memory_prompts.png")
    plt.close() 

    plt.figure(figsize=(10, 8))
    sns.heatmap(head_strength_group2 / (len(all_prompts) - len(memory_prompts)), cmap="viridis")
    plt.title(f"Average Attention Weights Heatmap - Other Prompts (Group 2)")
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.savefig(f"PLOTS/avg_attn_weights_other_prompts.png")
    plt.close()