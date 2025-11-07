# Intervention Module Quick Reference

## Activation Patching

### Basic Activation Patching
```python
from intervention import activation_patch_experiment

result = activation_patch_experiment(
    model, tokenizer,
    clean_prompt="The Eiffel Tower is in",
    corrupted_prompt="The Colosseum is in",
    layer_name="6",  # Layer index
    position=-1      # Token position (-1 = last)
)

# Access results
clean_logits = result['clean_logits']
patched_logits = result['patched_logits']
corrupted_logits = result['corrupted_logits']
```

### Path Patching (Test All Layers)
```python
from intervention import path_patching

layer_effects = path_patching(
    model, tokenizer,
    clean_prompt="The Eiffel Tower is in",
    corrupted_prompt="The Colosseum is in",
    layers_to_test=[0, 3, 6, 9, 11]  # Or None for all layers
)

# Results: {layer_idx: causal_effect_score}
for layer, effect in sorted(layer_effects.items()):
    print(f"Layer {layer}: {effect:.4f}")
```

### Manual Patching with InterventionHandler
```python
from intervention import InterventionHandler, ActivationPatch
import torch

handler = InterventionHandler(model, tokenizer)

# Define a patch
patch = ActivationPatch(
    layer_name="transformer.h.5",
    position=-1,  # Last token
    value=torch.zeros(768),  # New activation value
    mode='replace'  # 'replace', 'add', or 'subtract'
)

handler.register_activation_patch(patch)

# Run with patching
with handler:
    outputs = model(**inputs)
```

---

## Steering Vectors

### Create Steering Vector from Contrastive Prompts
```python
from intervention import create_steering_vector

steering_vec = create_steering_vector(
    model, tokenizer,
    positive_prompts=[
        "I am happy and excited",
        "This is wonderful and great",
        "I feel joyful"
    ],
    negative_prompts=[
        "I am sad and upset",
        "This is terrible and awful",
        "I feel miserable"
    ],
    layer_name="transformer.h.6",
    normalize=True  # Normalize to unit length
)

print(f"Steering vector shape: {steering_vec.shape}")
print(f"Norm: {steering_vec.norm().item()}")
```

### Generate Text with Steering
```python
from intervention import steer_generation

output = steer_generation(
    model, tokenizer,
    prompt="Today I feel",
    steering_vector=steering_vec,
    layer_name="transformer.h.6",
    coefficient=2.0,  # Steering strength (can be negative)
    max_length=50,
    positions=None  # None = all positions, or list of positions
)

print(output)
```

### Manual Steering with InterventionHandler
```python
from intervention import InterventionHandler, SteeringVector

handler = InterventionHandler(model, tokenizer)

steering = SteeringVector(
    vector=steering_vec,
    layer_name="transformer.h.6",
    coefficient=1.5,
    positions=[5, 6, 7]  # Only steer these positions
)

handler.register_steering_vector(steering)

with handler:
    outputs = model.generate(**inputs, max_length=50)
```

---

## Feature Attribution

### Find Important Neurons
```python
from intervention import compute_feature_attribution

top_features = compute_feature_attribution(
    model, tokenizer,
    prompt="The capital of France is",
    layer_name="6",
    position=-1,
    n_features=10  # Return top 10
)

for feature_idx, importance in top_features:
    print(f"Feature {feature_idx}: {importance:.4f}")
```

---

## Causal Tracing

### Trace MLP Layers
```python
from intervention import CausalTracer

tracer = CausalTracer(model, tokenizer)

mlp_effects = tracer.trace_mlp_layers(
    clean_prompt="The Eiffel Tower is in",
    corrupted_prompt="The Colosseum is in"
)

for layer, effect in sorted(mlp_effects.items()):
    print(f"MLP Layer {layer}: {effect:.4f}")
```

---

## Advanced Usage

### Multiple Patches at Once
```python
from intervention import patch_and_run, ActivationPatch

patches = [
    ActivationPatch(
        layer_name="transformer.h.3",
        position=-1,
        value=activation_3,
        mode='replace'
    ),
    ActivationPatch(
        layer_name="transformer.h.6",
        position=-1,
        value=activation_6,
        mode='replace'
    ),
]

logits = patch_and_run(model, tokenizer, "Your prompt", patches)
```

### Custom Patch Function
```python
# Patch with a custom function instead of a fixed value
def scale_activation(act):
    return act * 2.0  # Double all activations

patch = ActivationPatch(
    layer_name="transformer.h.5",
    position=slice(None),  # All positions
    value=scale_activation,  # Callable
    mode='replace'
)
```

### Patch Multiple Positions
```python
# Patch specific positions
patch = ActivationPatch(
    layer_name="transformer.h.5",
    position=[2, 3, 4],  # Positions 2, 3, 4
    value=torch.randn(768),
    mode='add'
)

# Patch a range
patch = ActivationPatch(
    layer_name="transformer.h.5",
    position=slice(5, 10),  # Positions 5-9
    value=torch.zeros(768),
    mode='replace'
)
```

### Context Manager Pattern
```python
# Best practice: use context manager to ensure cleanup
with InterventionHandler(model, tokenizer) as handler:
    handler.register_activation_patch(patch1)
    handler.register_steering_vector(steering1)
    
    # Run model with interventions
    outputs = model(**inputs)
    
# Hooks automatically cleaned up after with block
```

---

## Common Patterns

### Compare Steering Strengths
```python
coefficients = [-2.0, -1.0, 0.0, 1.0, 2.0]
prompt = "I think this is"

for coef in coefficients:
    output = steer_generation(
        model, tokenizer, prompt,
        steering_vector=steering_vec,
        layer_name="transformer.h.6",
        coefficient=coef,
        max_length=30
    )
    print(f"Coefficient {coef:+.1f}: {output}")
```

### Find Most Causal Layer
```python
layer_effects = path_patching(model, tokenizer, clean, corrupted)
most_causal = max(layer_effects.items(), key=lambda x: x[1])
print(f"Most causal layer: {most_causal[0]} (effect: {most_causal[1]:.4f})")
```

### Ablate Specific Neurons
```python
# Zero out specific neurons
def ablate_neurons(act, neuron_indices=[10, 20, 30]):
    act_copy = act.clone()
    act_copy[:, :, neuron_indices] = 0
    return act_copy

patch = ActivationPatch(
    layer_name="transformer.h.6",
    position=slice(None),
    value=lambda act: ablate_neurons(act, [10, 20, 30]),
    mode='replace'
)
```

### Steering in Multiple Layers
```python
handler = InterventionHandler(model, tokenizer)

for layer_idx in [4, 5, 6]:
    steering = SteeringVector(
        vector=steering_vec,
        layer_name=f"transformer.h.{layer_idx}",
        coefficient=1.0
    )
    handler.register_steering_vector(steering)

with handler:
    outputs = model(**inputs)
```

---

## Tips & Best Practices

1. **Start Small**: Test on single layers before scaling to full path patching
2. **Use Context Managers**: Always use `with handler:` to ensure cleanup
3. **Check Layer Names**: Use `model.named_modules()` to verify layer names
4. **Normalize Steering Vectors**: Usually helps with stability
5. **Experiment with Coefficients**: Try both positive and negative values
6. **Position Indexing**: Remember -1 is last token, 0 is first
7. **Batch Size**: Most functions assume batch_size=1
8. **Device Management**: Ensure steering vectors match model device
9. **Save Steering Vectors**: `torch.save(steering_vec, 'steering.pt')`
10. **Visualize Effects**: Combine with logit lens to see how patches affect predictions

---

## Troubleshooting

### "Could not find layer"
```python
# Check available layer names
for name, module in model.named_modules():
    if 'h' in name or 'layer' in name:
        print(name)
```

### Position Index Error
```python
# Check sequence length
inputs = tokenizer("Your text", return_tensors="pt")
print(f"Sequence length: {inputs['input_ids'].shape[1]}")

# Use valid position
position = min(-1, inputs['input_ids'].shape[1] - 1)
```

### Steering Not Working
```python
# Try different layers
for layer_idx in range(0, 12, 2):
    output = steer_generation(
        model, tokenizer, prompt, steering_vec,
        layer_name=f"transformer.h.{layer_idx}",
        coefficient=2.0
    )
    print(f"Layer {layer_idx}: {output}")
```

### Shape Mismatch
```python
# Ensure steering vector matches hidden size
config = model.config
hidden_size = config.n_embd if hasattr(config, 'n_embd') else config.hidden_size
steering_vec = steering_vec.view(hidden_size)  # Reshape if needed
```
