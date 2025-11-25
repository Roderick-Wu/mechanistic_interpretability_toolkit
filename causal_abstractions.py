# %%
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
import random
import copy
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm, trange

from sklearn.metrics import classification_report
from transformers import get_linear_schedule_with_warmup

import pyvene as pv
from pyvene import CausalModel
from pyvene.models.mlp.modelings_mlp import MLPConfig
from pyvene import create_mlp_classifier
from pyvene import (
    IntervenableModel,
    VanillaIntervention,
    RotatedSpaceIntervention,
    LowRankRotatedSpaceIntervention,
    RepresentationConfig,
    IntervenableConfig,
)


from model_analyzer import ModelAnalyzer
from activation_extraction import (
    ActivationRecord,
    save_activations,
    load_activations,
    compare_activations,
    get_activation_statistics
)
from intervention import (
    InterventionHandler,
    ActivationPatch,
    SteeringVector,
    create_steering_vector as _create_steering_vector
)

# %%
#device = "cuda" if torch.cuda.is_available() else "cpu"
#analyzer = ModelAnalyzer("../models/Llama-3.2-1B", device=device)
#analyzer.load_model()

#tokenizer = analyzer.tokenizer
#model = analyzer.model

# %%
def number_encoding(number):
    return tokenizer.encode(str(number), add_special_tokens=False)
    
number_of_entities = 20
#variables = ["X1X0", "Y1Y0", "X1X0*Y0", "Y1Y0*X0", "X1X0*Y1*10", "Y1Y0*X1*10", "Output"]
variables = ["X1X0", "Y1Y0", "X1X0*Y0", "X1X0*Y1*10", "Output"]
reps_start = [i for i in range(10, 20)]
reps_intermediate_1 = [i*j for i in range(10, 20) for j in range(0, 10)]
reps_intermediate_2 = [i*j for i in range(10, 20) for j in (10, )]
# Output = X1X0*Y0 + X1X0*Y1*10 = (X1X0 * (Y1Y0 % 10)) + (X1X0 * (Y1Y0 // 10) * 10)
# Range: (10*0) + (10*1*10) to (19*9) + (19*1*10) = 100 to 361
reps_final = [i*j for i in range(10, 20) for j in range(10, 20)]  # This is 100 to 361
values = {variable: reps_start for variable in ["X1X0", "Y1Y0"]}
#values |= {variable: reps_intermediate for variable in ["X1X0*Y0", "X1X0*Y1*10"]}
values["X1X0*Y0"] = reps_intermediate_1
values["X1X0*Y1*10"] = reps_intermediate_2
values |= {variable: reps_final for variable in ["Output"]}

for value in values:
    print(value)
    print(values[value])

#parents = {
    #"X1X0": [],
    #"Y1Y0": [],
    #"X1X0*Y0": ["X1X0", "Y1Y0"],
    #"Y1Y0*X0": ["Y1Y0", "X1X0"],
    #"X1X0*Y1*10": ["X1X0", "Y1Y0"],
    #"Y1Y0*X1*10": ["Y1Y0", "X1X0"],
    #"Output": ["X1X0*Y0", "Y1Y0*X0", "X1X0*Y1*10", "Y1Y0*X1*10"]
#}
parents = {
    "X1X0": [],
    "Y1Y0": [],
    "X1X0*Y0": ["X1X0", "Y1Y0"],
    "X1X0*Y1*10": ["X1X0", "Y1Y0"],
    "Output": ["X1X0*Y0", "X1X0*Y1*10"]
}

functions = {
    "X1X0": lambda: random.randint(10, 19),
    "Y1Y0": lambda: random.randint(10, 19),
    "X1X0*Y0": lambda x, y: x * (y % 10),
    "X1X0*Y1*10": lambda x, y: x * (y // 10) * 10,
    "Output": lambda a, b: a + b,
}

pos = {
    "X1X0": (0, 0.4),
    "Y1Y0": (0, 0.6),
    "X1X0*Y0": (2, 0.4),
    "X1X0*Y1*10": (2, 0.6),
    "Output": (4, 0.5)
}

causal_model = CausalModel(
    variables=variables,
    parents=parents,
    functions=functions,
    values=values,
    pos=pos
)

# %%
n_examples = 1000
#import pdb; pdb.set_trace()
dataset_fact = causal_model.generate_factual_dataset(
    size=n_examples, 
    sampler=causal_model.sample_input_tree_balanced
)

quit()


# %%
config = IntervenableConfig(
    model_type=type(model),
    representations=[
        RepresentationConfig(
            0,  # layer
            "block_output",  # intervention type
            "pos",  # intervention unit is now aligned with tokens
            1,  # max number of unit
            subspace_partition=None,  # binary partition with equal sizes
            intervention_link_key=0,
        ),
        RepresentationConfig(
            0,  # layer
            "block_output",  # intervention type
            "pos",  # intervention unit is now aligne with tokens
            1,  # max number of unit
            subspace_partition=None,  # binary partition with equal sizes,
            intervention_link_key=0,
        ),
    ],
    intervention_types=RotatedSpaceIntervention,
)

print(config)

# loss function
loss_fct = torch.nn.CrossEntropyLoss()

def calculate_loss(logits, label):
    """Calculate cross entropy between logits and a single target label (can be batched)"""
    shift_labels = label.to(logits.device)
    loss = loss_fct(logits, shift_labels)
    return loss


#ataset = 

# %%
intervenable = IntervenableModel(config, model)
intervenable.set_device(device)
intervenable.disable_model_gradients()
#print(intervenable.intervention_hooks)
#print(intervenable.model)

# %%
epochs = 10
gradient_accumulation_steps = 1
total_step = 0
target_total_step = len(dataset) * epochs

t_total = int(len(dataset) * epochs)
optimizer_params = []
for k, v in intervenable.interventions.items():
    optimizer_params += [{"params": v.rotate_layer.parameters()}]
    break
optimizer = torch.optim.Adam(optimizer_params, lr=0.001)

def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        total_count += 1
        correct_count += eval_pred == eval_label
    accuracy = float(correct_count) / float(total_count)
    return {"accuracy": accuracy}


def compute_loss(outputs, labels):
    CE = torch.nn.CrossEntropyLoss()
    return CE(outputs, labels)


def batched_random_sampler(data):
    batch_indices = [_ for _ in range(int(len(data) / batch_size))]
    random.shuffle(batch_indices)
    for b_i in batch_indices:
        for i in range(b_i * batch_size, (b_i + 1) * batch_size):
            yield i

# %%


# %%
# Causal graph for 12*14=168

variables = ["12", "14", "12*4", "14*2", "12*10", "14*10", "12*4+", "Output"]


