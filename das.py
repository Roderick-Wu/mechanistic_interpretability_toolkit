# %%
import os
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
    BoundlessRotatedSpaceIntervention,
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
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
analyzer = ModelAnalyzer("../models/Llama-3.2-1B", device=device)
analyzer.load_model()

tokenizer = analyzer.tokenizer
model = analyzer.model
model.eval()

# %%
def pricing_tag_game_example_sampler(
    tokenizer,
    amount,
    lower_bound,
    bound_width,
):
    (
        lower_bound_sample,
        upper_bound_sample,
        amount_sample,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)

    print(lower_bound_sample, upper_bound_sample, amount_sample)
    lower_bound_str = "%.2f" % lower_bound_sample
    upper_bound_str = "%.2f" % upper_bound_sample
    if amount_sample >= float(lower_bound_str) and amount_sample <= float(
        upper_bound_str
    ):
        label = tokenizer.convert_tokens_to_ids("Yes")
    else:
        label = tokenizer.convert_tokens_to_ids("No")

    amount_str = "%.2f dollars" % amount_sample
    instruction = f"Please say yes only if it costs between {lower_bound_str} and {upper_bound_str} dollars, otherwise no."
    alpaca_prompt = f"{instruction}, {amount_str}"
    input_ids = tokenizer(alpaca_prompt, return_tensors="pt").input_ids[0]
    output_ids = (torch.ones(input_ids.shape[0]) * -100).long().tolist()
    output_ids[-1] = label
    input_ids = input_ids.tolist()
    #assert len(input_ids) == 82
    print(alpaca_prompt, "fdsfadsf", label)
    print(input_ids, output_ids)

    return input_ids, output_ids

def custom_sampler():
    a = random.randint(10, 29)
    b = random.randint(10, 29)
    prompt = f"{a} + {b} = "
    answer = a + b
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
    output_ids = (torch.ones(input_ids.shape[0]) * -100).long().tolist()
    output_ids[-1] = tokenizer.convert_tokens_to_ids(f"{answer}")

    return input_ids, output_ids


def factual_sampler(
    tokenizer,
    max_n_training_examples,
):
    all_input_ids = []
    all_output_ids = []
    for _ in range(max_n_training_examples):
        input_ids, output_ids = custom_sampler()
        
        all_input_ids += [input_ids]
        all_output_ids += [output_ids]

    return all_input_ids, all_output_ids


raw_prealign = factual_sampler(tokenizer, 500)
prealign_dataset = Dataset.from_dict(
    {"input_ids": raw_prealign[0], "labels": raw_prealign[1]}
)
prealign_dataset.set_format("torch", columns=["input_ids", "labels"])
prealign_dataloader = DataLoader(prealign_dataset, batch_size=8)

# %%
total_count = 0
correct_count = 0
with torch.no_grad():
    for step, inputs in enumerate(tqdm(prealign_dataloader)):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)

        # aligning forward!
        outputs = model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        )

        actual_test_labels = inputs["labels"][:, -1]
        pred_test_labels = torch.argmax(outputs.logits[:, -1], dim=-1)

        #print(tokenizer.decode(tokens for tokens in inputs["input_ids"]))
        #for i, tokens in enumerate(inputs["input_ids"]):
            #print(tokenizer.decode(tokens))
            #print(tokenizer.decode(pred_test_labels[i]))

        correct_labels = actual_test_labels == pred_test_labels

        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()

current_acc = round(correct_count / total_count, 2)
print(f"[WARNING: THIS NEEDS TO BE GOOD!] prealign task accuracy: {current_acc}")

# %%
del(prealign_dataloader)

# %%
pv.set_seed(0)


def sample_with_region(region, lower_bound_sample, upper_bound_sample):
    if region == 1:
        amount_sample = round(random.uniform(0.01, lower_bound_sample - 0.01), 2)
    elif region == 2:
        amount_sample = round(random.uniform(lower_bound_sample, upper_bound_sample), 2)
    elif region == 3:
        amount_sample = round(random.uniform(upper_bound_sample + 0.01, 9.99), 2)
    return amount_sample


def lower_bound_alignment_example_sampler(
    tokenizer, amount=None, lower_bound=None, bound_width=None
):
    (
        base_lower_bound_sample,
        base_upper_bound_sample,
        _,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)
    (
        source_lower_bound_sample,
        source_upper_bound_sample,
        _,
    ) = pricing_tag_game_config_sampler(amount, lower_bound, bound_width)

    ctf_label_str = random.choice(["Yes", "No"])
    if ctf_label_str == "Yes":
        ctf_label = tokenizer.convert_tokens_to_ids("Yes")
        base_source_regions = [
            [1, 2],
            [1, 3],
            [2, 2],
            [2, 3],
        ]
    elif ctf_label_str == "No":
        ctf_label = tokenizer.convert_tokens_to_ids("No")
        base_source_regions = [[1, 1], [2, 1], [3, 1], [3, 2], [3, 3]]
    base_source_region = random.choice(base_source_regions)
    base_region = base_source_region[0]
    source_region = base_source_region[1]

    base_amount_sample = sample_with_region(
        base_region, base_lower_bound_sample, base_upper_bound_sample
    )
    source_amount_sample = sample_with_region(
        source_region, source_lower_bound_sample, source_upper_bound_sample
    )

    return (
        base_lower_bound_sample,
        base_upper_bound_sample,
        source_lower_bound_sample,
        source_upper_bound_sample,
        base_amount_sample,
        source_amount_sample,
        ctf_label,
        ctf_label_str,
    )
def pricing_tag_game_config_sampler(amount, lower_bound, bound_width):
    if bound_width == None:
        bound_width_sample = round(random.uniform(2.50, 7.50), 2)
    else:
        bound_width_sample = bound_width
    if lower_bound == None:
        lower_bound_sample = round(random.uniform(0.05, 9.95 - bound_width_sample), 2)
        # left a little room to cover corner cases.
    else:
        lower_bound_sample = lower_bound
    upper_bound_sample = bound_width_sample + lower_bound_sample
    if amount == None:
        amount_sample = round(random.uniform(0.01, 9.99), 2)
    else:
        amount_sample = amount

    return lower_bound_sample, upper_bound_sample, amount_sample

def bound_alignment_sampler(
    tokenizer,
    max_n_training_examples,
    bound_functors,
    amount=None,
    lower_bound=None,
    bound_width=None,
):
    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = []  # this one does not have input ids, etc..
    all_intervention_ids = []

    for _ in range(max_n_training_examples):
        bound_functor = random.choice(bound_functors)
        (
            base_lower_bound_sample,
            base_upper_bound_sample,
            source_lower_bound_sample,
            source_upper_bound_sample,
            base_amount_sample,
            source_amount_sample,
            ctf_label,
            ctf_label_str,
        ) = bound_functor(
            tokenizer,
            amount,
            lower_bound,
            bound_width,
        )

        base_amount_str = "%.2f dollars" % base_amount_sample
        source_amount_str = "%.2f dollars" % source_amount_sample
        base_lower_bound_str = "%.2f" % base_lower_bound_sample
        base_upper_bound_str = "%.2f" % base_upper_bound_sample
        source_lower_bound_str = "%.2f" % source_lower_bound_sample
        source_upper_bound_str = "%.2f" % source_upper_bound_sample

        print(f"base: [{base_lower_bound_str}, {base_upper_bound_str}], {base_amount_str}")
        print(f"source: [{source_lower_bound_str}, {source_upper_bound_str}], {source_amount_str}")
        print(f"ctf label: {ctf_label_str}")

        base_instruction = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no."
        source_instruction = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no."

        #base_alpaca_prompt = alpaca_prompt_template % (
            #base_instruction,
            #base_amount_str,
        #)
        #source_alpaca_prompt = alpaca_prompt_template % (
            #source_instruction,
            #source_amount_str,
        #)
        base_alpaca_prompt = f"Please say yes only if it costs between {base_lower_bound_str} and {base_upper_bound_str} dollars, otherwise no. {base_amount_str}"
        source_alpaca_prompt = f"Please say yes only if it costs between {source_lower_bound_str} and {source_upper_bound_str} dollars, otherwise no. {source_amount_str}"

        base_input_ids = tokenizer(base_alpaca_prompt, return_tensors="pt").input_ids[0]
        source_input_ids = tokenizer(
            source_alpaca_prompt, return_tensors="pt"
        ).input_ids[0]
        base_input_ids = base_input_ids.tolist()
        source_input_ids = source_input_ids.tolist()
        ctf_output_ids = (torch.ones(len(base_input_ids)) * -100).long().tolist()
        ctf_output_ids[-1] = ctf_label
        intervention_id = 0 if bound_functor == bound_functors[0] else 1

        print(bound_functor, bound_functors)
        print(intervention_id)

        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]

        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [intervention_id]

        #assert len(base_input_ids) == 82
        #assert len(source_input_ids) == 82

    return (
        all_base_input_ids,
        all_source_input_ids,
        all_ctf_output_ids,
        all_intervention_ids,
    )


def custom_sampler(
    tokenizer,
    max_n_training_examples,
):
    all_base_input_ids = []
    all_source_input_ids = []
    all_ctf_output_ids = []
    all_intervention_ids = []

    for _ in range(max_n_training_examples):
        base_a, base_b = random.randint(10, 29), random.randint(10, 29)
        source_a, source_b = random.randint(10, 29), random.randint(10, 29)
        
        base_prompt = f"{base_a} + {base_b} = "
        source_prompt = f"{source_a} + {source_b} = "
        
        base_answer = base_a + base_b
        source_answer = source_a + source_b
        
        base_input_ids = tokenizer(base_prompt, return_tensors="pt").input_ids[0].tolist()
        source_input_ids = tokenizer(source_prompt, return_tensors="pt").input_ids[0].tolist()
        
        #base_output_ids = (torch.ones(base_input_ids.shape[0]) * -100).long().tolist()
        #source_output_ids = (torch.ones(source_input_ids.shape[0]) * -100).long().tolist()
        
        #base_output_ids[-1] = tokenizer.convert_tokens_to_ids(f"{base_answer}")
        #source_output_ids[-1] = tokenizer.convert_tokens_to_ids(f"{source_answer}")



        ctf_output_ids = (torch.ones(len(base_input_ids)) * -100).long().tolist()
        ctf_output_ids[-1] = tokenizer.convert_tokens_to_ids(f"{source_answer}")

        intervention_id = 0


        #print(f"base: {base_prompt}{base_answer}")
        #print(f"source: {source_prompt}{source_answer}")
        #print(f"ctf label: {source_answer}")



        all_base_input_ids += [base_input_ids]
        all_source_input_ids += [source_input_ids]

        all_ctf_output_ids += [ctf_output_ids]
        all_intervention_ids += [intervention_id]

    return (
        all_base_input_ids,
        all_source_input_ids,
        all_ctf_output_ids,
        all_intervention_ids,
    )

###################
# data loaders
###################
#raw_data = bound_alignment_sampler(
    #tokenizer, 1, [lower_bound_alignment_example_sampler]
#)
raw_data = custom_sampler(
    tokenizer, 10000
)

# %%

raw_train = (
    raw_data[0][:8000],
    raw_data[1][:8000],
    raw_data[2][:8000],
    raw_data[3][:8000],
)
raw_eval = (
    raw_data[0][8000:9000],
    raw_data[1][8000:9000],
    raw_data[2][8000:9000],
    raw_data[3][8000:9000],
)
raw_test = (
    raw_data[0][9000:],
    raw_data[1][9000:],
    raw_data[2][9000:],
    raw_data[3][9000:],
)
train_dataset = Dataset.from_dict(
    {
        "input_ids": raw_train[0],
        "source_input_ids": raw_train[1],
        "labels": raw_train[2],
        "intervention_ids": raw_train[3],  # we will not use this field
    }
).with_format("torch")
train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
)
eval_dataset = Dataset.from_dict(
    {
        "input_ids": raw_eval[0],
        "source_input_ids": raw_eval[1],
        "labels": raw_eval[2],
        "intervention_ids": raw_eval[3],  # we will not use this field
    }
).with_format("torch")
eval_dataloader = DataLoader(
    eval_dataset,
    batch_size=16,
)
test_dataset = Dataset.from_dict(
    {
        "input_ids": raw_test[0],
        "source_input_ids": raw_test[1],
        "labels": raw_test[2],
        "intervention_ids": raw_test[3],  # we will not use this field
    }
).with_format("torch")
test_dataloader = DataLoader(
    test_dataset,
    batch_size=16,
)

# %%
def simple_boundless_das_position_config(model_type, intervention_type, layer):
    config = IntervenableConfig(
        model_type=model_type,
        representations=[
            RepresentationConfig(
                layer,              # layer
                intervention_type,  # intervention type
            ),
        ],
        intervention_types=BoundlessRotatedSpaceIntervention,
    )
    return config


config = simple_boundless_das_position_config(
    type(model), "block_output", 15
)
intervenable = IntervenableModel(config, model)
intervenable.set_device("cuda")
intervenable.disable_model_gradients()

# %%
t_total = int(len(train_dataloader) * 3)
warm_up_steps = 0.1 * t_total
optimizer_params = []
for k, v in intervenable.interventions.items():
    optimizer_params += [{"params": v.rotate_layer.parameters()}]
    optimizer_params += [{"params": v.intervention_boundaries, "lr": 1e-2}]
optimizer = torch.optim.Adam(optimizer_params, lr=1e-3)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=warm_up_steps, num_training_steps=t_total
)


# You can define your custom compute_metrics function.
def compute_metrics(eval_preds, eval_labels):
    total_count = 0
    correct_count = 0
    for eval_pred, eval_label in zip(eval_preds, eval_labels):
        actual_test_labels = eval_label[:, -1]
        pred_test_labels = torch.argmax(eval_pred[:, -1], dim=-1)
        correct_labels = actual_test_labels == pred_test_labels
        total_count += len(correct_labels)
        correct_count += correct_labels.sum().tolist()
    accuracy = round(correct_count / total_count, 2)
    return {"accuracy": accuracy}


epochs = 3
gradient_accumulation_steps = 4
total_step = 0
target_total_step = len(train_dataloader) * epochs
temperature_start = 50.0
temperature_end = 0.1
temperature_schedule = (
    torch.linspace(temperature_start, temperature_end, target_total_step)
    .to(torch.bfloat16)
    .to("cuda")
)
intervenable.set_temperature(temperature_schedule[total_step])


def calculate_loss(logits, labels):
    shift_logits = logits[..., :, :].contiguous()
    shift_labels = labels[..., :].contiguous()
    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss()
    shift_logits = shift_logits.view(-1, intervenable.model_config.vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)

    for k, v in intervenable.interventions.items():
        boundary_loss = 1.0 * v.intervention_boundaries.sum()
    loss += boundary_loss

    return loss

# %%
sample_prompt = tokenizer("24 + 19 = ", return_tensors="pt")
print(sample_prompt.input_ids[0].tolist())
print(tokenizer.convert_ids_to_tokens(sample_prompt.input_ids[0].tolist()))

sample_answer = tokenizer(f"{53}", return_tensors="pt")
print(sample_answer.input_ids[0].tolist())
print(tokenizer.convert_ids_to_tokens(sample_answer.input_ids[0].tolist()))

#print(model)
#print(intervenable.model)
print("llama trainable parameters: ", pv.count_parameters(intervenable.model))

# %%
intervenable.model.train()  # train enables drop-off but no grads
print("llama trainable parameters: ", pv.count_parameters(intervenable.model))
print("intervention trainable parameters: ", intervenable.count_parameters())
train_iterator = trange(0, int(epochs), desc="Epoch")
for epoch in train_iterator:
    epoch_iterator = tqdm(
        train_dataloader, desc=f"Epoch: {epoch}", position=0, leave=True
    )
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            #print(v)
            if v is not None and isinstance(v, torch.Tensor):
                # v = torch.zeros(v.shape).to(device)
                inputs[k] = v.to(device)
        b_s = inputs["input_ids"].shape[0]
        _, counterfactual_outputs = intervenable(
            {"input_ids": inputs["input_ids"]},
            [{"input_ids": inputs["source_input_ids"]}],
            {"sources->base": 6},  # swap 5th token
        )
        eval_metrics = compute_metrics(
            [counterfactual_outputs.logits], [inputs["labels"]]
        )

        # loss and backprop
        loss = calculate_loss(counterfactual_outputs.logits, inputs["labels"])
        loss_str = round(loss.item(), 2)
        epoch_iterator.set_postfix({"loss": loss_str, "acc": eval_metrics["accuracy"]})

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps
        loss.backward()
        if total_step % gradient_accumulation_steps == 0:
            if not (gradient_accumulation_steps > 1 and total_step == 0):
                optimizer.step()
                scheduler.step()
                intervenable.set_zero_grad()
                intervenable.set_temperature(temperature_schedule[total_step])
        total_step += 1

# %%
# evaluation on the test set
eval_labels = []
eval_preds = []
with torch.no_grad():
    epoch_iterator = tqdm(test_dataloader, desc=f"Test")
    for step, inputs in enumerate(epoch_iterator):
        for k, v in inputs.items():
            if v is not None and isinstance(v, torch.Tensor):
                inputs[k] = v.to("cuda")
        b_s = inputs["input_ids"].shape[0]
        _, counterfactual_outputs = intervenable(
            {"input_ids": inputs["input_ids"]},
            [{"input_ids": inputs["source_input_ids"]}],
            {"sources->base": 6},  # swap 80th token
        )
        eval_labels += [inputs["labels"]]
        eval_preds += [counterfactual_outputs.logits]
eval_metrics = compute_metrics(eval_preds, eval_labels)
print(eval_metrics)


