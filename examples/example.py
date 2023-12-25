# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,5,6,7'
# import torch
# print(f"Available GPUs: {torch.cuda.device_count()}")

from conformer import Calibrator, Sampler, Components
from random import randint
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
from transformers import StoppingCriteria, StoppingCriteriaList
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

import os
import torch

# Set the dataset
dataset = load_dataset('cnn_dailymail', '3.0.0')

# Format the QA
x = dataset['train'][:50]['article']
x = [x_i[:] + ". Write a highlight for the article: " for x_i in x]
y = dataset['train'][:50]['highlights']

# Set the model name
model_name = 'psmathur/orca_mini_3b'

# Set the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, low_cpu_mem_usage=True)
tokenizer.pad_token = tokenizer.eos_token


# Set the model
model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            low_cpu_mem_usage=True
        )

model.config.pad_token_id = model.config.eos_token_id
model.config.use_cache = True

# Set the calibrator
calibrator = Calibrator(
    model=model,
    tokenizer=tokenizer,
    calibration_prompts=x,
    calibration_targets=y,
    # delta=0.2,
    epsilon=0.3,
    calibration_path="calibration_store.pkl",
)

# Set the threshold
group_conf_lambdas = torch.tensor([0.2, 0.4, 0.6, 0.8, 1])
nll_rej_lambdas = torch.tensor([0.3, 0.6, 0.9, 1.2, 1.5, 1.8])
rouge_rej_lambdas = torch.tensor([0.2, 0.4, 0.6, 0.8, 0.9])


# Set the admission function
calibrator.set_admission_function(
    func=Components.admission.ppl_prefix(tokenizer, model, 'Article:'),
)

# Set the confidence function
calibrator.set_group_confidence_function(
    Components.group_confidence.sum_ngll, 
    group_conf_lambdas
)

# Set the diversity rejection function
calibrator.add_rejection_function(
    Components.rejection.rouge_1, 
    rouge_rej_lambdas
)

# Set the quality (by log-likelihood) rejection function
calibrator.add_rejection_function(
    Components.rejection.ngll, 
    nll_rej_lambdas
)

# Set the stopping criteria by the FWER controlling algorithm
calibrator.set_FWER(
    fwer_algorithm=Components.FWER.none_debug
)

# Search for the optimal lambda values
lambdaz = calibrator.search()

# Set the sampler
sampler = Sampler.from_calibrator(calibrator, lambdaz)

# Set the question prompt
prompt = 'The Londoner has filmed a TV movie called "My Boy Jack," about author Rudyard Kipling and his son, due for release later this year. Write a spotlight for this:'

# Get the response
responses = sampler.sample_with_rejection(prompt, 5)

# Decode the response
decoded_responses = [tokenizer.decode(response, skip_special_tokens=True) 
                     for response in responses]

# Print the response
for response in decoded_responses:
    print(response)
