print('python -m pip install --upgrade pip')
print('mkdir ntk_results')

import subprocess
import sys
import importlib
import os
import numpy as np
import time
import gc
import pickle
import copy
from typing import Dict, Tuple
import random
import logging
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, TrainerCallback, DataCollatorWithPadding
from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoConfig, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
from functorch import make_functional_with_buffers, vmap, jacrev
import evaluate


def install(package):
    try:
        __import__(package)
        print(f"{package} is already installed.")
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")

# List of packages to install
packages = ['transformers', 'datasets', 'peft', 'torch', 'evaluate']

# Install packages
for package in packages:
    install(package)

folder_name = 'reg_results'

# Create the folder if it doesn't exist
try:
    os.makedirs(folder_name, exist_ok=True)
    print(f"Folder '{folder_name}' created successfully.")
    
except OSError as e:
    print(f"Error creating folder '{folder_name}': {e}")

seed = 1337

def set_seed(seed=1337):
    """Set all seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Call it early in your script
set_seed(1337)  # Use any integer you prefer

torch.cuda.empty_cache()
# print(torch.cuda.memory_summary())
gc.collect()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# choose the model
model_name = "roberta-base"
# model_name = "EleutherAI/gpt-j-6B"
# 1. Model Selection
# model_name = "facebook/opt-125m"  #"gpt2"  # or "facebook/opt-125m"


lambda_ = 1
linear_lr = 4e-3

scale_param_influence = 1
sub_length = 100

ntk_sample_size = 32
chunk_size = 1
num_epoch = 10
rank = 8
ntk_iterations = 1
temperature = 1


print("\nTasks:")
print("sst2 , cola , mrpc, qqp , mnli , qnli , rte , wnli , stsb , imdb , yelp , amazon\n")

# Choose which dataset to use
dataset_name = input('Enter the name of task you want to use: ')

if dataset_name in ['sst2','cola','mrpc','qqp','mnli','qnli','rte','wnli','stsb']:
    dataset = load_dataset("glue", f'{dataset_name}')
elif dataset_name == "imdb":
    dataset = load_dataset("imdb") 
elif dataset_name == "yelp":
    dataset = load_dataset("yelp_review_full")
    dataset = dataset.map(lambda example: {'label': 1 if example['label'] > 3 else 0})
elif dataset_name == "amazon":
    dataset = load_dataset("amazon_reviews_multi", "en")
    dataset = dataset.map(lambda example: {'label': 1 if int(example['stars']) > 3 else 0})     # Convert to binary (1-2 stars: negative, 4-5 stars: positive)
else:
    raise ValueError("Unsupported dataset")

selected_layers = input('Enter the numbers of layers you want to select: (numbers up to 11 for roberta), separated by commas: ')
# Convert the input string to a list of integers
selected_layers = [int(layer.strip()) for layer in selected_layers.split(",") if layer.strip().isdigit()]

# Map shorthand to full names
param_mapping = {'q': 'query', 'v': 'value', 'k': 'key', 'a': 'all'}

# Get input from the user for parameters (q for query, v for value, k for key)
param_input = input("Enter the parameters you want to select (choose from 'q', 'v', 'k', 'a'), separated by commas: ")
selected_params = [param_mapping[param.strip()] for param in param_input.split(",") if param.strip() in param_mapping]


target_modules = []
idx_layer = ''
name_param_file = ''

if model_name == "gpt2":
    for i in selected_layers:
        # For GPT-2, all Q,K,V are in c_attn
        if any(param in ['query', 'key', 'value'] for param in selected_params):
            if f"gpt.h.{i}.attn.c_attn" not in target_modules:
                target_modules.append(f"gpt.h.{i}.attn.c_attn") # GPT-2 uses combined QKV
        idx_layer += str(i) + ','
                
elif model_name == "facebook/opt-125m":
    for i in selected_layers:
        for param in selected_params:
            if param == 'query':
                target_modules.append(f"decoder.layers.{i}.self_attn.q_proj")
            elif param == 'key':
                target_modules.append(f"decoder.layers.{i}.self_attn.k_proj")
            elif param == 'value':
                target_modules.append(f"decoder.layers.{i}.self_attn.v_proj")
        idx_layer += str(i) + ','

elif model_name == "EleutherAI/gpt-j-6B":
    # For GPT-J, we need to target different attention modules
    for i in selected_layers:
        for param in selected_params:
            if param == 'query':
                target_modules.append(f"transformer.h.{i}.attn.q_proj")
            elif param == 'key':
                target_modules.append(f"transformer.h.{i}.attn.k_proj")
            elif param == 'value':
                target_modules.append(f"transformer.h.{i}.attn.v_proj")
        idx_layer += str(i) + ','

else:
    # Original code for RoBERTa
    for i in selected_layers:
        for param in selected_params:
            target_modules.append(f"roberta.encoder.layer.{i}.attention.self.{param}")
        idx_layer += str(i) + ','


# Remove trailing comma from idx_layer
idx_layer = idx_layer[:-1]
# Concatenate the selected parameters for the file name
name_param_file += ','.join(selected_params)

print(f"Selected target modules: {target_modules}")
print(f"Generated idx_layer: {idx_layer}")
print(f"Generated name_param_file: {name_param_file}")


#-----------------------------------model

if model_name == "roberta-base":
    max_length = 512
elif model_name == "EleutherAI/gpt-j-6B":
    max_length = 2048
elif model_name == "gpt2":
    max_length = 1024
elif model_name == "facebook/opt-125m":
    max_length = 2048
else:
    max_length = 128

#---------------------------------- For collecting metrics
# Create a custom callback to track loss and accuracy
class TrainingMetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_loss_history = []
        self.eval_loss_history = []
        self.eval_metric_history = []
        self.eval_metric_mismatched_history = []  # For MNLI mismatched
        self.eval_metric_overall_history = [] # For MNLI overall

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_loss_history.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_loss_history.append(logs['eval_loss'])
                
            if dataset_name == 'mnli':
                if 'eval_accuracy' in logs:
                    self.eval_metric_history.append(logs['eval_accuracy'])  # This will be for matched
                if 'eval_mismatched_accuracy' in logs:
                    self.eval_metric_mismatched_history.append(logs['eval_mismatched_accuracy'])  
                if 'eval_overall_accuracy' in logs:
                    self.eval_metric_overall_history.append(logs['eval_overall_accuracy'])
            # elif dataset_name == 'cola':
            #     if 'eval_matthews_correlation' in logs:
            #         self.eval_metric_history.append(logs['eval_matthews_correlation'])
            elif dataset_name == 'stsb':
                if 'eval_pearson' in logs:
                    self.eval_metric_history.append(logs['eval_pearson'])
            else:
                if 'eval_accuracy' in logs:
                    self.eval_metric_history.append(logs['eval_accuracy'])

# Initialize the custom callback
metrics_callback = TrainingMetricsCallback()

#------------------------------------------------------------------------------------------------- 

if model_name in ["EleutherAI/gpt-j-6B", "gpt2", "facebook/opt-125m"]:
    # Load tokenizer for decoder-only models
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token    # Set pad token for models
else:
    # Load the tokenizer for RoBERTa and other encoder models
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

# 2. Tokenize the dataset with max sequence length of 512
def preprocess_function(examples):
    if model_name in ["EleutherAI/gpt-j-6B", "gpt2", "facebook/opt-125m"]:
        # For all decoder-only models and without using prompt
        if dataset_name in ['imdb', 'yelp', 'amazon']:
            return tokenizer(examples['text'], truncation=True, max_length=max_length, padding='max_length')
        elif dataset_name == 'qqp':
            return tokenizer(examples['question1'], examples['question2'], truncation=True, max_length=max_length, padding='max_length')
        elif dataset_name == 'qnli':
            return tokenizer(examples['question'], examples['sentence'], truncation=True, max_length=max_length, padding='max_length')
        elif dataset_name == 'mnli':
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, max_length=max_length, padding='max_length')
        elif dataset_name in ['mrpc', 'stsb', 'rte', 'wnli']: # stsb: Regression task
            # Tasks with sentence pairs
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=max_length, padding='max_length')
        elif dataset_name in ['sst2', 'cola']:
            # Single sentence tasks
            return tokenizer(examples['sentence'], truncation=True, max_length=max_length, padding='max_length')
        else:
            raise ValueError(f"Unsupported task: {dataset_name}")
              
    else:
        # For encoder models like RoBERTa
        if dataset_name in ['imdb', 'yelp', 'amazon']:
            return tokenizer(examples['text'], truncation=True, max_length=max_length)
        elif dataset_name == 'qqp':
            return tokenizer(examples['question1'], examples['question2'], truncation=True, max_length=max_length)
        elif dataset_name == 'qnli':
            return tokenizer(examples['question'], examples['sentence'], truncation=True, max_length=max_length)
        elif dataset_name == 'mnli':
            return tokenizer(examples['premise'], examples['hypothesis'], truncation=True, max_length=max_length)
        elif dataset_name in ['mrpc', 'stsb', 'rte', 'wnli']: # stsb: Regression task
            # Tasks with sentence pairs
            return tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=max_length)
        elif dataset_name in ['sst2', 'cola']:
            # Single sentence tasks
            return tokenizer(examples['sentence'], truncation=True, max_length=max_length)
        else:
            raise ValueError(f"Unsupported task: {dataset_name}")

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# This part prepares the input for NTK computation
if dataset_name in ['imdb', 'yelp', 'amazon']:
    texts_set_1 = tokenized_datasets["train"]['text'][:ntk_sample_size]
elif dataset_name == 'qqp':
    texts_set_1 = list(zip(tokenized_datasets["train"]['question1'][:ntk_sample_size],
                           tokenized_datasets["train"]['question2'][:ntk_sample_size]))
elif dataset_name == 'qnli':
    texts_set_1 = list(zip(tokenized_datasets["train"]['question'][:ntk_sample_size],
                           tokenized_datasets["train"]['sentence'][:ntk_sample_size]))
elif dataset_name in ['mrpc', 'stsb', 'rte', 'wnli']:
    texts_set_1 = list(zip(tokenized_datasets["train"]['sentence1'][:ntk_sample_size],
                           tokenized_datasets["train"]['sentence2'][:ntk_sample_size]))
elif dataset_name == 'mnli':
    texts_set_1 = list(zip(tokenized_datasets["train"]['premise'][:ntk_sample_size],
                           tokenized_datasets["train"]['hypothesis'][:ntk_sample_size]))
else:
    texts_set_1 = tokenized_datasets["train"]['sentence'][:ntk_sample_size]
    

def prepare_datasets(task_name, tokenized_datasets):
    if task_name == 'mnli':
        train_dataset = tokenized_datasets["train"]
        valid_dataset = {
            'matched': tokenized_datasets["validation_matched"],
            'mismatched': tokenized_datasets["validation_mismatched"]
        }
    else:
        train_dataset = tokenized_datasets["train"]
        if "validation" in tokenized_datasets:
            valid_dataset = tokenized_datasets["validation"]
        elif "test" in tokenized_datasets:
            valid_dataset = tokenized_datasets["test"]
    
    # Some GLUE tasks don't have a test set, so we need to check if it exists
    if "test" in tokenized_datasets:
        test_dataset = tokenized_datasets["test"]
    else:
        test_dataset = None
    
    return train_dataset, valid_dataset, test_dataset

train_dataset, valid_dataset, test_dataset = prepare_datasets(dataset_name, tokenized_datasets)


from transformers.modeling_outputs import SequenceClassifierOutput

class RobertaForSequenceClassificationWithTemp(RobertaForSequenceClassification):
    def __init__(self, config, temperature=1.0, **kwargs):  # Add **kwargs
        super().__init__(config)
        self.temperature = temperature

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, temperature=1.0, *model_args, **kwargs):
        # Create model with temperature
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.temperature = temperature
        return model

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        # Pass only the accepted arguments to super().forward()
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels
        )
        
        # Scale the logits
        if outputs.logits is not None:
            outputs.logits = outputs.logits / self.temperature
        return outputs

# Wrap the model for classification
num_labels = 3 if dataset_name == 'mnli' else 1 if dataset_name == 'stsb' else 2

if model_name == "EleutherAI/gpt-j-6B":
    # Load model
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model = GPTJForSequenceClassification(base_model, num_labels).to(device)  
    
elif model_name in ["gpt2", "facebook/opt-125m"]:
    # Create model instance
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    model = SmallGPTForSequenceClassification(base_model, num_labels=num_labels, temperature=temperature).to(device)
    
else:
    # Load the pre-trained RoBERTa model
    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    model = RobertaForSequenceClassificationWithTemp.from_pretrained(model_name, config=config, temperature=temperature).to(device)   #num_labels captured from config

# Total trainable parameters
total_trainable_params1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters: {total_trainable_params1}")

base_model = copy.deepcopy(model)

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS, # Task type: Sequence classification
    target_modules=target_modules,
    inference_mode=False,
    r=rank,                        # LoRA rank
    lora_alpha=8,               # LoRA alpha
    lora_dropout=0.1             # LoRA dropout
)

# Wrap the model with the LoRA configuration
model = get_peft_model(base_model, peft_config)
        
class LinearizedLoRAModel(nn.Module):
    def __init__(self, base_model, epsilon=1e-2, use_bias=False):
        super().__init__()
        
        # Store configuration
        self.epsilon = epsilon
        self.use_bias = use_bias
        self.config = base_model.config
        
        # Base model (frozen)
        self.base_model = copy.deepcopy(base_model) 
        self.base_model.eval()
        
        # Working model for perturbations
        self.working_model = copy.deepcopy(base_model)
        self.working_model.eval()
        
        # Freeze all parameters in both models
        for model in [self.base_model, self.working_model]:
            for param in model.parameters():
                param.requires_grad_(False)
        
        # Create bias parameter if needed
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(self.config.num_labels))
        
        # Find trainable parameters (LoRA parameters)
        trainable_params = []
        print("Trainable parameters in source model:")
        param_size = 0
        for name, param in base_model.named_parameters():
            if param.requires_grad:
                print(f"  {name}: {param.shape}")
                param_size += param.numel()
                trainable_params.append((name, param.shape))
        print(f"Total trainable parameters: {param_size}")
        
        # Create a single parameter vector for all deltas
        self.delta_vector = nn.Parameter(torch.randn(param_size) * 0.1)
        
        # Create mapping from vector to parameter shapes
        self.param_mapping = []
        current_pos = 0
        for name, shape in trainable_params:
            size = torch.tensor(shape).prod().item()
            self.param_mapping.append((name, shape, current_pos, current_pos + size))
            current_pos += size
    
    def get_param_deltas(self):
        """Map from delta vector to individual parameter deltas"""
        deltas = {}
        for name, shape, start, end in self.param_mapping:
            # Extract slice from delta vector and reshape
            delta = self.delta_vector[start:end].view(shape)
            deltas[name] = delta
        return deltas
    
    def apply_deltas_to_working_model(self, deltas, scale=1.0):
        """Apply parameter deltas to working model"""
        # Reset working model to match base model
        with torch.no_grad():
            for name, _ in self.working_model.named_parameters():
                if name in deltas:
                    # Get base parameter
                    base_param = None
                    for base_name, base_p in self.base_model.named_parameters():
                        if base_name == name:
                            base_param = base_p
                            break
                    
                    # Find working parameter
                    for work_name, work_p in self.working_model.named_parameters():
                        if work_name == name:
                            # Copy from base and add scaled delta
                            work_p.copy_(base_param + scale * deltas[name])
                            break
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        """Forward pass with linearization"""
        # Get current parameter deltas
        deltas = self.get_param_deltas()
        
        # Base model forward pass
        with torch.no_grad():
            base_outputs = self.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if token_type_ids is not None else None,
            )
            # base_logits = base_outputs.logits
            base_logits = base_outputs.logits.clone().detach()  # Detach to ensure no gradient dependencies

        
        # Apply scaled deltas to working model
        self.apply_deltas_to_working_model(deltas, scale=self.epsilon)
        
        # Working model forward pass
        with torch.no_grad():
            perturbed_outputs = self.working_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids if token_type_ids is not None else None,
            )
            perturbed_logits = perturbed_outputs.logits
        
        jvp = (perturbed_logits - base_logits) / self.epsilon

        # Combine base logits and JVP approximation (no gradients yet)
        with torch.no_grad():
            linearized_logits = base_logits + jvp

        # Get batch size and number of classes
        batch_size, num_classes = linearized_logits.shape
         # Create a copy of linearized_logits that will receive gradients
        linearized_logits = linearized_logits.clone()

        # Direct parameter connection, different for each class
        # This ensures proper gradient flow while maintaining the linearization concept
        for c in range(num_classes):
            # Get a subset of parameters for this class to ensure unique gradients
            param_idx = c % min(sub_length, len(self.delta_vector))
            param_influence = self.delta_vector[param_idx:(param_idx+100)].mean() * scale_param_influence
            linearized_logits[:, c] = linearized_logits[:, c] + param_influence
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(linearized_logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(linearized_logits.view(-1, self.config.num_labels), labels.view(-1))
        
        # Create return object
        from transformers.modeling_outputs import SequenceClassifierOutput
        return SequenceClassifierOutput(
            loss=loss,
            logits=linearized_logits,
        )

# Create the linearized model
linearized_model = LinearizedLoRAModel(
    base_model=base_model,  # Your fine-tuned LoRA model
    epsilon=1e-2,
    use_bias=False,
    # learning_scale=1.0  # This helps normalize the loss scale
)

linearized_model = linearized_model.to(device)
#--------------------------------------------------------------------------
# Save the initialized state BEFORE training
init_state_dict = {k: v.clone() for k, v in model.named_parameters()}

# Function to tokenize input text
def tokenize_input(texts, tokenizer, max_length=max_length):
    return tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

# Check which parameters are trainable
print("\nTrainable parameters after LoRA:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable: {name}")

# Count trainable parameters after LoRA
total_trainable_params2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total trainable parameters after LoRA: {total_trainable_params2}")
#--------------------------------------------------------

# Tokenize inputs
input_set_1 = tokenize_input(texts_set_1, tokenizer)
input_set_2 = input_set_1

# Move inputs to the same device as the model
input_set_1 = {k: v.to(device) for k, v in input_set_1.items()}
input_set_2 = {k: v.to(device) for k, v in input_set_2.items()}

model = model.to(device)

def get_word_embeddings(input_ids, attention_mask=None, chunk_size=512):
    """
    Get word embeddings in a memory-efficient way by processing in smaller chunks
    and using CPU offloading when necessary.
    """
    embeddings = []
    total_chunks = (input_ids.size(0) + chunk_size - 1) // chunk_size
    
    for i in range(0, input_ids.size(0), chunk_size):
        # Clear GPU cache before processing each chunk
        torch.cuda.empty_cache()
        
        try:
            # Process a smaller chunk
            chunk = input_ids[i:i+chunk_size]
            attention_chunk = attention_mask[i:i+chunk_size] if attention_mask is not None else None
            
            with torch.no_grad():
                if model_name == "facebook/opt-125m":
                    outputs = model.gpt.model.decoder.embed_tokens(chunk)
                elif model_name == "gpt2":
                    outputs = model.gpt.wte(chunk)
                else:
                    outputs = model.roberta.embeddings(chunk) # do all 1.Word embeddings 2.Position embeddings. 3.Token type embeddings. 4.Layer normalization. 5.Dropout
                    # outputs = model.roberta.embeddings.word_embeddings(chunk)   # Only converts token IDs to their corresponding vector representations
                    
                if dataset_name in ['mrpc', 'qqp', 'qnli', 'mnli', 'stsb', 'imdb', 'yelp', 'amazon']:
                # For sentence pair tasks, maintain the sequence dimension        
                    if attention_chunk is not None:
                        outputs = outputs * attention_chunk.unsqueeze(-1)
                
                # Move chunk result to CPU immediately to free GPU memory
                embeddings.append(outputs.cpu())
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                # If we still get OOM, try processing on CPU
                print(f"GPU OOM for chunk {i//chunk_size + 1}/{total_chunks}, falling back to CPU")
                torch.cuda.empty_cache()
                
                # Move chunk to CPU and process
                chunk = chunk.cpu()
                if attention_chunk is not None:
                    attention_chunk = attention_chunk.cpu()
                
                with torch.no_grad():
                    if model_name == "facebook/opt-125m":
                        outputs = model.gpt.model.decoder.embed_tokens.cpu()(chunk)
                    elif model_name == "gpt2":
                        outputs = model.gpt.wte.cpu()(chunk)
                    else:
                        outputs = model.roberta.embeddings.cpu()(chunk)
                        
                        
                    if dataset_name in ['mrpc', 'qqp', 'qnli', 'mnli', 'stsb', 'imdb', 'yelp', 'amazon']:
                    # For sentence pair tasks, maintain the sequence dimension   
                        if attention_chunk is not None:
                            outputs = outputs * attention_chunk.unsqueeze(-1)
                    
                    embeddings.append(outputs)
                
                # Move model back to GPU
                if model_name == "facebook/opt-125m":
                    model.gpt.model.decoder.embed_tokens.cuda()
                elif model_name == "gpt2":
                    model.gpt.wte.cuda()
            else:
                raise e
    
    # Concatenate all chunks
    result = torch.cat(embeddings, dim=0)
    
    # Move final result back to GPU if possible
    try:
        return result.cuda()
    except RuntimeError:
        print("Warning: Final result too large for GPU, keeping on CPU")
        return result

# Get word embeddings for the sentence
x_train = get_word_embeddings(input_set_1['input_ids'], input_set_1['attention_mask'])
# x_test = get_word_embeddings(input_set_2['input_ids'], input_set_2['attention_mask'])
x_test = x_train
     

class LogitModelWrapper(nn.Module):
    def __init__(self, model):
        super(LogitModelWrapper, self).__init__()
        self.model = model

    def forward(self, embeddings):
        if dataset_name in ['mrpc', 'qqp', 'qnli', 'mnli', 'stsb', 'imdb', 'yelp', 'amazon']:
            attention_mask = torch.ones(embeddings.shape[:2], device=embeddings.device)
        else:
            attention_mask = None

        if model_name in ["gpt2", "facebook/opt-125m"]:
            outputs = self.model.gpt(
                inputs_embeds=embeddings,
                attention_mask=attention_mask,
                output_hidden_states=True,  # Enable hidden states output
                return_dict=True  # Ensure we get a dictionary-like object
            )
            
            # Get the last hidden state from the tuple of all hidden states
            if outputs.hidden_states is not None:
                # Take the last layer's hidden states
                last_hidden_state = outputs.hidden_states[-1]
                # Take the last token representation
                logits = self.model.classifier(last_hidden_state[:, -1, :])
            else:
                # If no hidden states, use the raw output logits
                last_hidden_state = outputs.logits
                logits = self.model.classifier(last_hidden_state[:, -1, :])
                
            return logits
        else:
            # Original code for other models (RoBERTa etc.)
            outputs = self.model(
                inputs_embeds=embeddings, 
                attention_mask=attention_mask
            )
            return outputs.logits

model_wrapper = LogitModelWrapper(model).to(device)

# Function for a single pass through the functional model
def fnet_single(params, buffers, x):
    return fnet(params, buffers, x.unsqueeze(0)).squeeze(0)

def get_lora_params(model):
    return [(name, param) for name, param in model.named_parameters() if 'lora' in name.lower()]

def chunked_vmap(func, in_dims, chunk_size, *args, randomness='same'):
    results = []
    for i in range(0, args[0].shape[0], chunk_size):
        chunk_args = [arg[i:i+chunk_size] if isinstance(arg, torch.Tensor) else arg for arg in args]
        chunk_result = vmap(func, in_dims, randomness=randomness)(*(chunk_args))
        results.append(chunk_result)
    
    # Combine results correctly based on their structure
    if isinstance(results[0], list):
        return [torch.cat([r[i] for r in results]) for i in range(len(results[0]))]
    else:
        return torch.cat(results)

# Get LoRA parameters
lora_params = get_lora_params(model)
lora_param_indices = [i for i, (name, _) in enumerate(model.named_parameters()) if 'lora' in name.lower()]
lora_param_names = [name for i, (name, _) in enumerate(model.named_parameters()) if 'lora' in name.lower()]

def empirical_ntk_jacobian_contraction_lora(fnet_single, params, buffers, x1, x2, chunk_size, compute='full'):
    def epsilon_sign(x, epsilon=1e-8):
        return x / (torch.abs(x) + epsilon)

    def clear_memory():
        """Clear GPU cache and run garbage collection"""
        torch.cuda.empty_cache()
        gc.collect()
        
    def lora_jacrev(x):
        clear_memory()
        all_grads = jacrev(fnet_single, argnums=0)(params, buffers, x)
        lora_grads = [all_grads[i] for i in lora_param_indices]
        print('Number of LoRA param gradients------', len(lora_grads))
        
        # Clean up
        del all_grads
        clear_memory()
        
        return lora_grads
        # return all_grads

    try:
        # Compute J(x1) for LoRA parameters
        jac1 = chunked_vmap(lora_jacrev, (0,), chunk_size, x1, randomness='same')
        jac1 = [j.flatten(2) for j in jac1]

        # Compute J(x2) for LoRA parameters
        jac2 = chunked_vmap(lora_jacrev, (0,), chunk_size, x2, randomness='same')
        jac2 = [j.flatten(2) for j in jac2]

        # Compute J(x1) @ J(x2).T
        einsum_expr = 'Naf,Mbf->NMab' if compute == 'full' else 'Naf,Maf->NM'
        ntk_dict = {}
        j2_dict={}
        for j1, j2, name in zip(jac1, jac2, lora_param_names):
            clear_memory()
            # ntk = torch.einsum(einsum_expr, j1, j2)
            ntk = torch.einsum(einsum_expr, j1, epsilon_sign(j2))
            ntk_dict[name] = ntk
            j2_dict[name]=j2
            
            # Clean up intermediates
            del j1, j2
            clear_memory()
            
        del jac1, jac2
        clear_memory()
        
        # Summed result for LoRA parameters
        summed_ntk = sum(ntk_dict.values())

        return summed_ntk, ntk_dict, j2_dict

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory. Falling back to CPU computation.")
            clear_memory()
            x1, x2 = x1.cpu(), x2.cpu()
            params = [p.cpu() for p in params]
            buffers = [b.cpu() for b in buffers]
            torch.cuda.empty_cache()
            gc.collect()
            return empirical_ntk_jacobian_contraction_lora(fnet_single, params, buffers, x1, x2, chunk_size, compute)
        else:
            raise e


# Main NTK computation loop
output_dir = './ntk_results/'

# After creating the model but before training
model.gradient_checkpointing_disable()  # Explicitly disable

# Convert the BERT model to a functional model using functorch, including buffers
fnet, params, buffers = make_functional_with_buffers(model_wrapper)

from transformers import TrainingArguments

#-----------------------------------------------------------------------
# 5. Training arguments
# Modify training arguments for GPT-J
if model_name == "EleutherAI/gpt-j-6B":
    training_args = TrainingArguments(
        output_dir= f"./results_{model_name}_{dataset_name}_layer_{idx_layer}_param_{name_param_file}",
        # evaluation_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=4e-4,
        per_device_train_batch_size=8,  # Reduced batch size for GPT-J
        per_device_eval_batch_size=8,   # Reduced batch size for GPT-J
        num_train_epochs=num_epoch,
        gradient_accumulation_steps=4,   # Increased for GPT-J
        logging_dir='./logs',
        logging_steps=30,
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        fp16=True,
        gradient_checkpointing=False,    # Enable gradient checkpointing
        weight_decay=lambda_,  # Add weight decay here (common value for transformers)
        report_to="none",
    )
else:
    # Original training arguments for other models
    training_args = TrainingArguments(
        output_dir= f"./results_{model_name}_{dataset_name}_layer_{idx_layer}_param_{name_param_file}",
        # evaluation_strategy="epoch",
        eval_strategy="epoch",
        learning_rate=4e-3,  #4e-4
        per_device_train_batch_size=16, #32,
        per_device_eval_batch_size=16,  #32,
        num_train_epochs=num_epoch,
        gradient_accumulation_steps=4,
        logging_dir='./logs',
        logging_steps=30,
        save_strategy="epoch",
        load_best_model_at_end=True,
        warmup_ratio=0.06,
        lr_scheduler_type="linear",
        fp16=True,
        gradient_checkpointing=False,    # Enable gradient checkpointing
        weight_decay=lambda_,  # Add weight decay here (common value for transformers)
        report_to="none",
    )


class CustomDataCollator(DataCollatorWithPadding):
    def __call__(self, features):
        # Debug prints
        # print("Features received:", features[0] if features else "Empty features")
        
        # Convert single labels to proper format
        if all(isinstance(f, (int, float)) for f in features):
            # print("Converting simple labels to dict format")
            features = [{"label": f} for f in features]
        
        # Check if we only have labels
        if all(isinstance(f, dict) and set(f.keys()) == {"label"} for f in features):
            # print("Only labels found, retrieving from dataset")
            features = [
                {
                    "input_ids": self.tokenizer(
                        train_dataset[i]["sentence" if "sentence" in train_dataset[i] else "text"],
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    )["input_ids"].squeeze(),
                    "attention_mask": self.tokenizer(
                        train_dataset[i]["sentence" if "sentence" in train_dataset[i] else "text"],
                        truncation=True,
                        padding=True,
                        return_tensors="pt"
                    )["attention_mask"].squeeze(),
                    "label": f["label"]
                }
                for i, f in enumerate(features)
            ]

        # Process through parent class
        batch = super().__call__(features)
        
        # Enable gradients
        if isinstance(batch, dict):
            if 'input_ids' in batch:
                batch['input_ids'] = batch['input_ids'].requires_grad_(True)
            if 'attention_mask' in batch:
                batch['attention_mask'] = batch['attention_mask'].requires_grad_(True)
        
        return batch


# Use the custom collator
data_collator = CustomDataCollator(tokenizer)

# Define compute_metrics function for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1) if dataset_name != 'stsb' else logits.squeeze()
    
    if dataset_name == 'stsb':
        # Load glue metric for STS-B
        metric = evaluate.load('glue', 'stsb')    #This metric computes both Pearson and Spearman correlations, which are automatically handled by the GLUE metric for STS-B.
        return metric.compute(predictions=predictions, references=labels)
    else:
        metric = evaluate.load('accuracy')
        return metric.compute(predictions=predictions, references=labels)


#-----------------------------------------------------------------------Adam Optimizer 
from torch.optim import Adam
from transformers import get_scheduler

# Define your custom optimizer (Adam without weight decay)
optimizer_adam = Adam(model.parameters(), lr=training_args.learning_rate)

# Compute total training steps

total_steps = len(train_dataset) // training_args.per_device_train_batch_size
total_steps = total_steps // training_args.gradient_accumulation_steps
total_steps = total_steps * training_args.num_train_epochs

# Apply warmup
warmup_steps = int(training_args.warmup_ratio * total_steps)

# Step 2.3: Create the linear scheduler
scheduler_adam = get_scheduler(
    name="linear",
    optimizer=optimizer_adam,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

#--------------------------------------------------------------------

# For MNLI, we need to handle both matched and mismatched sets
if dataset_name == 'mnli':
    def compute_metrics_mnli(eval_pred):
        matched_preds, mismatched_preds = eval_pred.predictions
        matched_labels, mismatched_labels = eval_pred.label_ids
        
        matched_preds = np.argmax(matched_preds, axis=1)
        mismatched_preds = np.argmax(mismatched_preds, axis=1)
        
        metric = evaluate.load('accuracy')
        matched_acc = metric.compute(predictions=matched_preds, references=matched_labels)['accuracy']
        mismatched_acc = metric.compute(predictions=mismatched_preds, references=mismatched_labels)['accuracy']
        overall_acc = (matched_acc + mismatched_acc) / 2
        
        return {
            'matched_accuracy': matched_acc,
            'mismatched_accuracy': mismatched_acc,
            'overall_accuracy': overall_acc
        }

    # Update the Trainer for MNLI
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_mnli,
        optimizers=(optimizer_adam, scheduler_adam),  # Pass the Adam optimizer
        callbacks=[metrics_callback]
    )
else:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer_adam, scheduler_adam),  # Pass the Adam optimizer
        callbacks=[metrics_callback]
    )


output_dir = folder_name

for ntk_step in range(ntk_iterations):
    torch.cuda.empty_cache()
    gc.collect()
    del input_set_1, input_set_2

    # Ensure x_train and x_test are on the same device as the model
    x_train = x_train.to(model.device)
    x_test = x_test.to(model.device)

    print("LoRA params before training:")
    # print_lora_params(model)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}, {param.data.mean().item()}")


    # Clean model name for file path
    model_name_cleaned = model_name.replace('/', '_')
    
    torch.cuda.empty_cache()
    gc.collect()
    
    start_time_ntk = time.time()
    # Compute NTK for LoRA parameters only before training
    result_from_ntk_vps, ntk_dict, j2_dict_before = empirical_ntk_jacobian_contraction_lora(
        fnet_single, params, buffers, x_train, x_test, chunk_size, compute='full')

    end_time_ntk = time.time()
    # Calculate the time taken
    time_taken_ntk = end_time_ntk - start_time_ntk
    print("Time taken for ntk: {:.2f} seconds".format(time_taken_ntk))


    # Save NTK result before training
    with open(output_dir + f'ntk_lora_train_pre_{model_name_cleaned}_{dataset_name}_layer_{idx_layer}_param_{name_param_file}.pkl', 'wb') as file:
        pickle.dump(result_from_ntk_vps.detach().cpu(), file)

    # Save NTK blocks before training
    with open(output_dir + f'ntk_lora_train_pre_{model_name_cleaned}_{dataset_name}_layer_{idx_layer}_param_{name_param_file}_blocks.pkl', 'wb') as file:
        pickle.dump(ntk_dict, file)
        
    # Save j2 before training
    with open(output_dir + f'j2_lora_train_pre_{model_name_cleaned}_{dataset_name}_layer_{idx_layer}_param_{name_param_file}_epochs_{num_epoch}.pkl', 'wb') as file:
        pickle.dump(j2_dict_before, file) 

    # Train 
    start_time_train = time.time()
    #-------------------------------------------------training lora model
    # Set seed before training
    set_seed(1337)
    trainer.train()
    #-------------------------------------------------
    # print(torch.cuda.memory_summary())
    end_time_train = time.time()
    time_taken_train = end_time_train - start_time_train
    print("Time taken for LoRA fine-tuning: {:.2f} seconds".format(time_taken_train))

    # After your training loop or after trainer.train()
    print("Trainable parameters after training:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.shape}, {param.data.mean().item()}")

    torch.cuda.empty_cache()
    gc.collect()
    

# After applying LoRA
print("Trainable parameters with LoRA:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"Trainable parameter: {name} | Shape: {param.shape} | Number of parameters: {param.numel()}")

# Total trainable parameters
total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters before LoRA: {total_trainable_params1}")
print(f"Total trainable parameters after LoRA: {total_trainable_params}")

torch.cuda.empty_cache()
gc.collect()
#-------------------------------- write the metrics----------------------------------------------
if dataset_name == 'mnli':
    metrics_data = {
        'train_loss': metrics_callback.train_loss_history,
        'eval_loss': metrics_callback.eval_loss_history,
        'eval_matched_accuracy': metrics_callback.eval_metric_history,
        'eval_mismatched_accuracy': metrics_callback.eval_metric_mismatched_history,
        'eval_overall_accuracy': metrics_callback.eval_metric_overall_history  # Assuming overall accuracy is stored here
    }
else:
    metrics_data = {
        'train_loss': metrics_callback.train_loss_history,
        'eval_loss': metrics_callback.eval_loss_history,
        'eval_metric': metrics_callback.eval_metric_history  # This covers all other tasks
    }


# Save the metrics to a pickle file after each NTK iteration
filename = (output_dir + f'metrics_lora_train_{model_name_cleaned}_{dataset_name}_layer_{idx_layer}_param_{name_param_file}.txt')
with open(filename, 'wb') as file:
    pickle.dump(metrics_data, file)

#--------------------------------------------------------------------------------------------------
filename =(output_dir + f'ntk_lora_train_{model_name_cleaned}_{dataset_name}_layer_{idx_layer}_param_{name_param_file}.txt')


# Open the file in write mode
with open(filename, "w") as f:

    f.write(f"Total trainable parameters before LoRA: {total_trainable_params1}\n")
    f.write(f"Total trainable parameters after LoRA: {total_trainable_params}\n")
    f.write(f"Time taken for LoRA fine-tuning: {time_taken_train}\n")
    # f.write(f"Time taken for ntk: {time_taken_ntk}\n\n")
    
    # Write trainable parameters with LoRA
    f.write("Trainable parameters with LoRA:\n")
    num_param_for_ntk=0
    for name, param in model.named_parameters():
        if param.requires_grad:
            f.write(f"Trainable parameter: {name} | Shape: {param.shape} | Number of parameters: {param.numel()}\n")
            if 'lora' in name.lower():
                num_param_for_ntk += 1
    f.write(f"Number of NTK parameters: {num_param_for_ntk}\n\n")    

print('--------------------------------------------------')
print('')
print('metrics data:', metrics_data)


print('')
print('-------------------------------------------------')


trainable_key = "base_model.model.classifier.modules_to_save.default.dense.weight"
original_key = "base_model.model.classifier.original_module.dense.weight"

deviation2 = torch.norm(
    model.state_dict()[trainable_key] - init_state_dict[original_key], p='fro'
).item()

# Calculate the norm of the original weights
original_norm = torch.norm(init_state_dict[original_key], p='fro').item()

# Calculate normalized deviation
normalized_deviation2 = deviation2 / original_norm

print(f'Normalized dense deviation: ||W||_F / ||W_0||_F = {normalized_deviation2:.6f} ({deviation2:.4f} / {original_norm:.4f})' )
print('')


def compute_normalized_lora_update_norm_with_selected_params(peft_model, init_state_dict, selected_params):
    """
    Compute normalized LoRA update norms specifically for the given attention parameters.
    
    Args:
        peft_model: The LoRA model
        init_state_dict: Dictionary containing initial parameters
        selected_params: List of attention parameters to process (e.g., ['key', 'query'])
    
    Returns:
        Tuple of (average_norm, norm_by_module)
    """
    print(f"Computing normalized LoRA updates for attention types: {selected_params}")
    
    # Now continue with normalized norm calculation
    normalized_norms = {}
    total_normalized_norm = 0.0
    count = 0
    
    for name, module in peft_model.named_modules():
        # Find modules with LoRA components
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            # Check if this module corresponds to one of our selected params
            is_selected_param = False
            current_param = None
            
            for param in selected_params:
                if f"attention.self.{param}" in name:
                    is_selected_param = True
                    current_param = param
                    break
            
            if not is_selected_param:
                continue  # Skip modules that don't match our selected parameters
                
            # Calculate LoRA update
            A = module.lora_A['default'].weight
            B = module.lora_B['default'].weight
            delta_W = torch.matmul(B, A)
            lora_norm = torch.norm(delta_W, p='fro').item()
            
            # Extract layer number
            layer_idx = None
            for part in name.split('.'):
                if part.startswith("layer"):
                    parts = name.split(part)[1].split('.')
                    if len(parts) > 1 and parts[1].isdigit():
                        layer_idx = int(parts[1])
                        break
            
            if layer_idx is not None:
                print(f"Processing layer {layer_idx}, attention type: {current_param}")
                
                # Try direct model access first (most reliable approach)
                try:
                    # Access the original weight through the model structure
                    original_weight = getattr(peft_model.base_model.model.roberta.encoder.layer[layer_idx].attention.self, 
                                             current_param).weight
                    original_norm = torch.norm(original_weight, p='fro').item()
                    normalized_norm = lora_norm / original_norm
                    
                    count += 1
                    total_normalized_norm += normalized_norm
                    normalized_norms[name] = normalized_norm
                    
                    print(f"Layer {layer_idx} {current_param}: ||ΔW||_F / ||W_0||_F = {normalized_norm:.6f} ({lora_norm:.4f} / {original_norm:.4f})")
                except (AttributeError, IndexError) as e:
                    # Fallback to finding matching key in init_state_dict
                    matching_key = None
    
    # Calculate average normalized norm
    if count > 0:
        avg_normalized_norm = total_normalized_norm / count
        avg_all_normalized_norm = (total_normalized_norm + normalized_deviation2) / (count + 1)
        print(f"\nAverage normalized LoRA update across {count} {', '.join(selected_params)} components: {avg_normalized_norm:.6f}")
        print(f"Average normalized LoRA update across {count} {', '.join(selected_params)} components + dense: {avg_all_normalized_norm:.6f}")
        return avg_normalized_norm, normalized_norms, avg_all_normalized_norm
    else:
        print(f"No LoRA layers found for attention types: {selected_params}")
        return 0.0, {}, 0.0


avg_normalized_norm, normalized_norms, avg_all_normalized_norm = compute_normalized_lora_update_norm_with_selected_params(model, init_state_dict, selected_params)
print(' ')

#---------------------------------------------------Linearized model

class LinearizedModelTrainer(Trainer):
    """Custom Trainer for linearized models that ensures proper evaluation metrics"""
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Custom loss computation to ensure proper gradient flow"""
        outputs = model(**inputs)
        
        # Get loss from outputs
        loss = outputs.loss if hasattr(outputs, "loss") else outputs[0]
        
        # Log more detailed info about gradients during training
        if self.state.global_step % 20 == 0:
            # Check parameter magnitudes - adapt to your model's structure
            if hasattr(model, 'param_deltas'):
                # For models with direct param_deltas attribute
                param_norm = 0
                for p in model.param_deltas.values():
                    param_norm += p.norm().item() ** 2
                param_norm = param_norm ** 0.5
                print(f"[Linearized] Parameter norm: {param_norm:.6f}")
            elif hasattr(model, 'get_parameter_deltas'):
                # For models with get_parameter_deltas method
                deltas = model.get_parameter_deltas()
                param_norm = 0
                for p in deltas.values():
                    param_norm += p.norm().item() ** 2
                param_norm = param_norm ** 0.5
                print(f"[Linearized] Parameter norm: {param_norm:.6f}")
            else:
                # Fall back to checking all trainable parameters
                param_norm = 0
                for p in model.parameters():
                    if p.requires_grad:
                        param_norm += p.norm().item() ** 2
                param_norm = param_norm ** 0.5
                print(f"[Linearized] Trainable param norm: {param_norm:.6f}")
        
        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(self, *args, **kwargs):
        """Custom evaluation loop to ensure metrics are computed"""
        output = super().evaluation_loop(*args, **kwargs)
        
        # If eval_loss is missing, compute it from the prediction outputs
        if output.metrics and "eval_loss" not in output.metrics:
            # Add a computed eval_loss based on predictions
            if hasattr(output, "predictions") and hasattr(output, "label_ids"):
                if self.args.problem_type == "regression":
                    loss_fct = torch.nn.MSELoss()
                    eval_loss = loss_fct(
                        torch.tensor(output.predictions, device=self.args.device),
                        torch.tensor(output.label_ids, device=self.args.device)
                    ).item()
                else:
                    # Classification
                    loss_fct = torch.nn.CrossEntropyLoss()
                    eval_loss = loss_fct(
                        torch.tensor(output.predictions, device=self.args.device),
                        torch.tensor(output.label_ids, device=self.args.device)
                    ).item()
                
                # Add the computed loss to metrics
                output.metrics["eval_loss"] = eval_loss
                
            # Compute accuracy if not present
            if "eval_accuracy" not in output.metrics and hasattr(output, "predictions"):
                preds = np.argmax(output.predictions, axis=1)
                accuracy = (preds == output.label_ids).mean().item()
                output.metrics["eval_accuracy"] = accuracy
        
        return output



# Training arguments with appropriate metrics
linear_training_args = TrainingArguments(
    output_dir=f"./results_linearized_{model_name}_{dataset_name}",
    eval_strategy="epoch",
    learning_rate=linear_lr,  # Higher learning rate to overcome small gradients
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=num_epoch,
    gradient_accumulation_steps=4,
    logging_dir='./logs_linearized',
    logging_steps=30,
    save_strategy="epoch",
    load_best_model_at_end=True,
    # metric_for_best_model="eval_accuracy",  # Use accuracy instead of loss
    metric_for_best_model="eval_loss",
    # greater_is_better=True,
    warmup_ratio=0.06,
    # max_grad_norm=1.0,  # Add gradient clipping
    lr_scheduler_type="linear",
    fp16=True,
    weight_decay=0.0,
    report_to="none",
)

from torch.optim import AdamW 
optimizer_linear = AdamW([linearized_model.delta_vector], lr=linear_training_args.learning_rate, weight_decay = linear_training_args.weight_decay)

total_steps = len(train_dataset) // linear_training_args.per_device_train_batch_size
total_steps = total_steps // linear_training_args.gradient_accumulation_steps
total_steps = total_steps * linear_training_args.num_train_epochs

# # Apply warmup
warmup_steps = int(linear_training_args.warmup_ratio * total_steps)

# Step 2.3: Create the linear scheduler
scheduler_linear = get_scheduler(
    name="linear",
    optimizer=optimizer_linear,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps,
)

from torch.optim.lr_scheduler import LambdaLR
import math

# Use custom trainer for linearized model
linear_trainer = LinearizedModelTrainer(
    model=linearized_model,
    args=linear_training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer_linear, scheduler_linear), 
    compute_metrics=compute_metrics,  
)

#------------------------------------------------------ Train linearized model
set_seed(1337)
linear_trainer.train()
#------------------------------------------------------ 


import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader


def compare_model_outputs_fast(
    lora_model, 
    linearized_model, 
    validation_dataset, 
    tokenizer, 
    batch_size=16,
    num_examples=None,
    device=None
):
    """
    Fast comparison of LoRA and linearized models with batched processing.
    
    Args:
        lora_model: The fine-tuned LoRA model
        linearized_model: The linearized model
        validation_dataset: Validation dataset
        tokenizer: Tokenizer for the models
        batch_size: Number of examples to process in each batch
        num_examples: Number of examples to evaluate (None for all)
        device: Device to run models on (None for current device)
        
    Returns:
        dict: Dictionary with comparison metrics
    """
    import time
    start_time = time.time()

    set_seed(1337)
    
    # Ensure both models are in evaluation mode
    lora_model.eval()
    linearized_model.eval()
    
    # Set device
    if device is None:
        device = next(lora_model.parameters()).device
    
    # Determine total examples to process
    total_examples = min(len(validation_dataset), num_examples) if num_examples else len(validation_dataset)
    
    # Statistics to track
    total_norm_diff = 0.0
    total_kl_div = 0.0
    total_norm_lora = 0.0  # Track the total norm of LoRA logits
    total_norm_linearized = 0.0  # Track the total norm of linearized logits    
    lora_correct = 0
    linearized_correct = 0
    processed_examples = 0
    check = 0
    
    
    print(f"Comparing models on {total_examples} examples (batch size: {batch_size})...")
    
    # Process in batches
    for start_idx in range(0, total_examples, batch_size):
        # Determine batch end index
        end_idx = min(start_idx + batch_size, total_examples)
        current_batch_size = end_idx - start_idx
        
        # Collect examples for this batch
        batch_texts = []
        batch_labels = []
        
        for i in range(start_idx, end_idx):
            example = validation_dataset[i]
            # Get text field (sentence for CoLA)
            text_field = 'sentence' if 'sentence' in example else 'text'
            batch_texts.append(example[text_field])
            if 'label' in example:
                batch_labels.append(example['label'])
        
        # Tokenize inputs (batch processing)
        inputs = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        # Add labels if available
        if batch_labels:
            inputs['labels'] = torch.tensor(batch_labels)
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Forward passes for both models
        with torch.no_grad():
            # Extract just the inputs needed for inference (no labels)
            model_inputs = {k: v for k, v in inputs.items() 
                          if k in ["input_ids", "attention_mask", "token_type_ids"]}
            
            # Run both models
            lora_outputs = lora_model(**model_inputs)
            linearized_outputs = linearized_model(**model_inputs)
            
            # Get logits
            lora_logits = lora_outputs.logits
            linearized_logits = linearized_outputs.logits
            
            # Get predictions
            lora_preds = torch.argmax(lora_logits, dim=-1)
            linearized_preds = torch.argmax(linearized_logits, dim=-1)
            
            # Compute metrics for each example in the batch
            for i in range(current_batch_size):
                # L2 norm difference
                norm_diff = torch.norm(lora_logits[i] - linearized_logits[i]).item()
                total_norm_diff += norm_diff

                # Convert logits to probabilities
                lora_probs = F.softmax(lora_logits[i], dim=-1)
                linearized_probs = F.softmax(linearized_logits[i], dim=-1)

                # Compute KL divergence: D_KL(lora_probs || linearized_probs)
                # Small epsilon to avoid log(0)
                epsilon = 1e-10
                kl_div = torch.sum(lora_probs * torch.log((lora_probs + epsilon) / (linearized_probs + epsilon)))
                total_kl_div += kl_div
                
                # Norm of lora logits
                norm_lora = torch.norm(lora_logits[i]).item()
                norm_linearized = torch.norm(linearized_logits[i]).item()
                total_norm_lora += norm_lora
                total_norm_linearized += norm_linearized
                check += norm_diff/ norm_lora
                
        
                
                # Check agreement
                agreement_count += (lora_preds[i] == linearized_preds[i]).item()
                
                # Check accuracy if labels available
                if 'labels' in inputs:
                    true_label = inputs['labels'][i].item()
                    lora_correct += (lora_preds[i].item() == true_label)
                    linearized_correct += (linearized_preds[i].item() == true_label)
            
            processed_examples += current_batch_size
        
        # Print progress
        print(f"Processed {processed_examples}/{total_examples} examples...")
    

    # Calculate final metrics including ratio of average norms
    avg_norm_diff = total_norm_diff / processed_examples
    avg_total_kl_div = total_kl_div / processed_examples
    avg_norm_lora = total_norm_lora / processed_examples
    avg_norm_linearized = total_norm_linearized / processed_examples
    
    # Different ways to calculate relative metrics
    norm_ratio = avg_norm_linearized / avg_norm_lora  # Ratio of average norms
    avg_relative_diff = avg_norm_diff / avg_norm_lora  # Relative to LoRA norm

    lora_accuracy = lora_correct / processed_examples if batch_labels else "N/A"
    linearized_accuracy = linearized_correct / processed_examples if batch_labels else "N/A"
    avg_check =  check / processed_examples if batch_labels else "N/A"
    
    # Calculate processing speed
    end_time = time.time()
    processing_time = end_time - start_time
    examples_per_second = processed_examples / processing_time
    
    
    # Print summary
    print("\n===== MODEL OUTPUT COMPARISON =====")
    print(f"Average kl div: {avg_total_kl_div:.4f}")
    print(f"Average L2 norm difference: {avg_norm_diff:.4f}")
    print(f"Average LoRA logits norm: {avg_norm_lora:.4f}")
    print(f"Average linearized logits norm: {avg_norm_linearized:.4f}")
    print(f"Processing time: {processing_time:.2f} seconds ({examples_per_second:.2f} examples/sec)")
    print('-------------------------------------------------------')
    print(f"Relative difference (diff/lora_norm): {avg_relative_diff:.4f}")
    print(f"Norm ratio (linearized/lora): {norm_ratio:.4f}")
    print(f"LoRA model accuracy: {lora_accuracy}")
    print(f"Linearized model accuracy: {linearized_accuracy}")
    print(f"Average Check norm_diff/norm_lora: {avg_check:.4f}")
    

# Load your trained models
model.eval()  # trained LoRA model
linearized_model.eval()  # trained linearized model

# Compare the models with faster processing
comparison_results = compare_model_outputs_fast(
    lora_model=model,
    linearized_model=linearized_model,
    validation_dataset=valid_dataset,
    tokenizer=tokenizer,
    batch_size=32,  # Larger batch size for faster processing
    num_examples=None,  # Set to None to process all examples
    device=device
)

print('Regularization Value:', lambda_ )
