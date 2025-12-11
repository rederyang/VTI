
import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
import math

# import kornia
from transformers import set_seed

import random
from .pca import PCA
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import List, Tuple

def process_image(image_processor, image_raw):
    answer = image_processor(image_raw)

    # Check if the result is a dictionary and contains 'pixel_values' key
    if 'pixel_values' in answer:
        answer = answer['pixel_values'][0]
    
    # Convert numpy array to torch tensor if necessary
    if isinstance(answer, np.ndarray):
        answer = torch.from_numpy(answer)
    
    # If it's already a tensor, return it directly
    elif isinstance(answer, torch.Tensor):
        return answer
    
    else:
        raise ValueError("Unexpected output format from image_processor.")
    
    return answer

def mask_patches(tensor, indices, patch_size=14):
    """
    Creates a new tensor where specified patches are set to the mean of the original tensor.
    
    Args:
    tensor (torch.Tensor): Input tensor of shape (C, H, W)
    indices (list of int): Indices of the patches to modify
    patch_size (int): Size of one side of the square patch
    
    Returns:
    torch.Tensor: New tensor with modified patches
    """
    # Clone the original tensor to avoid modifying it
    new_tensor = tensor.clone()

    # Calculate the mean across the spatial dimensions
    mean_values = tensor.mean(dim=(1, 2), keepdim=True)
    
    # Number of patches along the width
    patches_per_row = tensor.shape[2] // patch_size
    total_patches = (tensor.shape[1] // patch_size) * (tensor.shape[2] // patch_size)


    for index in indices:
        # Calculate row and column position of the patch
        row = index // patches_per_row
        col = index % patches_per_row

        # Calculate the starting pixel positions
        start_x = col * patch_size
        start_y = row * patch_size

        # Replace the patch with the mean values
        new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size] = mean_values.expand(-1, patch_size, patch_size)#new_tensor[:, start_y:start_y + patch_size, start_x:start_x + patch_size].mean(dim=(1, 2), keepdim=True).expand(-1, patch_size, patch_size)# mean_values.expand(-1, patch_size, patch_size)

    return new_tensor


def get_prompts(args, model, tokenizer, data_demos, question, model_is_llaval=True):
    if model_is_llaval:
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
        qs_pos = question
        qs_neg = question

        if hasattr(model.config, 'mm_use_im_start_end'):

            if model.config.mm_use_im_start_end:
                qs_pos = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_pos
            else:
                qs_pos = DEFAULT_IMAGE_TOKEN + '\n' + qs_pos

            if model.config.mm_use_im_start_end:
                qs_neg = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs_neg
            else:
                qs_neg = DEFAULT_IMAGE_TOKEN + '\n' + qs_neg

            conv_pos = conv_templates[args.conv_mode].copy()
            conv_pos.append_message(conv_pos.roles[0], qs_pos)
            conv_pos.append_message(conv_pos.roles[1], None)
            conv_neg = conv_templates[args.conv_mode].copy()
            conv_neg.append_message(conv_neg.roles[0], qs_neg)
            conv_neg.append_message(conv_neg.roles[1], None)


            prompts_positive  = [conv_pos.get_prompt() + k['value'] for k in data_demos]
            prompts_negative  = [conv_neg.get_prompt() + k['h_value'] for k in data_demos]

            input_ids_positive = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_positive]
            input_ids_negative = [tokenizer_image_token(p, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda() for p in prompts_negative]

        else:
            from transformers import InstructBlipProcessor
            processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

            input_ids_positive = []
            input_ids_negative = []

            for k in data_demos:
                image_path = os.path.join(args.data_file, 'train2014', k['image'])

                image_raw = Image.open(image_path).convert("RGB")
                input_ids_positive.append(processor(images=image_raw, text=question + k['value'], return_tensors="pt").to(model.device))
                input_ids_negative.append(processor(images=image_raw, text=question + k['h_value'], return_tensors="pt").to(model.device))

        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    else:

        prompts_positive = []
        prompts_negative = []

        for k in data_demos:
            image_path = os.path.join(args.data_file, 'train2014', k['image'])    
            prompts_positive.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['value']}]))
            prompts_negative.append(tokenizer.from_list_format([{'image': image_path},{'text':question + k['h_value']}]))

        input_ids_positive = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_positive]
        input_ids_negative = [tokenizer(p, return_tensors='pt').to(model.device) for p in prompts_negative]
        inputs = [(input_ids_negative[demo_id], input_ids_positive[demo_id]) for demo_id in range(len(input_ids_negative))]
        inputs = tuple(inputs)
    return inputs

def get_demos(args, image_processor, model, tokenizer, patch_size = 14, file_path = './experiments/data/hallucination_vti_demos.jsonl', model_is_llaval=True): 
    # Initialize a list to store the JSON objects
    data = []

    # Open the file and read line by line
    with open(file_path, 'r') as file:
        for line in file:
            # Each line is a complete JSON object
            json_object = json.loads(line.strip())
            data.append(json_object)
    data_demos = random.sample(data, args.num_demos)

    inputs_images = []
    for i in range(len(data_demos)):
        question = data_demos[i]['question']
        image_path = os.path.join(args.data_file, 'train2014', data_demos[i]['image'])
        image_raw = Image.open(image_path).convert("RGB")
        image_tensor = process_image(image_processor, image_raw)
        image_tensor_cd_all_trials = []

        for t in range(args.num_trials):
            token_numbers = image_tensor.shape[-1]*image_tensor.shape[-2]/patch_size**2
            mask_index = torch.randperm(int(token_numbers))[:int(args.mask_ratio * token_numbers)]
            image_tensor_cd = mask_patches(image_tensor, mask_index, patch_size=patch_size)
                
            image_tensor_cd_all_trials.append(image_tensor_cd)

        inputs_images.append([image_tensor_cd_all_trials, image_tensor])

    input_ids = get_prompts(args, model, tokenizer, data_demos, question, model_is_llaval=model_is_llaval)
    
    return inputs_images, input_ids

# Textual VTI 相关

def get_hiddenstates(model, inputs, image_tensor):
        h_all = []
        with torch.no_grad():
            for example_id in range(len(inputs)):
                embeddings_for_all_styles= []
                for style_id in range(len(inputs[example_id])):
                    if image_tensor is None:
                        h = model(
                                **inputs[example_id][style_id],
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                    else:
                        h = model(
                                inputs[example_id][style_id],
                                images=image_tensor[example_id][-1].unsqueeze(0).half(),
                                use_cache=False,
                                output_hidden_states=True,
                                return_dict=True).hidden_states

                    embedding_token = []
                    for layer in range(len(h)):
                        embedding_token.append(h[layer][:,-1].detach().cpu())
                    
                    embedding_token = torch.cat(embedding_token, dim=0).cpu().clone()
                    embeddings_for_all_styles.append(embedding_token)
                h_all.append(tuple(embeddings_for_all_styles))
        return h_all

def obtain_textual_vti(model, inputs, image_tensor, rank=1):
    hidden_states = get_hiddenstates(model, inputs, image_tensor)
    hidden_states_all = []
    num_demonstration = len(hidden_states)
    neg_all = []
    pos_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1].view(-1) - hidden_states[demonstration_id][0].view(-1)
        hidden_states_all.append(h)
        neg_all.append(hidden_states[demonstration_id][0].view(-1))
        pos_all.append(hidden_states[demonstration_id][1].view(-1))
    fit_data = torch.stack(hidden_states_all)  # (n_demos, n_layers * feat_dim)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    eval_data =  pca.transform(fit_data.float())
    h_pca = pca.inverse_transform(eval_data) 

    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))#h_pca.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    reading_direction = fit_data.mean(0).view(hidden_states[demonstration_id][0].size(0), hidden_states[demonstration_id][0].size(1))
    return direction, reading_direction

def obtain_textual_vti_fix(model, inputs, image_tensor, rank=1):
    print("Using fix version of textual direction")
    hidden_states = get_hiddenstates(model, inputs, image_tensor)

    n_layers, feat_dim = hidden_states[0][0].shape

    hidden_states_all = []
    num_demonstration = len(hidden_states)
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][1] - hidden_states[demonstration_id][0]
        hidden_states_all.append(h)
    fit_data = torch.stack(hidden_states_all, dim=1)  # (n_layers, n_demos, feat_dim)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())

    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, feat_dim)
    reading_direction = fit_data.mean(1).view(n_layers, feat_dim)
    return direction, reading_direction

# Visual VTI 相关

def average_tuples(tuples: List[Tuple[torch.Tensor]]) -> Tuple[torch.Tensor]:
    # Check that the input list is not empty
    if not tuples:
        raise ValueError("The input list of tuples is empty.")

    # Check that all tuples have the same length
    n = len(tuples[0])
    if not all(len(t) == n for t in tuples):
        raise ValueError("All tuples must have the same length.")

    # Initialize a list to store the averaged tensors
    averaged_tensors = []

    # Iterate over the indices of the tuples
    for i in range(n):
        # Stack the tensors at the current index and compute the average
        tensors_at_i = torch.stack([t[i].detach().cpu() for t in tuples])
        averaged_tensor = tensors_at_i.mean(dim=0)
        averaged_tensors.append(averaged_tensor)

    # Convert the list of averaged tensors to a tuple
    averaged_tuple = tuple(averaged_tensors)

    return averaged_tuple

def get_visual_hiddenstates(model, image_tensor, model_is_llaval=True):
    h_all = []
    with torch.no_grad():
        if model_is_llaval:
            try:
                vision_model = model.model.vision_tower.vision_tower.vision_model
            except:
                vision_model = model.vision_model
        else:
            vision_model = model.transformer.visual
            model.transformer.visual.output_hidden_states = True
            
        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles= []
            for style_id in range(len(image_tensor[example_id])):
                if isinstance(image_tensor[example_id][style_id], list):
                    h = []
                    for image_tensor_ in image_tensor[example_id][style_id]:
                        if model_is_llaval:
                            h_ = vision_model(
                                image_tensor_.unsqueeze(0).half().cuda(),
                                output_hidden_states=True,
                                return_dict=True).hidden_states
                        else:
                            _, h_ = vision_model(
                                image_tensor_.unsqueeze(0).cuda())
                        h.append(h_)
                    h = average_tuples(h)
                else:
                    if model_is_llaval:
                        h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).cuda(),
                            output_hidden_states=True,
                            return_dict=True).hidden_states

                    else:
                        _, h = vision_model(
                            image_tensor[example_id][style_id].unsqueeze(0).cuda())
                
                embedding_token = []
                for layer in range(len(h)):
                    embedding_token.append(h[layer][:,:].detach().cpu())
                embedding_token = torch.cat(embedding_token, dim=0)
                embeddings_for_all_styles.append(embedding_token)
            h_all.append(tuple(embeddings_for_all_styles))
        if not model_is_llaval:
            model.transformer.visual.output_hidden_states = False

    del h, embedding_token

    return h_all

def obtain_visual_vti(model, image_tensor, rank=1, model_is_llaval=True):

    hidden_states = get_visual_hiddenstates(model, image_tensor, model_is_llaval = model_is_llaval)
    n_layers, n_tokens, feat_dim = hidden_states[0][0].shape
    num_demonstration = len(hidden_states)

    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][0].reshape(n_tokens,-1) - hidden_states[demonstration_id][1].reshape(n_tokens,-1)
        hidden_states_all.append(h)

    fit_data = torch.stack(hidden_states_all,dim=1)[:]  # (n_tokens, n_demos, n_layers * feat_dim)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)
    return direction, reading_direction

def obtain_visual_vti_fix(model, image_tensor, rank=1, model_is_llaval=True):
    print("Using fix version of visual direction")
    hidden_states = get_visual_hiddenstates(model, image_tensor, model_is_llaval = model_is_llaval)
    n_layers, n_tokens, feat_dim = hidden_states[0][0].shape
    num_demonstration = len(hidden_states)

    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        h = hidden_states[demonstration_id][0] - hidden_states[demonstration_id][1]
        hidden_states_all.append(h)  # (n_layers, n_tokens, feat_dim)

    fit_data = torch.stack(hidden_states_all,dim=2)[:]  # (n_layers, n_tokens, n_demos, feat_dim)
    fit_data = fit_data.reshape(n_layers * n_tokens, num_demonstration, feat_dim)
    pca = PCA(n_components=rank).to(fit_data.device).fit(fit_data.float())
    direction = (pca.components_.sum(dim=1,keepdim=True) + pca.mean_).mean(1).view(n_layers, n_tokens, -1)
    reading_direction = fit_data.mean(1).view(n_layers, n_tokens, -1)
    return direction, reading_direction

# Head-wise Attention VTI 相关

def get_attention_hiddenstates(model, image_tensor):
    """
    Extract per-head attention outputs from vision encoder.
    Uses forward hook on out_proj to capture the input (which is the per-head attention output).
    
    Args:
        model: LLaVA model
        image_tensor: list of [masked_images_list, original_image] pairs
        
    Returns:
        h_all: list of tuples, each tuple contains (masked_attn_outputs, original_attn_outputs)
               where each output is tensor of shape (n_layers, n_heads, n_tokens, head_dim)
    """
    try:
        vision_model = model.model.vision_tower.vision_tower.vision_model
    except:
        vision_model = model.vision_model
    
    # Get model config
    encoder_layers = vision_model.encoder.layers
    n_layers = len(encoder_layers)
    n_heads = encoder_layers[0].self_attn.num_heads
    head_dim = encoder_layers[0].self_attn.head_dim
    
    h_all = []
    
    with torch.no_grad():
        for example_id in range(len(image_tensor)):
            embeddings_for_all_styles = []
            
            for style_id in range(len(image_tensor[example_id])):
                # Storage for per-head attention outputs from each layer
                attn_outputs_per_layer = []
                hooks = []
                
                def make_hook(storage):
                    def hook_fn(module, args, kwargs, output):
                        # CLIPAttention output: (attn_output, attn_weights) or just attn_output
                        # attn_output shape: (batch, seq_len, hidden_dim)
                        # We need to reshape to (batch, seq_len, n_heads, head_dim)
                        if isinstance(output, tuple):
                            attn_output = output[0]
                        else:
                            attn_output = output
                        batch, seq_len, hidden_dim = attn_output.shape
                        # Reshape to per-head format
                        attn_per_head = attn_output.view(batch, seq_len, n_heads, head_dim)
                        storage.append(attn_per_head.detach().cpu())
                    return hook_fn
                
                # Register hooks on each layer's self_attn
                for layer in encoder_layers:
                    storage = []
                    attn_outputs_per_layer.append(storage)
                    hook = layer.self_attn.register_forward_hook(make_hook(storage), with_kwargs=True)
                    hooks.append(hook)
                
                # Handle masked images (multiple trials) or single image
                if isinstance(image_tensor[example_id][style_id], list):
                    # Multiple masked images - average their outputs
                    all_trials_outputs = []
                    for image_tensor_ in image_tensor[example_id][style_id]:
                        # Clear storage for new forward pass
                        for storage in attn_outputs_per_layer:
                            storage.clear()
                        
                        _ = vision_model(
                            image_tensor_.unsqueeze(0).half().cuda(),
                            output_hidden_states=True,
                            return_dict=True
                        )
                        
                        # Collect outputs from this trial: (n_layers, n_heads, n_tokens, head_dim)
                        trial_output = torch.stack([s[0].squeeze(0).transpose(0, 1) for s in attn_outputs_per_layer], dim=0)
                        all_trials_outputs.append(trial_output)
                    
                    # Average across trials
                    embedding_token = torch.stack(all_trials_outputs, dim=0).mean(dim=0)
                else:
                    # Single image
                    for storage in attn_outputs_per_layer:
                        storage.clear()
                    
                    _ = vision_model(
                        image_tensor[example_id][style_id].unsqueeze(0).half().cuda(),
                        output_hidden_states=True,
                        return_dict=True
                    )
                    
                    # Collect: (n_layers, n_heads, n_tokens, head_dim)
                    embedding_token = torch.stack([s[0].squeeze(0).transpose(0, 1) for s in attn_outputs_per_layer], dim=0)
                
                # Remove hooks
                for hook in hooks:
                    hook.remove()
                
                embeddings_for_all_styles.append(embedding_token)
            
            h_all.append(tuple(embeddings_for_all_styles))
    
    return h_all

def obtain_attention_vti_per_head(model, image_tensor, rank=1):
    """
    Compute per-head VTI directions from attention outputs.
    
    Args:
        model: LLaVA model
        image_tensor: list of [masked_images_list, original_image] pairs
        rank: number of PCA components
        
    Returns:
        direction: tensor of shape (n_layers, n_heads, n_tokens, head_dim)
        reading_direction: tensor of shape (n_layers, n_heads, n_tokens, head_dim)
    """
    hidden_states = get_attention_hiddenstates(model, image_tensor)
    # hidden_states[i][j] has shape (n_layers, n_heads, n_tokens, head_dim)
    
    n_layers, n_heads, n_tokens, head_dim = hidden_states[0][0].shape
    num_demonstration = len(hidden_states)
    
    # Compute difference: masked - original for each demo
    hidden_states_all = []
    for demonstration_id in range(num_demonstration):
        # h: (n_layers, n_heads, n_tokens, head_dim)
        h = hidden_states[demonstration_id][0] - hidden_states[demonstration_id][1]
        hidden_states_all.append(h)
    
    # Stack demos: (n_demos, n_layers, n_heads, n_tokens, head_dim)
    fit_data = torch.stack(hidden_states_all, dim=0)
    
    # Reshape for per-head PCA: (n_layers, n_heads, n_tokens, n_demos, head_dim)
    fit_data = fit_data.permute(1, 2, 3, 0, 4)
    
    # Apply PCA per layer (with n_heads * n_tokens as batch dimension)
    # Output shape: (n_layers, n_heads, n_tokens, head_dim)
    directions = []
    reading_directions = []
    
    for layer_idx in range(n_layers):
        # data: (n_heads, n_tokens, n_demos, head_dim)
        # Reshape to (n_heads * n_tokens, n_demos, head_dim) for batch PCA
        data = fit_data[layer_idx].reshape(n_heads * n_tokens, num_demonstration, head_dim)
        
        # PCA: batch=n_heads*n_tokens, n_samples=n_demos, features=head_dim
        pca = PCA(n_components=rank).to(data.device).fit(data.float())
        
        # pca.components_: (n_heads * n_tokens, rank, head_dim)
        # pca.mean_: (n_heads * n_tokens, 1, head_dim)
        direction_layer = (pca.components_.sum(dim=1, keepdim=True) + pca.mean_).squeeze(1)
        # direction_layer: (n_heads * n_tokens, head_dim)
        
        # Reshape back to (n_heads, n_tokens, head_dim)
        direction_layer = direction_layer.view(n_heads, n_tokens, head_dim)
        
        # Reading direction: mean across demos
        reading_direction_layer = data.mean(dim=1).view(n_heads, n_tokens, head_dim)
        
        directions.append(direction_layer)
        reading_directions.append(reading_direction_layer)
    
    direction = torch.stack(directions, dim=0)  # (n_layers, n_heads, n_tokens, head_dim)
    reading_direction = torch.stack(reading_directions, dim=0)
    
    return direction, reading_direction
