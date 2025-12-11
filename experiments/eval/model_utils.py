import argparse
import time
import torch
import json
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
from transformers import set_seed

from vti_utils.utils import get_demos, obtain_textual_vti, obtain_visual_vti, obtain_textual_vti_fix, obtain_visual_vti_fix
from vti_utils.llm_layers import add_vti_layers, remove_vti_layers

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
parser.add_argument("--model-base", type=str, default=None)
parser.add_argument("--image-folder", type=str, default="")
parser.add_argument("--data-file", type=str, default="/workspace/data/COCO2014/")
parser.add_argument("--conv-mode", type=str, default="llava_v1")
parser.add_argument("--num_demos", type=int, default=50)
parser.add_argument("--alpha_image", type=float, default=0.9)
parser.add_argument("--alpha_text", type=float, default=0.9)
parser.add_argument("--num_beams", type=int, default=5)
parser.add_argument("--sample", action='store_true')

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--mask_ratio", type=float, default=0.99)
parser.add_argument("--num_trials", type=int, default=50)

parser.add_argument("--visual_direction_path", type=str, default="/workspace/VTI/runs/vti_fix/d50t50/visual_direction.pt")
parser.add_argument("--textual_direction_path", type=str, default="/workspace/VTI/runs/vti_fix/d50t50/textual_direction.pt")

# Wether use fix version of VTI
parser.add_argument("--use_fix_vti", action='store_true',
                    help="Use fix version of VTI")

# Competitor intervention arguments
parser.add_argument("--competitor", type=str, default=None,
                    choices=['clipping', 'smoothing', 'quantization'],
                    help="Competitor robustness method to apply on vision features")
parser.add_argument("--competitor_position", type=str, default="all",
                    choices=['all', 'last', 'encoder'],
                    help="Position of the competitor intervention")
# clipping
parser.add_argument("--clip_percentile", type=float, default=95,
                    help="Percentile for dynamic clipping (e.g., 95 means clip to 5-95%%)")
parser.add_argument("--clip_mode", type=str, default="per-channel",
                    choices=['per-channel', 'per-token', 'global'],
                    help="Clipping mode for dynamic clipping")
# smoothing
parser.add_argument("--smooth_kernel", type=int, default=3,
                    help="Kernel size for spatial smoothing (odd number)")
parser.add_argument("--grid_size", type=int, default=24,
                    help="Grid size for spatial smoothing")
# quantization
parser.add_argument("--quant_scale", type=float, default=10.0,
                    help="Scale factor for feature quantization")

# Head-wise attention VTI arguments
parser.add_argument("--head_wise_alpha_image", type=float, default=0.0,
                    help="Alpha for head-wise attention VTI on image")

# Demo image file path
parser.add_argument("--demo_image_file_path", type=str, default="/workspace/VTI/experiments/data/hallucination_vti_demos.jsonl")
args, _ = parser.parse_known_args([])


def load_model():
    # Model
    global args
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Get demo images
    input_images, input_ids = get_demos(args, image_processor, model, tokenizer, file_path=args.demo_image_file_path)

    # Get 'good' and 'bad' latent states and calculate the steering vectors
    print('Obtaining direction\n')

    torch.cuda.empty_cache()

    if args.alpha_image != 0:
        if os.path.exists(args.visual_direction_path):
            print(f"Loading visual direction from {args.visual_direction_path}")
            visual_direction = torch.load(args.visual_direction_path)
        else:
            print(f"Computing visual direction")
            if args.use_fix_vti:
                obtain_visual = obtain_visual_vti_fix
            else:
                obtain_visual = obtain_visual_vti
            time_start = time.time()
            vti_vision, _ = obtain_visual(
                model, input_images, rank=1
                )  # shape: (n_layers, n_tokens, feat_dim)
            time_end = time.time()
            print(f"Time taken: {time_end - time_start} seconds")
            visual_direction = vti_vision[1:]  # skip the input token embeddings
            print(f"Saving visual direction to {args.visual_direction_path}")
            torch.save(visual_direction, args.visual_direction_path)

        print("visual_direction shape: ", visual_direction.shape)
        print(f"Adding visual direction to the model")
        add_vti_layers(model.model.vision_tower.vision_tower.vision_model, torch.stack([visual_direction],dim=1).cuda(), alpha = [args.alpha_image])
    
    if args.alpha_text != 0:
        if os.path.exists(args.textual_direction_path):
            print(f"Loading textual direction from {args.textual_direction_path}")
            textual_direction = torch.load(args.textual_direction_path)
        else:
            print(f"Computing textual direction")
            if args.use_fix_vti: 
                obtain_textual = obtain_textual_vti_fix
            else:
                obtain_textual = obtain_textual_vti
            time_start = time.time()
            vti_text, _ = obtain_textual(
                model, input_ids, input_images, rank=1
                )
            time_end = time.time()
            print(f"Time taken: {time_end - time_start} seconds")
            textual_direction = vti_text[1:]
            print(f"Saving textual direction to {args.textual_direction_path}")
            torch.save(textual_direction, args.textual_direction_path)

        print("textual_direction shape: ", textual_direction.shape)
        print(f"Adding textual direction to the model")
        add_vti_layers(model, torch.stack([textual_direction],dim=1).cuda(), alpha = [args.alpha_text])

    # Register competitor intervention hook
    competitor_hook = None
    if args.competitor:
        from vti_utils.competitors import CompetitorIntervention
        
        kwargs = {}
        if args.competitor == 'clipping':
            kwargs = {'percentile': args.clip_percentile, 'mode': args.clip_mode}
        elif args.competitor == 'smoothing':
            kwargs = {'kernel_size': args.smooth_kernel, 'grid_size': args.grid_size}
        elif args.competitor == 'quantization':
            kwargs = {'scale': args.quant_scale}
        
        competitor_hook = CompetitorIntervention(args.competitor, **kwargs)
        if args.competitor_position == 'all':  # register on all layers
            for layer in model.model.vision_tower.vision_tower.vision_model.encoder.layers:
                competitor_hook.register(layer)
        elif args.competitor_position == 'last':  # register on the last layer
            competitor_hook.register(model.model.vision_tower.vision_tower.vision_model.encoder.layers[-1])
        elif args.competitor_position == 'encoder':  # register on the encoder layer
            competitor_hook.register(model.model.vision_tower)
        else:
            raise ValueError(f"Invalid competitor position: {args.competitor_position}")

    # Register head-wise attention VTI hook
    head_wise_attention_vti_hook = None
    if args.head_wise_alpha_image != 0:
        from vti_utils.utils import obtain_attention_vti_per_head
        from vti_utils.llm_layers import HeadWiseAttentionVTI

        vti_directions, _ = obtain_attention_vti_per_head(model, input_images, rank=1)
        head_wise_attention_vti_hook = HeadWiseAttentionVTI(vti_directions, lam=args.head_wise_alpha_image)
        head_wise_attention_vti_hook.register(model)

    # Store hooks and args on model for cleanup
    model._vti_hooks = [competitor_hook, head_wise_attention_vti_hook]
    model._vti_args = args
    
    return model, tokenizer, image_processor, model_name


def cleanup_model(model):
    # Cleanup interventions
    _args = getattr(model, '_vti_args', None)
    hooks = getattr(model, '_vti_hooks', [])
    if _args is None:
        return
    if _args.alpha_image != 0:
        remove_vti_layers(model.model.vision_tower.vision_tower.vision_model)
    if _args.alpha_text != 0:
        remove_vti_layers(model)
    for hook in hooks:
        if hook is not None:
            hook.remove()
