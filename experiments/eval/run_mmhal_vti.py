import argparse
import time
import torch
import json
from tqdm import tqdm
import requests
from io import BytesIO
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from PIL import Image
from transformers import set_seed

from vti_utils.utils import get_demos, obtain_textual_vti, obtain_visual_vti
from vti_utils.llm_layers import add_vti_layers, remove_vti_layers

from datasets import load_dataset

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def eval_model(args):
    # Model
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    # Get demo images
    input_images, input_ids = get_demos(args, image_processor, model, tokenizer)

    # Get 'good' and 'bad' latent states and calculate the steering vectors
    print('Obtaining direction\n')

    torch.cuda.empty_cache()

    if args.alpha_image != 0:
        if os.path.exists(args.visual_direction_path):
            print(f"Loading visual direction from {args.visual_direction_path}")
            visual_direction = torch.load(args.visual_direction_path)
        else:
            print(f"Computing visual direction")
            time_start = time.time()
            vti_vision, _ = obtain_visual_vti(
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
            time_start = time.time()
            vti_text, _ = obtain_textual_vti(
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
            kwargs = {'percentile': args.clip_percentile}
        elif args.competitor == 'smoothing':
            kwargs = {'kernel_size': args.smooth_kernel, 'grid_size': 24}
        elif args.competitor == 'quantization':
            kwargs = {'scale': args.quant_scale}
        
        competitor_hook = CompetitorIntervention(args.competitor, **kwargs)
        competitor_hook.register(model.model.vision_tower)

    # Run MMHal benchmark
    print(f"Running MMHal benchmark\n")

    torch.cuda.empty_cache()

    # Create answers file
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    # Load dataset
    dataset = load_dataset("Shengcao1006/MMHal-Bench")['test']

    ans_file = open(answers_file, "w")
    for img_id in tqdm(range(len(dataset))):
        # prepare image
        image_path = dataset[img_id]['image_path']
        raw_image = load_image(image_path)

        image_tensor = image_processor.preprocess(raw_image, return_tensors='pt')['pixel_values'][0]

        # prepare text
        qs = dataset[img_id]['question']
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().cuda(),
                num_beams=5,
                max_new_tokens=256,
                do_sample=args.sample,
                use_cache=False)

        outputs = tokenizer.batch_decode(output_ids[:, :], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        img_save = {}
        img_save["question_type"] = dataset[img_id]["question_type"]
        img_save["question_topic"] = dataset[img_id]["question_topic"]
        img_save["image_id"] = dataset[img_id]["image_id"]
        img_save["image_src"] = dataset[img_id]["image_src"]
        img_save["image_content"] = dataset[img_id]["image_content"]
        img_save["question"] = dataset[img_id]["question"]
        img_save["gt_answer"] = dataset[img_id]["gt_answer"]
        img_save["model_answer"] = outputs

        ans_file.write(json.dumps(img_save) + "\n")
        ans_file.flush()
    ans_file.close()

    # Cleanup interventions
    if args.alpha_image != 0:
        remove_vti_layers(model.model.vision_tower.vision_tower.vision_model)
    if args.alpha_text != 0:
        remove_vti_layers(model)
    if competitor_hook:
        competitor_hook.remove()

    print(f"Finished evaluation, results saved to {answers_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="liuhaotian/llava-v1.5-7b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="/data/datasets/MSCOCO/val2014")
    parser.add_argument("--answers-file", type=str, default="/results/coco_pope_popular_answer.jsonl")
    parser.add_argument("--data-file", type=str, default="/data/datasets/MSCOCO/")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num_demos", type=int, default=50)
    parser.add_argument("--alpha_image", type=float, default=0)
    parser.add_argument("--alpha_text", type=float, default=0.8)
    parser.add_argument("--num_beams", type=int, default=5)
    parser.add_argument("--sample", action='store_true')

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask_ratio", type=float, default=0.99)
    parser.add_argument("--num_trials", type=int, default=50)

    parser.add_argument("--visual_direction_path", type=str, required=True)
    parser.add_argument("--textual_direction_path", type=str, required=True)

    # Competitor intervention arguments
    parser.add_argument("--competitor", type=str, default=None,
                        choices=['clipping', 'smoothing', 'quantization'],
                        help="Competitor robustness method to apply on vision features")
    parser.add_argument("--clip_percentile", type=float, default=95,
                        help="Percentile for dynamic clipping (e.g., 95 means clip to 5-95%%)")
    parser.add_argument("--smooth_kernel", type=int, default=3,
                        help="Kernel size for spatial smoothing (odd number)")
    parser.add_argument("--quant_scale", type=float, default=10.0,
                        help="Scale factor for feature quantization")
    
    args = parser.parse_args()
    set_seed(args.seed)
    eval_model(args)
