import argparse
import os
import glob
from pathlib import Path

import torch
from PIL import Image, ImageFilter
from diffusers import (
    StableDiffusionInpaintPipeline, 
    UNet2DConditionModel,
    DDPMScheduler
)
from transformers import CLIPTextModel


parser = argparse.ArgumentParser(description="Batch RealFill Inference for Multi-Stage Processing")
parser.add_argument(
    "--model_path",
    type=str,
    required=True,
    help="Path to pretrained model or model identifier from huggingface.co/models.",
)
parser.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Base output directory containing stage{i} folders (e.g., /logs/output_dir/)",
)
parser.add_argument(
    "--prompt",
    type=str,
    default="a photo of man",
    help="The prompt for the image",
)

parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible inference.")

args = parser.parse_args()

def overlay_mask_on_target(mask_path, target_path):
    """
    Overlays a mask on the target image, turning masked areas white.

    Args:
        mask_path (str): Path to the mask image file.
        target_path (str): Path to the target image file.
    """
    # Open the mask and target images
    mask = Image.open(mask_path).convert("L")  # Convert mask to grayscale
    target = Image.open(target_path).convert("RGB")  # Ensure target is in RGB

    # Create a white image of the same size
    white_image = Image.new("RGB", target.size, "white")

    # Composite the images: where mask is white, use white_image; otherwise, use target
    result = Image.composite(white_image, target, mask)

    # Save the result
    result.save(target_path)

def process_image_pair(pipe, target_image_path, mask_image_path, output_path, prompt, generator=None):
    """Process a single image-mask pair through the RealFill pipeline."""
    
    # Create a temporary copy of the target image for preprocessing
    temp_target_path = str(target_image_path).replace('.png', '_temp.png')
    
    # Copy target image to temp location
    target_img = Image.open(target_image_path)
    target_img.save(temp_target_path)
    
    # Preprocess: overlay mask on target to ensure compatibility
    overlay_mask_on_target(mask_image_path, temp_target_path)
    
    # Load preprocessed image and mask
    image = Image.open(temp_target_path)
    mask_image = Image.open(mask_image_path)
    
    # Ensure images have the same size and proper modes
    image = image.convert("RGB")
    mask_image = mask_image.convert("L")  # Convert mask to grayscale
    
    if image.size != mask_image.size:
        print(f"Resizing mask from {mask_image.size} to {image.size}")
        mask_image = mask_image.resize(image.size, Image.Resampling.LANCZOS)

    # Apply filters to mask
    erode_kernel = ImageFilter.MaxFilter(3)
    mask_image = mask_image.filter(erode_kernel)

    blur_kernel = ImageFilter.BoxBlur(1)
    mask_image = mask_image.filter(blur_kernel)
    
    # Run inference
    result = pipe(
        prompt=prompt, image=image, mask_image=mask_image, 
        num_inference_steps=200, guidance_scale=1, generator=generator, 
    ).images[0]
    
    # Ensure all images have the same size and proper modes for compositing
    result = result.convert("RGB")
    image = image.convert("RGB")
    mask_image = mask_image.convert("L")
    
    # Resize result to match image size if needed
    if result.size != image.size:
        print(f"Resizing result from {result.size} to {image.size}")
        result = result.resize(image.size, Image.Resampling.LANCZOS)
    
    # Resize mask to match image size if needed
    if mask_image.size != image.size:
        print(f"Resizing mask from {mask_image.size} to {image.size}")
        mask_image = mask_image.resize(image.size, Image.Resampling.LANCZOS)
    
    result = Image.composite(result, image, mask_image)
    result.save(output_path)
    
    # Clean up temporary file
    os.remove(temp_target_path)
    
    print(f"Processed: {os.path.basename(target_image_path)} -> {os.path.basename(output_path)}")


def find_stage_directories(base_dir):
    """Find all stage{i} directories in the base directory."""
    stage_dirs = []
    base_path = Path(base_dir)
    
    # Look for stage{i} directories
    for path in base_path.iterdir():
        if path.is_dir() and path.name.startswith('stage'):
            stage_dirs.append(path)
    
    # Sort by stage number
    stage_dirs.sort(key=lambda x: int(x.name.replace('stage', '')))
    return stage_dirs


def find_matching_image_pairs(stage_dir):
    """Find matching mask and optim_1499 image pairs in a stage directory."""
    mask_dir = stage_dir / 'mask'
    optim_dir = stage_dir / 'optim_1499'
    
    if not mask_dir.exists() or not optim_dir.exists():
        print(f"Warning: Missing mask or optim_1499 directory in {stage_dir}")
        return []
    
    # Get all mask images
    mask_files = list(mask_dir.glob('*.png'))
    pairs = []
    
    for mask_file in mask_files:
        # Look for corresponding optim image with same filename
        optim_file = optim_dir / mask_file.name
        if optim_file.exists():
            pairs.append((optim_file, mask_file))
        else:
            print(f"Warning: No matching optim image for {mask_file.name}")
    
    return pairs


if __name__ == "__main__":
    generator = None 

    # create & load model
    print("Loading RealFill model...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-inpainting",
        torch_dtype=torch.float32,
        revision=None
    )

    pipe.unet = UNet2DConditionModel.from_pretrained(
        args.model_path, subfolder="unet", revision=None,
    )
    pipe.text_encoder = CLIPTextModel.from_pretrained(
        args.model_path, subfolder="text_encoder", revision=None,
    )
    pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    if args.seed is not None:
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
    
    print("Model loaded successfully!")
    
    # Find all stage directories
    stage_dirs = find_stage_directories(args.output_dir)
    print(f"Found {len(stage_dirs)} stage directories: {[d.name for d in stage_dirs]}")
    
    total_processed = 0
    
    for stage_dir in stage_dirs:
        print(f"\nProcessing {stage_dir.name}...")
        
        # Create output directory for this stage
        realfill_out_dir = stage_dir / 'realfill_out'
        realfill_out_dir.mkdir(exist_ok=True)
        
        # Find matching image pairs
        image_pairs = find_matching_image_pairs(stage_dir)
        print(f"Found {len(image_pairs)} image pairs in {stage_dir.name}")
        
        for target_image_path, mask_image_path in image_pairs:
            # Generate output filename (same as original)
            output_filename = target_image_path.name
            output_path = realfill_out_dir / output_filename
            
            process_image_pair(
                pipe, target_image_path, mask_image_path, 
                output_path, args.prompt, generator
            )
            total_processed += 1
             
    
    print(f"\nBatch processing complete! Processed {total_processed} images total.")
    
    del pipe
    torch.cuda.empty_cache()
