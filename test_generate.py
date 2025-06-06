#!/usr/bin/env python
"""
Generates images using a base SDXL model and the same model with a specified LoRA.
"""
import argparse
import os
import random
import string
import torch
from diffusers import StableDiffusionXLPipeline
from PIL import Image

def generate_random_word(length=8):
    """Generates a random alphanumeric string of a given length."""
    letters_and_digits = string.ascii_lowercase + string.digits
    return ''.join(random.choice(letters_and_digits) for _ in range(length))

def main():
    parser = argparse.ArgumentParser(description="Generate images with and without a specified LoRA.")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt for image generation.")
    parser.add_argument("--lora", type=str, required=True, help="Name of the LoRA file (e.g., 'generated_sdxl_lora') located in the 'output' directory. Assumed .safetensors format.")
    parser.add_argument("--model_id", type=str, default="stabilityai/stable-diffusion-xl-base-1.0", help="The base model ID from Hugging Face Hub.")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for generation for reproducibility.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "test_out"
    os.makedirs(output_dir, exist_ok=True)

    random_suffix = generate_random_word()
    
    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=device).manual_seed(args.seed)

    # --- Load Base Pipeline ---
    print(f"Loading base pipeline: {args.model_id}...")
    try:
        pipeline = StableDiffusionXLPipeline.from_pretrained(
            args.model_id,
            torch_dtype=torch.float16,
            variant="fp16", # Common variant for SDXL base
            use_safetensors=True
        )
        #pipeline.to(device)
        #if device.type == "cuda": # Offloading is typically for GPU memory saving
            #print("Enabling sequential CPU offload for the pipeline...")
        pipeline.enable_sequential_cpu_offload()
        print("Base pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading base pipeline: {e}")
        return

    # --- Generate Base Image ---
    print(f"\nGenerating base image for prompt: '{args.prompt}'")
    try:
        image_base = pipeline(
            prompt=args.prompt,
            num_inference_steps=args.num_inference_steps,
            generator=generator
        ).images[0]
        
        base_image_filename = f"test_img_base_{random_suffix}.png"
        base_image_path = os.path.join(output_dir, base_image_filename)
        image_base.save(base_image_path)
        print(f"Base image saved to {base_image_path}")
    except Exception as e:
        print(f"Error generating base image: {e}")
        # Continue to try LoRA image if base fails, or decide to return

    # --- Load LoRA and Generate LoRA Image ---
    lora_file_name = f"{args.lora}.safetensors"
    lora_full_path = os.path.join("output", lora_file_name)

    if not os.path.exists(lora_full_path):
        print(f"\nError: LoRA file not found at {lora_full_path}")
        print("Skipping LoRA image generation.")
    else:
        print(f"\nLoading LoRA weights from {lora_full_path}...")
        try:
            # load_lora_weights modifies the pipeline in place.
            # It loads weights into UNet and text encoders based on standard prefixes.
            pipeline.load_lora_weights(lora_full_path)
            print("LoRA weights loaded successfully.")

            print(f"Generating LoRA image for prompt: '{args.prompt}'")
            # Reset seed for LoRA image if you want it to be different from base,
            # or use the same generator for direct comparison of LoRA effect.
            # For this test, using the same generator (if seed provided) is good for comparison.
            image_lora = pipeline(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                generator=generator
            ).images[0]
            
            lora_image_filename = f"test_img_lora_{random_suffix}.png"
            lora_image_path = os.path.join(output_dir, lora_image_filename)
            image_lora.save(lora_image_path)
            print(f"LoRA image saved to {lora_image_path}")

            # Unload LoRA weights if the pipeline instance is to be reused for other tasks without LoRA
            # pipeline.unload_lora_weights() 
            # print("LoRA weights unloaded.")

        except Exception as e:
            print(f"Error during LoRA image generation: {e}")

    print("\nDone.")

if __name__ == "__main__":
    main()