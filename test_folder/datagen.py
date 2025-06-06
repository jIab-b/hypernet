import argparse
import torch
import os
import json
from PIL import Image
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
# Removed UNet2DConditionModel, hf_hub_download, load_file as they are for specific UNet loading like Lightning
# from diffusers.utils import load_image

def generate_images(pipeline, prompts: list[str], output_dir: str,
                    lora_filename: str | None, lora_default_search_path: str,
                    batch_size: int = 1):
    """
    Generates images for each prompt.
    If LoRA is specified, generates only LoRA-applied images.
    Otherwise, generates only base model images.
    Saves images numbered 1 to num_prompts.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Ensured output directory exists: {output_dir}")

    lora_name_no_ext = os.path.splitext(lora_filename)[0] if lora_filename else None
    actual_lora_path = None

    if lora_filename:
        if os.path.isabs(lora_filename):
            actual_lora_path = lora_filename
        else:
            actual_lora_path = os.path.join(lora_default_search_path, lora_filename)
        
        if not os.path.exists(actual_lora_path):
            print(f"Error: LoRA file not found at {actual_lora_path}. Cannot generate LoRA images.")
            # Depending on desired behavior, you could exit or fall back to base.
            # For this request, if LoRA is specified but not found, we should probably error out or do nothing for LoRA.
            # The request implies if --lora is specified, ONLY lora images. So if not found, it's an issue.
            return # Exit if LoRA specified but not found


    total_prompts = len(prompts)
    for i in range(0, total_prompts, batch_size):
        batch_prompts = prompts[i:i + batch_size]
        current_batch_actual_size = len(batch_prompts)

        if not batch_prompts:
            continue

        if actual_lora_path and lora_name_no_ext:
            # --- Generate LoRA images ONLY ---
            print(f"Loading LoRA: {actual_lora_path} for prompts {i+1} to {i+current_batch_actual_size}")
            try:
                pipeline.load_lora_weights(actual_lora_path)
                print(f"Generating LoRA images with '{lora_name_no_ext}'...")
                with torch.inference_mode():
                    generated_images = pipeline(
                        prompt=batch_prompts,
                        num_inference_steps=30,
                        guidance_scale=7.0,
                        width=1024,
                        height=1024
                    ).images
                
                for batch_idx, image in enumerate(generated_images):
                    original_prompt_idx = i + batch_idx
                    current_prompt_text = prompts[original_prompt_idx]

                    image_filename = f"image_{original_prompt_idx + 1:04d}-{lora_name_no_ext}.png"
                    prompt_filename = f"image_{original_prompt_idx + 1:04d}-{lora_name_no_ext}.txt"
                    image_path = os.path.join(output_dir, image_filename)
                    prompt_path = os.path.join(output_dir, prompt_filename)

                    try:
                        image.save(image_path)
                        with open(prompt_path, "w", encoding="utf-8") as f:
                            f.write(current_prompt_text)
                        print(f"Saved LoRA image: {image_path} and prompt: {prompt_path}")
                    except Exception as e:
                        print(f"Error saving LoRA image or prompt {image_filename}: {e}")
            except Exception as e:
                print(f"Error during LoRA loading or generation for {lora_filename}: {e}")
            finally:
                # Ensure LoRA weights are unloaded even if an error occurs during generation
                if hasattr(pipeline, 'unload_lora_weights'): # Check if method exists
                    print(f"Unloading LoRA weights for {lora_name_no_ext}")
                    pipeline.unload_lora_weights()
        else:
            # --- Generate BASE images ONLY ---
            print(f"Generating BASE images for prompts {i+1} to {i+current_batch_actual_size}...")
            with torch.inference_mode():
                generated_images = pipeline(
                    prompt=batch_prompts,
                    num_inference_steps=30,
                    guidance_scale=7.0,
                    width=1024,
                    height=1024
                ).images

            for batch_idx, image in enumerate(generated_images):
                original_prompt_idx = i + batch_idx
                current_prompt_text = prompts[original_prompt_idx]
                
                image_filename = f"image_{original_prompt_idx + 1:04d}.png"
                prompt_filename = f"image_{original_prompt_idx + 1:04d}.txt"
                image_path = os.path.join(output_dir, image_filename)
                prompt_path = os.path.join(output_dir, prompt_filename)

                try:
                    image.save(image_path)
                    with open(prompt_path, "w", encoding="utf-8") as f:
                        f.write(current_prompt_text)
                    print(f"Saved base image: {image_path} and prompt: {prompt_path}")
                except Exception as e:
                    print(f"Error saving base image or prompt {image_filename}: {e}")

    print(f"Finished generating images in {output_dir}.")

def load_prompts_from_jsonl(file_path: str) -> list[str]:
    """Loads prompts from a .jsonl file."""
    prompts = []
    if not os.path.exists(file_path):
        print(f"Error: Prompts file not found at {file_path}")
        return prompts
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if "prompt" in data:
                        prompts.append(data["prompt"])
                    else:
                        print(f"Warning: Line in {file_path} does not contain 'prompt' key: {line.strip()}")
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from line in {file_path}: {line.strip()}")
        print(f"Loaded {len(prompts)} prompts from {file_path}")
    except Exception as e:
        print(f"Error reading or parsing {file_path}: {e}")
    return prompts

def main():
    parser = argparse.ArgumentParser(description="Generate images from prompts using SDXL Base and optionally a LoRA.")
    parser.add_argument("--prompts_file", type=str, required=True, help="Path to a .jsonl file containing prompts.")
    parser.add_argument("--output_base_dir", type=str, default="generated_images", help="Base directory for output. A subfolder named after the prompts file will be created here.")
    parser.add_argument("--lora", type=str, default=None, help="Optional: Filename of the LoRA (e.g., 'my_lora.safetensors'). Default search path is '../output/' relative to this script.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to run on ('cuda', 'cpu').")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for image generation.")
    
    args = parser.parse_args()

    # Determine LoRA default search path
    # os.path.dirname(__file__) gives the directory of the current script (test_folder)
    # Then go up one level ('..') and into 'output'
    lora_default_search_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    print(f"Default LoRA search path: {lora_default_search_path}")
    if args.lora:
        print(f"LoRA specified: {args.lora}")


    # Determine specific output directory based on prompts_file name
    prompts_file_basename = os.path.basename(args.prompts_file)
    prompts_set_name = os.path.splitext(prompts_file_basename)[0]
    specific_output_dir = os.path.join(args.output_base_dir, prompts_set_name)
    
    print(f"Loading SDXL Base model on device: {args.device}")
    try:
        base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        torch_dtype = torch.float16 if args.device == "cuda" else torch.float32

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            base_model_id,
            torch_dtype=torch_dtype,
            variant="fp16" if args.device == "cuda" else None, # fp16 for CUDA
            use_safetensors=True
        )
        
        if args.device == "cuda":
            # Enable sequential CPU offloading for potentially lower VRAM usage on CUDA during inference
            print("Enabling sequential CPU offload for the pipeline...")
            pipeline.enable_sequential_cpu_offload()
        else:
            pipeline = pipeline.to(args.device)

        # Use a standard scheduler for SDXL Base
        pipeline.scheduler = EulerDiscreteScheduler.from_config(pipeline.scheduler.config)
        
        print("SDXL Base model loaded successfully.")

    except Exception as e:
        print(f"Error loading SDXL Base model: {e}")
        return

    prompts = load_prompts_from_jsonl(args.prompts_file)
    if not prompts:
        print(f"No prompts loaded from {args.prompts_file}. Exiting.")
        return

    print(f"\n--- Generating images from {args.prompts_file} into {specific_output_dir} ---")
    generate_images(
        pipeline,
        prompts,
        specific_output_dir,
        args.lora,
        lora_default_search_path,
        args.batch_size
    )
    
    print("\nImage generation process complete.")

if __name__ == "__main__":
    main()