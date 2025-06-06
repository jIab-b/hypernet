"""
Main script for the Per-Prompt Optimized Text-to-LoRA System.
Orchestrates the workflow from text input to LoRA weight generation.
"""
import os
import argparse
import sys # For sys.exit()

# Import module functions
from text_processing import preprocess_text
from config_manager import load_model_configuration
from lora_proposer import propose_initial_lora_parameters
from lora_assembler import assemble_lora_matrices
from optimizer_engine import optimize_lora_parameters
from output_manager import save_lora_weights

def main():
    """
    Main execution function.
    """
    parser = argparse.ArgumentParser(description="Per-Prompt Optimized Text-to-LoRA System")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt to generate LoRA for."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the target diffusion model (e.g., 'sdxl_lightning'). "
             "A corresponding '<model_name>_config.json' must exist in 'model_configs/'."
    )
    # Add an optional argument for the output directory/filename later if needed
    parser.add_argument("--output_path", type=str, default="output/generated_lora.safetensors", help="Path to save the generated LoRA weights.")

    args = parser.parse_args()

    print("Starting Per-Prompt Optimized Text-to-LoRA System...")
    text_prompt = args.prompt
    model_name_arg = args.model_name
    print(f"Input prompt: \"{text_prompt}\"")
    print(f"Target model name: \"{model_name_arg}\"")

    # 2. Determine and load target model configuration
    config_filename = f"{model_name_arg}_config.json"
    config_path = os.path.join("model_configs", config_filename)

    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found for model '{model_name_arg}'.")
        print(f"Looked for: {config_path}")
        print("Please ensure a config file named '<model_name>_config.json' exists in the 'model_configs' directory.")
        # List available configs
        available_configs = [f for f in os.listdir("model_configs") if f.endswith("_config.json")]
        if available_configs:
            print("\nAvailable model configurations (based on filenames):")
            for ac_file in available_configs:
                print(f"  - {ac_file.replace('_config.json', '')}")
        else:
            print("No configuration files found in 'model_configs'.")
        sys.exit(1)

    try:
        model_config = load_model_configuration(config_path)
    except Exception as e:
        print(f"Failed to load model configuration from '{config_path}': {e}")
        return

    # 3. Preprocess text input
    # preprocess_text now returns a dictionary including tokenized outputs and original_prompt
    try:
        processed_prompt_data = preprocess_text(text_prompt, model_config)
    except Exception as e:
        print(f"Failed during text preprocessing: {e}")
        return
    # print(f"Prompt tokenized.") # Already printed in function

    # 4. Propose initial LoRA parameters
    # propose_initial_lora_parameters now expects processed_prompt_data
    try:
        initial_lora_params = propose_initial_lora_parameters(processed_prompt_data, model_config)
    except Exception as e:
        print(f"Failed during LoRA parameter proposal: {e}")
        return
    # print("Initial LoRA parameters proposed.") # Already printed

    # 5. Assemble LoRA matrices
    # This now handles torch.Tensors
    try:
        candidate_lora_matrices = assemble_lora_matrices(initial_lora_params, model_config)
    except Exception as e:
        print(f"Failed during LoRA matrix assembly: {e}")
        return
    # print("LoRA matrices assembled.") # Already printed

    # 6. Optimize LoRA parameters
    # optimize_lora_parameters now expects processed_prompt_data
    try:
        optimized_lora_matrices = optimize_lora_parameters(
            candidate_lora_matrices, # Start with the initially proposed/assembled ones
            processed_prompt_data,
            model_config
        )
    except Exception as e:
        print(f"Failed during LoRA optimization: {e}")
        # If optimization fails, we might not have meaningful weights to save.
        # Depending on the error, we could try to save initial_lora_params or skip.
        # For now, just return.
        return
    # print("LoRA parameters optimized.") # Already printed

    # 7. Output/Save LoRA weights
    # Use the output_path argument provided by the user
    full_output_path = args.output_path
    
    # Ensure the directory for the output path exists (saver also does this, but good practice here too)
    output_dir_from_arg = os.path.dirname(full_output_path)
    if output_dir_from_arg and not os.path.exists(output_dir_from_arg):
        os.makedirs(output_dir_from_arg)
        print(f"Created output directory: {output_dir_from_arg}")
    
    metadata_to_save = {
        "prompt": text_prompt,
        "base_model_name": model_config.get("model_name"),
        "lora_rank": str(model_config.get("lora_rank")), # Ensure metadata is string for safetensors
        "lora_alpha": str(model_config.get("lora_alpha")),
        "description": "LoRA generated by Per-Prompt Optimized Text-to-LoRA System for SDXL Lightning"
    }
    try:
        # save_lora_weights now expects model_config as the third argument
        save_lora_weights(optimized_lora_matrices, full_output_path, model_config, metadata=metadata_to_save)
    except Exception as e:
        print(f"Failed to save LoRA weights: {e}")
        return
    # print(f"Optimized LoRA weights saved to {full_output_path}") # Already printed in function

    print("System finished.")

if __name__ == "__main__":
    main()