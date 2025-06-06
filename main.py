"""
Main script for the Per-Prompt Optimized Text-to-LoRA System.
Orchestrates the workflow from text input to LoRA weight generation.
"""
import os # For path joining

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
    print("Starting Per-Prompt Optimized Text-to-LoRA System...")

    # 1. Get text prompt
    text_prompt = "A hyper-realistic cat astronaut exploring a vibrant nebula"
    print(f"Input prompt: {text_prompt}")

    # 2. Load target model configuration
    # Construct path relative to main.py's location or use absolute paths
    # For simplicity, assuming model_configs is at the same level as main.py's parent if main.py is in a src dir
    # Or directly if main.py is at the root. Given current structure, it's at root.
    config_path = "model_configs/example_diffusion_config.json"
    try:
        model_config = load_model_configuration(config_path)
    except Exception as e:
        print(f"Failed to load model configuration: {e}")
        return

    # 3. Preprocess text input
    tokenized_prompt = preprocess_text(text_prompt, model_config)
    # print(f"Prompt tokenized: {tokenized_prompt}") # Already printed in function

    # 4. Propose initial LoRA parameters
    initial_lora_params = propose_initial_lora_parameters(tokenized_prompt, model_config)
    # print("Initial LoRA parameters proposed.") # Already printed

    # 5. Assemble LoRA matrices
    # In our current setup, this is a pass-through but good to keep for the flow
    candidate_lora_matrices = assemble_lora_matrices(initial_lora_params, model_config)
    # print("LoRA matrices assembled.") # Already printed

    # 6. Optimize LoRA parameters
    optimized_lora_matrices = optimize_lora_parameters(
        candidate_lora_matrices, # Start with the initially proposed/assembled ones
        tokenized_prompt,
        model_config
    )
    # print("LoRA parameters optimized.") # Already printed

    # 7. Output/Save LoRA weights
    output_filename = "generated_lora.npz" # Using .npz due to placeholder saver
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    full_output_path = os.path.join(output_dir, output_filename)
    
    metadata_to_save = {
        "prompt": text_prompt,
        "base_model_name": model_config.get("model_name"),
        "lora_rank": model_config.get("lora_rank")
    }
    save_lora_weights(optimized_lora_matrices, full_output_path, metadata=metadata_to_save)
    # print(f"Optimized LoRA weights saved to {full_output_path}") # Already printed

    print("System finished.")

if __name__ == "__main__":
    main()