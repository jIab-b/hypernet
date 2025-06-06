"""
Saves the generated LoRA weights.
"""
from typing import Dict, Any
import numpy as np
import os
# from safetensors.numpy import save_file # Would be used for actual safetensors saving

def save_lora_weights(
    lora_weights: Dict[str, Dict[str, np.ndarray]],
    output_path: str,
    metadata: Dict[str, Any] = None
):
    """
    Saves the LoRA weights to a file.
    Currently, this is a placeholder. A real implementation would likely
    save in a standard format like .safetensors.

    Args:
        lora_weights: A dictionary where keys are target layer names,
                      and values are dictionaries with 'lora_A' and 'lora_B'
                      LoRA matrices (numpy arrays).
        output_path: The path to save the LoRA weights file.
        metadata: Optional metadata to include in the saved file (e.g.,
                  prompt, model_config details).
    """
    print(f"Saving LoRA weights to: {output_path}")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Placeholder for .safetensors saving
    # For safetensors, tensors need to be prepared correctly.
    # The structure would be something like:
    # tensors_to_save = {}
    # for layer_name, matrices in lora_weights.items():
    #     # Naming convention for safetensors often includes "lora_down" for A
    #     # and "lora_up" for B, and might include layer path.
    #     # Example: "lora_unet_down_blocks_0_attentions_0_proj_in.lora_down.weight"
    #     # For simplicity, we'll use a flatter structure here for the placeholder.
    #     tensors_to_save[f"{layer_name}.lora_A"] = matrices["lora_A"]
    #     tensors_to_save[f"{layer_name}.lora_B"] = matrices["lora_B"]
    #
    # if metadata:
    #     # safetensors save_file can take a metadata dict
    #     # save_file(tensors_to_save, output_path, metadata=metadata)
    # else:
    #     # save_file(tensors_to_save, output_path)
    
    # For now, let's just save as a .npz file as a placeholder
    # to demonstrate saving the numpy arrays.
    
    # Prepare for npz: flatten the keys for lora_A and lora_B
    npz_dict = {}
    for layer_name, matrices in lora_weights.items():
        safe_layer_name = layer_name.replace('.', '_') # Make key safe for npz
        npz_dict[f"{safe_layer_name}_lora_A"] = matrices["lora_A"]
        npz_dict[f"{safe_layer_name}_lora_B"] = matrices["lora_B"]

    if metadata: # np.savez doesn't directly support metadata like safetensors
        print("Metadata (not saved in this npz placeholder):", metadata)

    try:
        np.savez(output_path, **npz_dict)
        print(f"LoRA weights saved successfully as .npz (placeholder) to {output_path}")
    except Exception as e:
        print(f"Error saving LoRA weights to {output_path}: {e}")
        raise

if __name__ == '__main__':
    # Example Usage
    sample_lora_weights = {
        "attn.to_q": {
            "lora_A": np.random.randn(4, 320).astype(np.float32),
            "lora_B": np.zeros((768, 4), dtype=np.float32)
        },
        "attn.to_k": {
            "lora_A": np.random.randn(4, 320).astype(np.float32),
            "lora_B": np.zeros((768, 4), dtype=np.float32)
        }
    }
    sample_metadata = {
        "prompt": "a cat astronaut",
        "base_model": "GenericDiffusionModel-v1.0",
        "lora_rank": 4
    }
    
    # Create an 'output' directory if it doesn't exist for the example
    if not os.path.exists("output"):
        os.makedirs("output")
        
    save_lora_weights(sample_lora_weights, "output/example_lora.npz", sample_metadata)
    
    # Verify save (optional)
    try:
        loaded_data = np.load("output/example_lora.npz")
        print("\nVerification: Loaded keys from .npz file:")
        for key in loaded_data.keys():
            print(f"  - {key}, shape: {loaded_data[key].shape}")
    except Exception as e:
        print(f"Error during verification load: {e}")