"""
Saves the generated LoRA weights using safetensors.
"""
from typing import Dict, Any
import torch
import os
from safetensors.torch import save_file

def save_lora_weights(
    lora_weights: Dict[str, Dict[str, Dict[str, torch.Tensor]]], # component -> layer -> {lora_A/B -> tensor}
    output_path: str,
    model_config: Dict[str, Any], # To help determine prefixes for PEFT compatibility
    metadata: Dict[str, Any] = None
):
    """
    Saves the LoRA weights to a file in .safetensors format.

    Args:
        lora_weights: A dictionary structured as:
                      component_name -> { layer_name -> {"lora_A": tensor, "lora_B": tensor} }
        output_path: The path to save the LoRA weights file (e.g., "output/lora.safetensors").
        model_config: The main configuration, used to find which component (UNet, TE1, TE2)
                      a layer belongs to, for PEFT-compatible naming.
        metadata: Optional metadata to include in the saved file.
    """
    print(f"Saving LoRA weights to: {output_path} using safetensors")

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    tensors_to_save = {}
    # Determine component prefixes for PEFT compatibility
    # This is a simplified way; PEFT's actual naming can be more complex.
    # Diffusers `load_lora_weights` often expects keys like:
    # "lora_unet_down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight"
    # "lora_te1_text_model.encoder.layers.22.self_attn.q_proj.lora_A.weight"
    
    component_prefix_map = {
        "unet": "lora_unet_",
        "text_encoder_1": "lora_te1_",
        "text_encoder_2": "lora_te2_"
    }

    for component_name, component_layers in lora_weights.items():
        prefix = component_prefix_map.get(component_name)
        if prefix is None:
            print(f"Warning: Unknown component name '{component_name}'. Using default prefix 'lora_generic_'.")
            prefix = f"lora_{component_name.lower()}_" # Fallback prefix

        for layer_name, matrices in component_layers.items():
            if "lora_A" not in matrices or "lora_B" not in matrices:
                print(f"Warning: Missing 'lora_A' or 'lora_B' for layer {component_name}/{layer_name}. Skipping.")
                continue

            # Ensure tensors are on CPU before saving
            lora_A_cpu = matrices["lora_A"].cpu()
            lora_B_cpu = matrices["lora_B"].cpu()

            # Construct names compatible with what `pipeline.load_lora_weights` might expect
            # The `layer_name` is now the generic key like "q_proj" or a specific UNet path.
            # PEFT typically saves LoRA weights with keys that include the full path to the adapted module.
            # For diffusers, the convention is often:
            # {prefix_for_component}{original_module_path_dots_preserved}.lora_A.weight
            # Example: "lora_unet_down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight"
            # Example: "lora_te1_text_model.encoder.layers.0.self_attn.q_proj.lora_A.weight" (if PEFT targets this full name)
            #
            # However, our `target_modules` in LoraConfig are now generic (e.g., "q_proj").
            # PEFT will create LoRA layers for all modules whose names contain "q_proj".
            # The state_dict of the PEFT model will have keys like:
            # "base_model.model.text_model.encoder.layers.0.self_attn.q_proj.lora_A.default.weight"
            # The `optimizer_engine` extracts these and stores them using `original_key` which is "q_proj".
            # This means `layer_name` here is "q_proj".
            # This saving scheme might need to be more aligned with how PEFT names things if we want
            # to load these weights back into a PEFT model directly.
            # For now, we'll save with the generic name, which might be okay for some loading mechanisms
            # or if the loading side also knows to map generic names.
            # A common convention for diffusers `load_lora_weights` is to have the full path.
            # This part is tricky and depends on the exact loading mechanism.
            # The current `optimizer_engine` saves `original_key` which is the generic name for text encoders.
            # So `layer_name` here will be "q_proj", "k_proj" etc. for text encoders.
            # For UNet, `layer_name` is the full path like "down_blocks...to_q".

            # Let's stick to the {prefix}{layer_name}.lora_{A/B}.weight format.
            # This means for text encoders, keys will be like "lora_te1_q_proj.lora_A.weight".
            # This is simpler but might not be directly loadable by all tools without remapping.
            tensors_to_save[f"{prefix}{layer_name}.lora_A.weight"] = lora_A_cpu
            tensors_to_save[f"{prefix}{layer_name}.lora_B.weight"] = lora_B_cpu


    # Ensure metadata values are strings for safetensors
    final_metadata = {}
    if metadata:
        for k, v in metadata.items():
            final_metadata[k] = str(v) # Convert all metadata values to strings

    try:
        save_file(tensors_to_save, output_path, metadata=final_metadata if final_metadata else None)
        print(f"LoRA weights saved successfully to {output_path}")
    except Exception as e:
        print(f"Error saving LoRA weights with safetensors to {output_path}: {e}")
        print("Ensure you have the `safetensors` library installed.")
        raise

if __name__ == '__main__':
    # Example Usage
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_lora_weights_ = {
        "UNet": {
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q": {
                "lora_A": torch.randn(8, 320, dtype=torch.float32, device=device_),
                "lora_B": torch.zeros(320, 8, dtype=torch.float32, device=device_)
            }
        },
        "TextEncoder1": {
            "q_proj": { # Generic name as per optimizer output
                "lora_A": torch.randn(8, 768, dtype=torch.float32, device=device_),
                "lora_B": torch.zeros(768, 8, dtype=torch.float32, device=device_)
            },
            "k_proj": {
                "lora_A": torch.randn(8, 768, dtype=torch.float32, device=device_),
                "lora_B": torch.zeros(768, 8, dtype=torch.float32, device=device_)
            }
        }
    }
    sample_metadata_ = {
        "prompt": "a cat astronaut on the moon",
        "base_model_name": "stabilityai/sdxl-lightning",
        "lora_rank": 8,
        "lora_alpha": 8,
        "description": "Test LoRA generated by per-prompt optimizer"
    }
    
    # Mock model_config (model_config is not strictly used by the saver for prefixing anymore with this change,
    # but kept for function signature and potential future use)
    mock_config_for_saver = {
        "model_name": "stabilityai/sdxl-lightning", # Example
        # Other config items that might be relevant for metadata or validation if added later
    }

    output_dir_ = "output"
    if not os.path.exists(output_dir_):
        os.makedirs(output_dir_)
    output_file_path = os.path.join(output_dir_, "example_sdxl_lora.safetensors")
        
    save_lora_weights(sample_lora_weights_, output_file_path, mock_config_for_saver, sample_metadata_)
    
    # To verify, you would typically load it, e.g., with:
    # from safetensors.torch import load_file
    # loaded_tensors = load_file(output_file_path)
    # print("\nVerification: Loaded tensors from .safetensors file:")
    # for key, tensor in loaded_tensors.items():
    #     print(f"  - {key}, shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"\nExample LoRA weights saved to {output_file_path}")
    print("To verify, load with `safetensors.torch.load_file`.")