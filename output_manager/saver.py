"""
Saves the generated LoRA weights using safetensors.
"""
from typing import Dict, Any
import torch
import os
from safetensors.torch import save_file

def save_lora_weights(
    lora_weights: Dict[str, Dict[str, torch.Tensor]], # Expects PyTorch tensors
    output_path: str,
    model_config: Dict[str, Any], # To help determine prefixes for PEFT compatibility
    metadata: Dict[str, Any] = None
):
    """
    Saves the LoRA weights to a file in .safetensors format.

    Args:
        lora_weights: A dictionary where keys are original target layer names
                      (e.g., "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"),
                      and values are dictionaries with 'lora_A' and 'lora_B'
                      LoRA matrices (PyTorch tensors).
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
    
    # Get all target layer lists from config
    unet_layers = model_config.get("target_layers_unet", [])
    te1_layers = model_config.get("target_layers_text_encoder", [])
    te2_layers = model_config.get("target_layers_text_encoder_2", [])

    for layer_name, matrices in lora_weights.items():
        # Determine the prefix based on which list the layer_name is in
        prefix = "lora_unet_"
        if layer_name in te1_layers:
            prefix = "lora_te1_" # Assuming text_encoder is TE1
        elif layer_name in te2_layers:
            prefix = "lora_te2_" # Assuming text_encoder_2 is TE2
        
        # Ensure tensors are on CPU before saving
        lora_A_cpu = matrices["lora_A"].cpu()
        lora_B_cpu = matrices["lora_B"].cpu()

        # Construct names compatible with what `pipeline.load_lora_weights` might expect
        # or how PEFT saves them. This often involves replacing '.' with '_' in the prefix part.
        # The actual layer name part should retain its dots.
        # Example: lora_unet_down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.weight
        # The `optimizer` extracts weights with `original_key` which is like `down_blocks...`
        
        # Standard PEFT naming convention for LoRA layers:
        # For a layer `module_name` (e.g., `transformer_blocks.0.attn1.to_q`)
        # LoRA A weight: `base_model.model.{module_name}.lora_A.default.weight`
        # LoRA B weight: `base_model.model.{module_name}.lora_B.default.weight`
        # When saving standalone LoRA, often the `base_model.model.` is stripped,
        # and a component prefix (unet, te1, te2) is added.
        
        # Let's use a common convention for standalone LoRA files loaded by diffusers:
        # {component_prefix}{original_layer_name_with_dots}.lora_A.weight
        # {component_prefix}{original_layer_name_with_dots}.lora_B.weight
        
        # The `layer_name` from `lora_weights` is the `original_layer_name_with_dots`
        tensors_to_save[f"{prefix}{layer_name}.lora_A.weight"] = lora_A_cpu
        tensors_to_save[f"{prefix}{layer_name}.lora_B.weight"] = lora_B_cpu
        
        # Also save alpha if present in config (though it's often part of model loading)
        # safetensors metadata is good for this.
        # lora_alpha = model_config.get("lora_alpha")
        # if lora_alpha is not None:
        #     tensors_to_save[f"{prefix}{layer_name}.alpha"] = torch.tensor(lora_alpha, dtype=lora_A_cpu.dtype)


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
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q": { # UNet layer
            "lora_A": torch.randn(8, 320, dtype=torch.float32, device=device_),
            "lora_B": torch.zeros(320, 8, dtype=torch.float32, device=device_)
        },
        "text_model.encoder.layers.22.self_attn.q_proj": { # Text Encoder 1 layer
            "lora_A": torch.randn(8, 768, dtype=torch.float32, device=device_),
            "lora_B": torch.zeros(768, 8, dtype=torch.float32, device=device_)
        }
    }
    sample_metadata_ = {
        "prompt": "a cat astronaut on the moon",
        "base_model_name": "stabilityai/sdxl-lightning",
        "lora_rank": 8,
        "lora_alpha": 8,
        "description": "Test LoRA generated by per-prompt optimizer"
    }
    
    # Mock model_config for determining prefixes
    mock_config_for_saver = {
        "target_layers_unet": ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"],
        "target_layers_text_encoder": ["text_model.encoder.layers.22.self_attn.q_proj"],
        "target_layers_text_encoder_2": []
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