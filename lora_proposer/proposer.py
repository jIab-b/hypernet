"""
Proposes initial LoRA parameters using random initialization.
"""
from typing import Dict, Any
import torch

# Note: LLM-based initialization (GPT-2) and related functions
# (get_proposer_llm_and_tokenizer, generate_llm_prompt_for_lora)
# have been removed to simplify the codebase as the LLM output was not being used.
# Initialization is now purely random for LoRA A and zeros for LoRA B.

def propose_initial_lora_parameters(
    processed_prompt_data: Dict[str, Any], # Output from preprocess_text
    model_config: Dict[str, Any]
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Proposes initial LoRA parameters (A and B matrices) for each target layer
    using random initialization for LoRA A and zeros for LoRA B.
    Outputs PyTorch tensors.
    """
    original_prompt = processed_prompt_data["original_prompt"]
    print(f"Proposing initial LoRA parameters for prompt: '{original_prompt}' using random/zero initialization.")
    
    # Determine device for tensor creation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} for LoRA parameter initialization.")

    lora_rank = model_config.get("lora_rank", 8)
    lora_alpha = model_config.get("lora_alpha", lora_rank) # Common practice

    initial_parameters: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}

    # Define layer groups from config
    layer_groups = [
        ("UNet", model_config.get("target_layers_unet", []), model_config.get("layer_dimensions_unet", {})),
        ("TextEncoder1", model_config.get("target_layers_text_encoder", []), model_config.get("layer_dimensions_text_encoder", {})),
        ("TextEncoder2", model_config.get("target_layers_text_encoder_2", []), model_config.get("layer_dimensions_text_encoder_2", {}))
    ]

    for component_name, target_layers, layer_dimensions_map in layer_groups:
        if not target_layers:
            continue
        
        print(f"\nProposing for component: {component_name}")
        
        # LLM-specific info gathering and prompt generation removed.
        # Directly proceed to random/zero initialization.

        if component_name not in initial_parameters:
            initial_parameters[component_name] = {}

        for layer_name in target_layers:
            if layer_name not in layer_dimensions_map:
                # Already warned above, but double check before assignment
                continue

            # layer_dimensions_map provides [out_features, in_features]
            out_features, in_features = layer_dimensions_map[layer_name]
            
            # Standard LoRA initialization: A is random, B is zeros
            # lora_A (rank, in_features)
            # lora_B (out_features, rank)
            lora_A_tensor = torch.randn(lora_rank, in_features, device=device, dtype=torch.float32) * 0.01
            lora_B_tensor = torch.zeros(out_features, lora_rank, device=device, dtype=torch.float32)

            initial_parameters[component_name][layer_name] = {
                "lora_A": lora_A_tensor,
                "lora_B": lora_B_tensor
            }
            print(f"  Proposed for {component_name} layer '{layer_name}': A shape {lora_A_tensor.shape}, B shape {lora_B_tensor.shape}")

    print("\nInitial LoRA parameters proposed (as PyTorch tensors).")
    return initial_parameters

if __name__ == '__main__':
    # Example Usage
    sample_processed_prompt = {
        "tokenizer_1_output": None, # Not used by this proposer directly
        "tokenizer_2_output": None, # Not used by this proposer directly
        "original_prompt": "A futuristic cityscape with flying cars"
    }
    # More detailed mock config matching the new structure
    mock_config = {
        "model_name": "stabilityai/sdxl-lightning",
        "lora_rank": 8,
        "lora_alpha": 8,
        "target_layers_unet": ["unet_attn1.to_q", "unet_attn1.to_v"],
        "layer_dimensions_unet": {
            "unet_attn1.to_q": [320, 320], # out_features, in_features
            "unet_attn1.to_v": [320, 320],
        },
        "target_layers_text_encoder": ["te1_layer22.q_proj"],
        "layer_dimensions_text_encoder": {
            "te1_layer22.q_proj": [768, 768]
        },
        "target_layers_text_encoder_2": [], # No TE2 LoRA for this example
        "layer_dimensions_text_encoder_2": {},
        "conditioning_signals": {
            "sdxl_text_encoder_1_name": "openai/clip-vit-large-patch14",
            "sdxl_text_encoder_2_name": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
        },
        # "proposer_llm_config": {"model_name": "gpt2", "max_new_tokens": 50} # This config is no longer used by proposer
    }

    print("\nRunning example for propose_initial_lora_parameters:")
    try:
        # This example no longer downloads gpt2 or any LLM for proposal.
        proposed_params_ = propose_initial_lora_parameters(sample_processed_prompt, mock_config)
        
        for component_name_, component_layers_ in proposed_params_.items():
            print(f"Component: {component_name_}")
            for layer_name_, matrices_ in component_layers_.items():
                print(f"  Layer: {layer_name_}")
                print(f"    LoRA A (lora_A) shape: {matrices_['lora_A'].shape}, device: {matrices_['lora_A'].device}")
                print(f"    LoRA B (lora_B) shape: {matrices_['lora_B'].shape}, device: {matrices_['lora_B'].device}")
    except Exception as e:
        print(f"Error in example: {e}")
        print("This example requires the `torch` library.")
