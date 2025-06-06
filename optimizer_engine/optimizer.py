"""
Performs per-prompt LoRA parameter optimization using PyTorch, Diffusers, and PEFT.
"""
from typing import Dict, Any, Tuple, Optional
import torch
from diffusers import DiffusionPipeline
from peft import LoraConfig, get_peft_model, PeftModel

# Global cache for the SDXL pipeline to avoid reloading
_sdxl_pipeline_cache: Optional[DiffusionPipeline] = None
_loaded_sdxl_model_name: Optional[str] = None

def load_sdxl_pipeline_and_components(model_name: str, model_variant: Optional[str] = None) -> DiffusionPipeline:
    global _sdxl_pipeline_cache, _loaded_sdxl_model_name
    if _sdxl_pipeline_cache is not None and _loaded_sdxl_model_name == model_name:
        print(f"Using cached SDXL pipeline: {model_name}")
        return _sdxl_pipeline_cache

    print(f"Loading SDXL pipeline: {model_name}, variant: {model_variant}")
    try:
        pipeline = DiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=torch.float16, # SDXL Lightning often uses float16
            variant=model_variant, # e.g., "fp16"
            # use_safetensors=True # Recommended
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        pipeline.to(device)
        
        # Freeze all original parameters
        for component_name in ["unet", "text_encoder", "text_encoder_2", "vae"]:
            component = getattr(pipeline, component_name, None)
            if component:
                for param in component.parameters():
                    param.requires_grad_(False)
                component.eval() # Set to eval mode
        
        _sdxl_pipeline_cache = pipeline
        _loaded_sdxl_model_name = model_name
        print(f"SDXL pipeline {model_name} loaded to {device} and frozen.")
        return pipeline
    except Exception as e:
        print(f"Error loading SDXL pipeline '{model_name}': {e}")
        print("Ensure model name, variant are correct, and you have internet/credentials.")
        raise

def inject_lora_and_set_weights(
    base_model_component: torch.nn.Module,
    component_name_for_logging: str, # e.g. "UNet"
    lora_config: LoraConfig,
    initial_lora_weights_for_component: Dict[str, Dict[str, torch.Tensor]],
    device: torch.device
) -> PeftModel:
    """Injects LoRA adapters and sets their initial weights."""
    print(f"Injecting LoRA into {component_name_for_logging}...")
    peft_component = get_peft_model(base_model_component, lora_config)
    peft_component.to(device) # Ensure PEFT model is on the correct device

    print(f"Setting initial LoRA weights for {component_name_for_logging}...")
    with torch.no_grad():
        for peft_param_name, peft_param_value in peft_component.named_parameters():
            if not peft_param_value.requires_grad: # Skip frozen base model params
                continue

            # Expected PEFT LoRA weight names:
            # base_model.model.TARGET_MODULE.lora_A.ADAPTER_NAME.weight
            # base_model.model.TARGET_MODULE.lora_B.ADAPTER_NAME.weight
            # ADAPTER_NAME is 'default' if not specified in LoraConfig.
            # TARGET_MODULE is the key from initial_lora_weights_for_component
            
            # Simplify: find original_module_name and lora_type (A or B)
            parts = peft_param_name.split('.')
            # Example: parts = ['base_model', 'model', 'down_blocks', '0', ..., 'to_q', 'lora_A', 'default', 'weight']
            # We need to reconstruct the original_module_name that matches our initial_lora_weights keys.
            # This depends on how PEFT constructs names and how `target_modules` in LoraConfig was specified.
            # If target_modules were like "to_q", PEFT might make it "...attention_0.to_q.lora_A..."
            # This name matching is tricky.
            
            # Let's assume initial_lora_weights_for_component keys are the full module paths
            # that PEFT targets.
            
            # A more robust way: Iterate initial_lora_weights_for_component keys
            for init_lora_key, init_lora_mats in initial_lora_weights_for_component.items():
                # init_lora_key is like "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"
                # peft_param_name is like "base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight"
                
                # Check for lora_A
                expected_peft_lora_A_suffix = f"{init_lora_key}.lora_A.default.weight"
                if peft_param_name.endswith(expected_peft_lora_A_suffix):
                    if "lora_A" in init_lora_mats:
                        print(f"  Setting {peft_param_name} from initial_lora_weights_for_component['{init_lora_key}']['lora_A']")
                        peft_param_value.copy_(init_lora_mats["lora_A"].to(device))
                    break
                
                # Check for lora_B
                expected_peft_lora_B_suffix = f"{init_lora_key}.lora_B.default.weight"
                if peft_param_name.endswith(expected_peft_lora_B_suffix):
                    if "lora_B" in init_lora_mats:
                        print(f"  Setting {peft_param_name} from initial_lora_weights_for_component['{init_lora_key}']['lora_B']")
                        peft_param_value.copy_(init_lora_mats["lora_B"].to(device))
                    break
    peft_component.train() # Set PEFT model to train mode for optimization
    return peft_component


def calculate_alignment_score(
    peft_unet: PeftModel,
    peft_text_encoder_1: Optional[PeftModel],
    peft_text_encoder_2: Optional[PeftModel],
    processed_prompt_data: Dict[str, Any], # Contains tokenized_1_output, tokenized_2_output
    model_config: Dict[str, Any], # Contains SDXL model info
    sdxl_pipeline: DiffusionPipeline # To access other components if needed
) -> torch.Tensor:
    """
    Calculates the alignment score (loss to be minimized).
    This version implements a basic prompt-conditioning by encouraging
    LoRA-adapted text encoders to produce larger magnitude embeddings for the prompt.
    """
    original_prompt_text = processed_prompt_data["original_prompt"]
    print(f"  Calculating alignment score for prompt: '{original_prompt_text}'")

    device = peft_unet.device # Assume unet is always present and on the correct device
    total_loss = torch.tensor(0.0, device=device, requires_grad=False)
    
    # Loss from Text Encoders based on prompt
    prompt_embedding_loss = torch.tensor(0.0, device=device)
    num_text_encoders_processed = 0

    if peft_text_encoder_1 and "tokenizer_1_output" in processed_prompt_data:
        token_data_1 = processed_prompt_data["tokenizer_1_output"]
        input_ids_1 = token_data_1["input_ids"].to(device)
        attention_mask_1 = token_data_1["attention_mask"].to(device)
        
        try:
            outputs_1 = peft_text_encoder_1(input_ids=input_ids_1, attention_mask=attention_mask_1, output_hidden_states=True)
            # Use last_hidden_state. We want to encourage LoRA to make this significant.
            # A simple way is to maximize its L2 norm (or minimize negative L2 norm).
            # Or, maximize sum of absolute values.
            last_hidden_state_1 = outputs_1.last_hidden_state
            # Loss = -mean(abs(embeddings)) to encourage larger values
            prompt_embedding_loss = prompt_embedding_loss - torch.mean(torch.abs(last_hidden_state_1))
            num_text_encoders_processed += 1
        except Exception as e:
            print(f"    Warning: Error processing prompt with TextEncoder1: {e}")


    if peft_text_encoder_2 and "tokenizer_2_output" in processed_prompt_data:
        token_data_2 = processed_prompt_data["tokenizer_2_output"]
        input_ids_2 = token_data_2["input_ids"].to(device)
        attention_mask_2 = token_data_2["attention_mask"].to(device)

        try:
            outputs_2 = peft_text_encoder_2(input_ids=input_ids_2, attention_mask=attention_mask_2, output_hidden_states=True)
            last_hidden_state_2 = outputs_2.last_hidden_state
            prompt_embedding_loss = prompt_embedding_loss - torch.mean(torch.abs(last_hidden_state_2))
            num_text_encoders_processed += 1
        except Exception as e:
            print(f"    Warning: Error processing prompt with TextEncoder2: {e}")

    if num_text_encoders_processed > 0:
        total_loss = total_loss + (prompt_embedding_loss / num_text_encoders_processed) # Average if both used
    else:
        # If no text encoders are LoRA-adapted or processed, this part of loss is zero.
        # This might happen if only UNet is targeted.
        pass

    # L2 regularization for LoRA weights (small penalty to prevent weights from exploding, or to encourage sparsity if negative)
    # Here, we add a small penalty to the sum of squares of LoRA weights to keep them from growing too large.
    # A very small weight_decay can also encourage weights to be non-zero if the main loss is pushing them down.
    l2_reg_loss = torch.tensor(0.0, device=device)
    regularization_strength = 1e-5 # Small regularization

    for peft_model_component in [peft_unet, peft_text_encoder_1, peft_text_encoder_2]:
        if peft_model_component:
            for param in peft_model_component.parameters():
                if param.requires_grad: # Only LoRA params
                    l2_reg_loss = l2_reg_loss + torch.sum(param ** 2)
    
    total_loss = total_loss + regularization_strength * l2_reg_loss

    # Ensure the final loss requires gradients.
    # If total_loss was built from non-leaf tensors that require_grad, it should be fine.
    # If it was all constants, make it a leaf that requires grad.
    # The prompt_embedding_loss part should carry gradients back to text encoder LoRAs.
    # The l2_reg_loss part should carry gradients to all LoRA params.
    # If total_loss is a Python scalar (e.g. 0.0) and then we add tensors, it becomes a tensor.
    # If all components were skipped, total_loss could remain a 0.0 tensor not requiring grad.
    # However, l2_reg_loss should always make it require grad if there are trainable params.

    print(f"  Alignment loss: {total_loss.item():.4f} (Prompt Emb Loss part: {prompt_embedding_loss.item() if isinstance(prompt_embedding_loss, torch.Tensor) else prompt_embedding_loss:.4f}, L2 Reg Loss part: {l2_reg_loss.item():.4f})")
    return total_loss


def optimize_lora_parameters(
    initial_lora_params: Dict[str, Dict[str, Dict[str, torch.Tensor]]], # Nested: component -> layer -> {lora_A/B -> tensor}
    processed_prompt_data: Dict[str, Any],
    model_config: Dict[str, Any]
) -> Dict[str, Dict[str, Dict[str, torch.Tensor]]]:
    """
    Optimizes LoRA parameters for the given prompt using SDXL Lightning and PEFT.
    """
    print("Starting LoRA parameter optimization with SDXL Lightning and PEFT...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sdxl_model_name = model_config.get("model_name", "stabilityai/sdxl-lightning")
    sdxl_model_variant = model_config.get("model_variant")
    pipeline = load_sdxl_pipeline_and_components(sdxl_model_name, sdxl_model_variant)

    lora_rank = model_config.get("lora_rank", 8)
    lora_alpha = model_config.get("lora_alpha", lora_rank)
    # PEFT's LoraConfig can take target_modules as a list of strings (module names/regex)
    # For SDXL, common targets are 'Attention', 'FeedForward' or specific names like 'to_q', 'to_k', 'to_v', 'to_out'.
    # The `target_module_types` in our config can guide this.
    # For simplicity, we'll assume `initial_lora_params` keys are the exact module names PEFT needs.
    # This might need adjustment based on how PEFT resolves target_modules.
    # A common approach is to target by module type, e.g. all diffusers.models.attention_processor.Attention layers.
    
    # For this placeholder, we assume `initial_lora_params` keys are specific enough to be found by PEFT
    # if we pass them as `target_modules`. This is a simplification.
    # A more robust way is to use `target_modules` with class names or broad regex
    # and then map our `initial_lora_params` to the created LoRA layers.
    
    # Let's define target_modules for PEFT based on the keys in initial_lora_params for each component.
    # This assumes initial_lora_params keys are module names that PEFT can directly find and replace/wrap.
    
    peft_models = {}
    all_trainable_params = []

    # --- UNet LoRA ---
    initial_lora_unet_component_params = initial_lora_params.get("UNet", {})
    if initial_lora_unet_component_params:
        lora_config_unet = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(initial_lora_unet_component_params.keys()), # Use the exact layer names from this component
            lora_dropout=0.05,
            bias="none", # or "all" or "lora_only"
        )
        peft_models["unet"] = inject_lora_and_set_weights(pipeline.unet, "UNet", lora_config_unet, initial_lora_unet_component_params, device)
        all_trainable_params.extend(filter(lambda p: p.requires_grad, peft_models["unet"].parameters()))

    # --- Text Encoder 1 LoRA ---
    initial_lora_te1_component_params = initial_lora_params.get("TextEncoder1", {})
    if initial_lora_te1_component_params and pipeline.text_encoder:
        lora_config_te1 = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=list(initial_lora_te1_component_params.keys()), lora_dropout=0.05, bias="none")
        peft_models["text_encoder_1"] = inject_lora_and_set_weights(pipeline.text_encoder, "TextEncoder1", lora_config_te1, initial_lora_te1_component_params, device)
        all_trainable_params.extend(filter(lambda p: p.requires_grad, peft_models["text_encoder_1"].parameters()))

    # --- Text Encoder 2 LoRA ---
    initial_lora_te2_component_params = initial_lora_params.get("TextEncoder2", {})
    if initial_lora_te2_component_params and pipeline.text_encoder_2:
        lora_config_te2 = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=list(initial_lora_te2_component_params.keys()), lora_dropout=0.05, bias="none")
        peft_models["text_encoder_2"] = inject_lora_and_set_weights(pipeline.text_encoder_2, "TextEncoder2", lora_config_te2, initial_lora_te2_component_params, device)
        all_trainable_params.extend(filter(lambda p: p.requires_grad, peft_models["text_encoder_2"].parameters()))

    if not all_trainable_params:
        print("Warning: No LoRA layers were targeted or no initial parameters provided for them. Optimization will not occur.")
        return initial_lora_params # Return original if nothing to optimize

    opt_config = model_config.get("optimization_params", {})
    iterations = opt_config.get("iterations", 10)
    learning_rate = opt_config.get("learning_rate", 1e-4)
    optimizer_type = opt_config.get("optimizer_type", "AdamW")

    if optimizer_type.lower() == "adamw":
        optimizer = torch.optim.AdamW(all_trainable_params, lr=learning_rate)
    else: # Default to Adam
        optimizer = torch.optim.Adam(all_trainable_params, lr=learning_rate)
    
    print(f"Optimizer: {optimizer_type}, LR: {learning_rate}, Iterations: {iterations}")
    print(f"Number of trainable LoRA parameters: {sum(p.numel() for p in all_trainable_params)}")

    for i in range(iterations):
        optimizer.zero_grad()
        
        loss = calculate_alignment_score(
            peft_models.get("unet"),
            peft_models.get("text_encoder_1"),
            peft_models.get("text_encoder_2"),
            processed_prompt_data,
            model_config,
            pipeline
        )
        
        loss.backward()
        optimizer.step()
        print(f"  Optimization iteration {i+1}/{iterations}, Loss: {loss.item():.6f}")

    # Extract optimized LoRA weights
    optimized_lora_params: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
    
    # Retrieve target layer keys from model_config for mapping back
    target_layers_unet_keys = model_config.get("target_layers_unet", [])
    target_layers_te1_keys = model_config.get("target_layers_text_encoder", [])
    target_layers_te2_keys = model_config.get("target_layers_text_encoder_2", [])

    for component_key, peft_model_component in peft_models.items():
        if peft_model_component:
            if component_key not in optimized_lora_params:
                optimized_lora_params[component_key] = {}

            component_lora_keys = []
            if component_key == "unet": component_lora_keys = target_layers_unet_keys
            elif component_key == "text_encoder_1": component_lora_keys = target_layers_te1_keys
            elif component_key == "text_encoder_2": component_lora_keys = target_layers_te2_keys

            for param_name, param_value in peft_model_component.state_dict().items():
                if "lora_" in param_name and param_value.requires_grad:
                    for original_key in component_lora_keys: # original_key is "q_proj", "down_blocks..." etc.
                        if original_key in param_name: # Check if "q_proj" is in the full PEFT param name
                            lora_type = "lora_A" if ".lora_A." in param_name else "lora_B"
                            if original_key not in optimized_lora_params[component_key]:
                                optimized_lora_params[component_key][original_key] = {}
                            optimized_lora_params[component_key][original_key][lora_type] = param_value.cpu().clone()
                            print(f"  Extracted optimized {lora_type} for {component_key}/{original_key}, shape {param_value.shape}")
                            break
                            
    print("LoRA parameter optimization finished.")
    return optimized_lora_params

if __name__ == '__main__':
    # Example Usage - This will be very slow if actually loading SDXL
    # and requires significant setup (model download, GPU).
    # For a quick test, mock the pipeline and peft model interactions.
    
    print("Optimizer example: This is a complex example and may require actual model downloads.")
    print("Ensure you have `diffusers`, `transformers`, `peft`, `accelerate` installed and a CUDA GPU for speed.")

    # Mock initial LoRA parameters (as if from proposer)
    # Keys should be full module paths that PEFT will target.
    # These names need to be exact and depend on the SDXL model structure.
    # The ones in example_diffusion_config.json are illustrative.
    
    # Let's use a very small set for the example to run faster if it tries.
    # These layer names are illustrative for a small part of SDXL.
    # In a real scenario, these would come from the config file.
    mock_initial_lora = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q": {
            "lora_A": torch.randn(8, 320, dtype=torch.float16) * 0.01, # rank, in_features
            "lora_B": torch.zeros(320, 8, dtype=torch.float16)       # out_features, rank
        },
        # Only one UNet layer for this example
        # "text_model.encoder.layers.22.self_attn.q_proj": { # For text_encoder_1
        #     "lora_A": torch.randn(8, 768, dtype=torch.float16) * 0.01,
        #     "lora_B": torch.zeros(768, 8, dtype=torch.float16)
        # }
    }

    mock_processed_prompt = {
        "tokenizer_1_output": {"input_ids": torch.randint(0,100,(1,77)), "attention_mask": torch.ones(1,77)}, # Dummy tensors
        "tokenizer_2_output": {"input_ids": torch.randint(0,100,(1,77)), "attention_mask": torch.ones(1,77)},
        "original_prompt": "A cat astronaut"
    }

    # Use the updated example_diffusion_config.json for this test
    # Make sure the target_layers in the config match the keys in mock_initial_lora for the test
    # For this example, let's simplify the config passed to the optimizer
    mock_model_cfg = {
        "model_name": "stabilityai/sdxl-lightning", # Will attempt download if not cached
        "model_variant": "fp16",
        "lora_rank": 8,
        "lora_alpha": 8,
        "target_module_types": ["Attention"], # PEFT uses this to find layers
        "target_layers_unet": ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"], # Match mock_initial_lora
        "layer_dimensions_unet": { # Not directly used by optimizer if PEFT handles shapes
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q": [320, 320]
        },
        "target_layers_text_encoder": [], # No text encoder LoRA in this simplified example
        "layer_dimensions_text_encoder": {},
        "target_layers_text_encoder_2": [],
        "layer_dimensions_text_encoder_2": {},
        "optimization_params": {"iterations": 3, "learning_rate": 1e-5, "optimizer_type": "AdamW"},
        "conditioning_signals": {} # Not used by optimizer directly
    }

    print("Attempting to run optimizer example (may download models)...")
    try:
        # To prevent actual model download for a quick syntax check, one might mock
        # load_sdxl_pipeline_and_components and inject_lora_and_set_weights.
        # For now, it will try to run.
        
        # Reduce iterations for example
        # mock_model_cfg["optimization_params"]["iterations"] = 2
        
        # Check if CUDA is available, otherwise this will be very slow or fail on CPU for large models
        if not torch.cuda.is_available():
            print("WARNING: No CUDA GPU detected. This example will be extremely slow or may fail on CPU due to memory.")
            # To skip the heavy part for a quick test if no GPU:
            # raise NotImplementedError("Skipping optimizer example without CUDA GPU.")


        final_lora_params = optimize_lora_parameters(
            mock_initial_lora,
            mock_processed_prompt,
            mock_model_cfg
        )
        print("\nOptimizer Example - Final LoRA Parameters (shapes):")
        for layer, matrices in final_lora_params.items():
            print(f"  Layer: {layer}, A: {matrices['lora_A'].shape}, B: {matrices['lora_B'].shape}")

    except Exception as e:
        print(f"Error during optimizer example: {e}")
        import traceback
        traceback.print_exc()
