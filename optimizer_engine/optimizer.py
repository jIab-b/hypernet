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
    Placeholder for actual conditional language modeling objective.
    """
    original_prompt_text = processed_prompt_data["original_prompt"]
    print(f"  Calculating alignment score for prompt: '{original_prompt_text}'")

    # --- Placeholder Score Calculation ---
    # A real implementation would use the peft_models and tokenized inputs
    # to compute a loss based on the "conditional language modeling objective".
    # This might involve:
    # 1. Getting embeddings from peft_text_encoder_1 and peft_text_encoder_2.
    # 2. If a temporary LM head is added to one of them, calculate cross-entropy loss
    #    for predicting the input prompt tokens.
    # For now, simulate a loss that changes based on LoRA params.
    dummy_loss = torch.tensor(0.0, device=peft_unet.device, requires_grad=True)
    
    # Make loss slightly dependent on sum of LoRA params to see optimizer working
    for peft_model_component in [peft_unet, peft_text_encoder_1, peft_text_encoder_2]:
        if peft_model_component:
            for param in peft_model_component.parameters():
                if param.requires_grad: # Only LoRA params
                    dummy_loss = dummy_loss + param.abs().sum() * 0.0001 # Small contribution

    # Simulate a more meaningful change based on prompt content (very arbitrary)
    if "cat" in original_prompt_text and "astronaut" in original_prompt_text:
        dummy_loss = dummy_loss - torch.tensor(0.1, device=peft_unet.device) # Lower loss if keywords match
    elif "cat" in original_prompt_text:
        dummy_loss = dummy_loss - torch.tensor(0.05, device=peft_unet.device)

    print(f"  Dummy alignment loss: {dummy_loss.item():.4f}")
    return dummy_loss


def optimize_lora_parameters(
    initial_lora_params: Dict[str, Dict[str, torch.Tensor]], # Keys are full module paths
    processed_prompt_data: Dict[str, Any],
    model_config: Dict[str, Any]
) -> Dict[str, Dict[str, torch.Tensor]]:
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
    target_layers_unet_keys = model_config.get("target_layers_unet", [])
    initial_lora_unet = {k: v for k, v in initial_lora_params.items() if k in target_layers_unet_keys}
    if initial_lora_unet:
        # For PEFT, target_modules should be the names of the nn.Linear, nn.Conv2d etc. layers to adapt.
        # The keys in initial_lora_unet are these names.
        lora_config_unet = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=list(initial_lora_unet.keys()), # Use the exact layer names
            lora_dropout=0.05,
            bias="none", # or "all" or "lora_only"
        )
        peft_models["unet"] = inject_lora_and_set_weights(pipeline.unet, "UNet", lora_config_unet, initial_lora_unet, device)
        all_trainable_params.extend(filter(lambda p: p.requires_grad, peft_models["unet"].parameters()))

    # --- Text Encoder 1 LoRA ---
    target_layers_te1_keys = model_config.get("target_layers_text_encoder", [])
    initial_lora_te1 = {k: v for k, v in initial_lora_params.items() if k in target_layers_te1_keys}
    if initial_lora_te1 and pipeline.text_encoder:
        lora_config_te1 = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=list(initial_lora_te1.keys()), lora_dropout=0.05, bias="none")
        peft_models["text_encoder_1"] = inject_lora_and_set_weights(pipeline.text_encoder, "TextEncoder1", lora_config_te1, initial_lora_te1, device)
        all_trainable_params.extend(filter(lambda p: p.requires_grad, peft_models["text_encoder_1"].parameters()))

    # --- Text Encoder 2 LoRA ---
    target_layers_te2_keys = model_config.get("target_layers_text_encoder_2", [])
    initial_lora_te2 = {k: v for k, v in initial_lora_params.items() if k in target_layers_te2_keys}
    if initial_lora_te2 and pipeline.text_encoder_2:
        lora_config_te2 = LoraConfig(r=lora_rank, lora_alpha=lora_alpha, target_modules=list(initial_lora_te2.keys()), lora_dropout=0.05, bias="none")
        peft_models["text_encoder_2"] = inject_lora_and_set_weights(pipeline.text_encoder_2, "TextEncoder2", lora_config_te2, initial_lora_te2, device)
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
    optimized_lora_params: Dict[str, Dict[str, torch.Tensor]] = {}
    for component_key, peft_model_component in peft_models.items():
        if peft_model_component:
            # state_dict keys will be like 'base_model.model.TARGET_MODULE.lora_A.default.weight'
            # We need to map these back to our original `initial_lora_params` keys.
            component_lora_keys = []
            if component_key == "unet": component_lora_keys = target_layers_unet_keys
            elif component_key == "text_encoder_1": component_lora_keys = target_layers_te1_keys
            elif component_key == "text_encoder_2": component_lora_keys = target_layers_te2_keys

            for param_name, param_value in peft_model_component.state_dict().items():
                if "lora_" in param_name and param_value.requires_grad: # Check if it's a LoRA parameter we trained
                    # Example param_name: base_model.model.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight
                    # Find original_key that matches the module path part
                    for original_key in component_lora_keys:
                        if original_key in param_name:
                            lora_type = "lora_A" if ".lora_A." in param_name else "lora_B"
                            if original_key not in optimized_lora_params:
                                optimized_lora_params[original_key] = {}
                            optimized_lora_params[original_key][lora_type] = param_value.cpu().clone() # Store on CPU
                            print(f"  Extracted optimized {lora_type} for {original_key}, shape {param_value.shape}")
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
