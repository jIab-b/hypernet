# Per-Prompt Optimized Text-to-LoRA System

## Overview

This project implements a system for generating LoRA (Low-Rank Adaptation) weights tailored to a specific text prompt for SDXL (Stable Diffusion XL) models. The goal is to create a LoRA file that, when applied to the base SDXL model, modifies its output to better align with the nuances of the input prompt. The system achieves this by initializing LoRA layers and then optimizing their weights based on a custom loss function derived from the prompt's interaction with the model's text encoders.

## Workflow

1.  **Input:** User provides a text prompt, a target base model name (e.g., `sdxl_base`), and an optional output path for the LoRA file.
2.  **Configuration Loading:** The system loads a JSON configuration file corresponding to the target model. This file specifies LoRA rank, alpha, target layers for adaptation (in UNet, TextEncoder1, TextEncoder2), and their dimensions.
3.  **Text Preprocessing:** The input prompt is tokenized using the SDXL model's specific text encoders (typically CLIP ViT-L and OpenCLIP ViT-bigG).
4.  **LoRA Initialization:** Initial LoRA A and B matrices are created for all configured target layers.
5.  **LoRA Injection & Optimization:**
    *   The base SDXL model components (UNet, Text Encoders) are loaded, and their original weights are frozen.
    *   LoRA layers with the initial weights are injected into the target modules of these components using the PEFT library.
    *   The weights of these LoRA layers are then iteratively optimized using a custom loss function.
6.  **Output:** The optimized LoRA weights are saved as a `.safetensors` file.

## Technical Implementation Details

### LoRA Initialization (`lora_proposer.py`)

The initial LoRA weights are not learned from an external LLM but are procedurally generated:
*   **Configuration-Driven:** The `model_config/<model_name>_config.json` file dictates:
    *   `lora_rank`: The rank of the LoRA matrices.
    *   `lora_alpha`: The scaling factor for LoRA.
    *   `target_layers_unet`, `target_layers_text_encoder`, `target_layers_text_encoder_2`: Lists of specific module names (e.g., `"q_proj"`, `"down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"`) to be adapted in each component.
    *   `layer_dimensions_unet`, `layer_dimensions_text_encoder`, `layer_dimensions_text_encoder_2`: Maps these layer names to their `[out_features, in_features]` dimensions.
*   **Matrix Creation:**
    *   For each target layer, LoRA A and B matrices are created:
        *   **LoRA A:** Shape `(lora_rank, in_features)`. Initialized with small random values drawn from a normal distribution (`torch.randn(...) * 0.01`).
        *   **LoRA B:** Shape `(out_features, lora_rank)`. Initialized to zeros (`torch.zeros(...)`).
    *   Tensors are created on the appropriate device (CUDA if available, else CPU).
*   **Structure:** The initial parameters are stored in a nested dictionary: `Dict[component_name, Dict[layer_name, {"lora_A": tensor, "lora_B": tensor}]]`.

### LoRA Injection & Optimization (`optimizer_engine/optimizer.py`)

*   **Base Model Preparation:** The specified SDXL pipeline (e.g., `stabilityai/stable-diffusion-xl-base-1.0`) is loaded. All original parameters of its components (UNet, TextEncoder1, TextEncoder2, VAE) are frozen (`param.requires_grad_(False)`).
*   **PEFT Integration:**
    *   For each component targeted for LoRA (UNet, TextEncoder1, TextEncoder2), a `LoraConfig` is created using the `lora_rank`, `lora_alpha`, and the list of `target_modules` (derived from the keys of the initialized parameters for that component).
    *   `peft.get_peft_model()` is used to inject trainable LoRA layers into the specified `target_modules` of the frozen base component.
    *   The initial random/zero weights (prepared by `lora_proposer`) are then copied into these newly injected LoRA layers.
*   **Optimization Loop:**
    *   An optimizer (default AdamW) is configured with the trainable LoRA parameters.
    *   The system iterates for a specified number of steps (configurable via `--iterations` command-line argument, or from `model_config`, defaulting to 50).
    *   In each iteration:
        1.  Gradients are zeroed.
        2.  The `calculate_alignment_score` (loss) is computed.
        3.  `loss.backward()` computes gradients for the LoRA parameters.
        4.  `optimizer.step()` updates the LoRA parameters.

### Loss Function (`calculate_alignment_score` in `optimizer_engine/optimizer.py`)

The loss function is designed to make the LoRA layers adapt the text encoders' outputs based on the input prompt. It does **not** involve generating images during the optimization loop.
The total loss comprises two main parts:

1.  **Prompt Embedding Loss:**
    *   **Input:** The tokenized `input_ids` and `attention_mask` from the `processed_prompt_data` (derived from the user's input prompt).
    *   **Process:**
        *   These tokenized inputs are passed through the LoRA-adapted PEFT versions of `TextEncoder1` and `TextEncoder2` (if they are targeted and have LoRA layers).
        *   The `last_hidden_state` (i.e., the output embeddings for each token) is retrieved from each text encoder's output.
        *   **Objective:** To encourage the LoRA layers to make these prompt embeddings more "significant" or "active." This is achieved by minimizing the negative mean of the absolute values of the `last_hidden_state`:
            `loss_te = -torch.mean(torch.abs(last_hidden_state_te))`
            Minimizing this term effectively maximizes `torch.mean(torch.abs(last_hidden_state_te))`.
        *   If both text encoders are processed, their individual prompt embedding losses are averaged.
    *   **Rationale:** This part of the loss directly conditions the LoRA weights on the input prompt by rewarding LoRA configurations that produce stronger (higher magnitude) contextual embeddings from the text encoders for that specific prompt.

2.  **L2 Regularization Loss:**
    *   **Process:** A standard L2 penalty is applied to all trainable LoRA parameters:
        `l2_reg_loss = regularization_strength * torch.sum(param ** 2)`
        where `regularization_strength` is a small coefficient (e.g., `1e-5`).
    *   **Rationale:** This helps prevent the LoRA weights from becoming excessively large, which can aid in generalization and stabilize training.

**Total Loss = (Average Prompt Embedding Loss) + L2 Regularization Loss**

This `total_loss` is then backpropagated to update only the trainable LoRA A and B weights.

## Key Modules

*   **`main.py`:** Orchestrates the entire process.
*   **`config_manager/loader.py`:** Loads model-specific configurations.
*   **`text_processing/preprocessor.py`:** Tokenizes the input prompt.
*   **`lora_proposer/proposer.py`:** Initializes LoRA A (random) and B (zero) matrices with correct shapes based on config.
*   **`lora_assembler/assembler.py`:** Validates and passes through the initial LoRA tensors (currently a simple pass-through as proposer outputs tensors directly).
*   **`optimizer_engine/optimizer.py`:** Loads the base model, injects LoRA layers using PEFT, runs the optimization loop using the custom prompt-conditioned loss function.
*   **`output_manager/saver.py`:** Saves the final optimized LoRA weights to a `.safetensors` file.

## Usage Example

```bash
python main.py --prompt "a vibrant oil painting of a steampunk owl" --model_name sdxl_base --output_path "output/steampunk_owl_lora.safetensors" --iterations 100
```
This command will generate a LoRA named `steampunk_owl_lora.safetensors` in the `output/` directory, trained for 100 iterations on the prompt "a vibrant oil painting of a steampunk owl" using the `sdxl_base` configuration.