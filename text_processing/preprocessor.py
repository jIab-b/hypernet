"""
Preprocessor for text prompts, adapted for SDXL tokenization.
"""
from typing import Dict, Any
from transformers import AutoTokenizer #, CLIPTokenizer # CLIPTokenizer is often used directly

def preprocess_text(text_prompt: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tokenizes the input text prompt using SDXL's two tokenizers.

    Args:
        text_prompt: The raw input string.
        model_config: Configuration for the target diffusion model.
                      Expected to have `conditioning_signals` with
                      `sdxl_text_encoder_1_name` and `sdxl_text_encoder_2_name`.

    Returns:
        A dictionary containing the tokenized outputs from both tokenizers.
        Example: {
            "tokenizer_1_output": {"input_ids": ..., "attention_mask": ...},
            "tokenizer_2_output": {"input_ids": ..., "attention_mask": ...}
        }
    """
    print(f"Original prompt: '{text_prompt}'")

    conditioning_signals = model_config.get("conditioning_signals", {})
    tokenizer_1_name = conditioning_signals.get("sdxl_text_encoder_1_name")
    tokenizer_2_name = conditioning_signals.get("sdxl_text_encoder_2_name")

    if not tokenizer_1_name or not tokenizer_2_name:
        raise ValueError(
            "SDXL tokenizer names not found in model_config.conditioning_signals. "
            "Expected 'sdxl_text_encoder_1_name' and 'sdxl_text_encoder_2_name'."
        )

    try:
        print(f"Loading tokenizer 1: {tokenizer_1_name}")
        # For SDXL, the tokenizers are often derived from the main model checkpoint
        # or specific CLIP model names.
        # Example: "openai/clip-vit-large-patch14" for tokenizer_1
        # and "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" for tokenizer_2's text encoder,
        # but the actual tokenizer might be a more generic CLIPTokenizer.
        # Using AutoTokenizer should work if the names are correct HF model IDs.
        tokenizer_1 = AutoTokenizer.from_pretrained(tokenizer_1_name)
        print(f"Loading tokenizer 2: {tokenizer_2_name}")
        tokenizer_2 = AutoTokenizer.from_pretrained(tokenizer_2_name)
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        print("Make sure you have an internet connection and the model names are correct.")
        print("You might need to log in to Hugging Face CLI: `huggingface-cli login`")
        raise

    # Standard tokenization arguments
    # For SDXL, padding="max_length", truncation=True, max_length=tokenizer.model_max_length
    # is common.
    tokenized_1_output = tokenizer_1(
        text_prompt,
        padding="max_length",
        max_length=tokenizer_1.model_max_length,
        truncation=True,
        return_tensors="pt" # Return PyTorch tensors
    )
    print(f"Tokenizer 1 (e.g., CLIP ViT-L) output shape: {tokenized_1_output.input_ids.shape}")

    tokenized_2_output = tokenizer_2(
        text_prompt,
        padding="max_length",
        max_length=tokenizer_2.model_max_length,
        truncation=True,
        return_tensors="pt"
    )
    print(f"Tokenizer 2 (e.g., OpenCLIP ViT-bigG) output shape: {tokenized_2_output.input_ids.shape}")
    
    # For now, returning the direct output. The optimizer engine will handle embedding.
    # We might want to convert to list of IDs if other modules expect that,
    # but PyTorch tensors are good for direct use with models.
    return {
        "tokenizer_1_output": tokenized_1_output,
        "tokenizer_2_output": tokenized_2_output,
        "original_prompt": text_prompt # Keep original prompt for proposer LLM
    }

if __name__ == '__main__':
    # Example usage
    sample_prompt_ = "A hyper-realistic cat astronaut exploring a vibrant nebula"
    # This example config needs to be available or mocked for the test to run.
    # For a standalone test, we might need to provide a simpler mock_config.
    mock_sdxl_config = {
        "conditioning_signals": {
            # Using the actual SDXL base model to derive its tokenizer names
            # This is a common pattern.
            # Tokenizer 1 for SDXL is usually from "openai/clip-vit-large-patch14"
            # Tokenizer 2 for SDXL is usually from the second text encoder like OpenCLIP
            # The specific HF names for these tokenizers as standalone might vary.
            # Let's use the ones from the SDXL 1.0 base config.
            "sdxl_text_encoder_1_name": "openai/clip-vit-large-patch14", # Or "runwayml/stable-diffusion-v1-5" subfolder tokenizer
            "sdxl_text_encoder_2_name": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k" # This is a model, tokenizer might be different
                                                                                # For SDXL, often tokenizer_2 is also a CLIPTokenizer
                                                                                # but configured for the second text encoder.
                                                                                # Let's use a known compatible one for ViT-bigG if possible,
                                                                                # or rely on AutoTokenizer to fetch correctly if the model ID implies it.
                                                                                # For simplicity, we'll assume these are valid tokenizer IDs.
        },
        # Other config parts not strictly needed for this isolated test
    }
    print("\nRunning example for preprocess_text with mock SDXL config:")
    try:
        # Note: This example will try to download tokenizers from Hugging Face Hub.
        tokenized_outputs = preprocess_text(sample_prompt_, mock_sdxl_config)
        print("\nTokenized Outputs (shapes):")
        print(f"  Tokenizer 1 input_ids shape: {tokenized_outputs['tokenizer_1_output']['input_ids'].shape}")
        print(f"  Tokenizer 2 input_ids shape: {tokenized_outputs['tokenizer_2_output']['input_ids'].shape}")
        print(f"  Original prompt: {tokenized_outputs['original_prompt']}")
    except Exception as e:
        print(f"Error in example: {e}")
        print("This example requires the `transformers` library and internet access.")
        print("Ensure `openai/clip-vit-large-patch14` and `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` are valid for AutoTokenizer or use appropriate ones.")
        print("Often, for SDXL, you'd load a DiffusionPipeline and use pipeline.tokenizer and pipeline.tokenizer_2.")
