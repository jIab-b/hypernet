"""
Preprocessor for text prompts.
"""
from typing import List, Any

def preprocess_text(text_prompt: str, model_config: Any = None) -> List[str]:
    """
    Tokenizes the input text prompt.
    In a real scenario, this would use a tokenizer appropriate for the
    target model or the LoRA Parameter Proposer LLM.

    Args:
        text_prompt: The raw input string.
        model_config: Configuration for the target diffusion model,
                      which might specify tokenizer details. (Currently unused)

    Returns:
        A list of tokens (strings).
    """
    print(f"Original prompt: '{text_prompt}'")
    # Simple whitespace tokenization for placeholder
    tokens = text_prompt.lower().split()
    print(f"Tokens: {tokens}")
    return tokens

if __name__ == '__main__':
    # Example usage
    sample_prompt = "A beautiful sunset over the mountains"
    sample_tokens = preprocess_text(sample_prompt)
    # Expected: ['a', 'beautiful', 'sunset', 'over', 'the', 'mountains']