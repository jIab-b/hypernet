"""
Loads and validates model configuration files.
"""
import json
from typing import Dict, Any

def load_model_configuration(config_path: str) -> Dict[str, Any]:
    """
    Loads a model configuration from a JSON file.

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        A dictionary containing the model configuration.
    
    Raises:
        FileNotFoundError: If the config file is not found.
        json.JSONDecodeError: If the config file is not valid JSON.
    """
    print(f"Loading model configuration from: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        print("Model configuration loaded successfully.")
        # Basic validation (can be expanded)
        if "model_name" not in config_data or "lora_rank" not in config_data:
            print("Warning: Basic config fields 'model_name' or 'lora_rank' missing.")
        return config_data
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        raise
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in configuration file {config_path}: {e}")
        raise

if __name__ == '__main__':
    # Example usage (assuming a dummy config file exists)
    # Create a dummy config for testing
    dummy_config_content = {
        "model_name": "ExampleDiffusionModel-v1",
        "lora_rank": 8,
        "target_layers": [
            "attention_block_1.q_proj",
            "attention_block_1.k_proj",
            "attention_block_1.v_proj"
        ],
        "layer_dimensions": {
            "attention_block_1.q_proj": [768, 320],
            "attention_block_1.k_proj": [768, 320],
            "attention_block_1.v_proj": [768, 320]
        },
        "conditioning_signals": {
            "text_embedding_dimensionality": 768
        }
    }
    dummy_path = "dummy_config.json"
    with open(dummy_path, 'w', encoding='utf-8') as f_dummy:
        json.dump(dummy_config_content, f_dummy, indent=4)

    try:
        loaded_config = load_model_configuration(dummy_path)
        print("\nLoaded dummy configuration:")
        print(json.dumps(loaded_config, indent=4))
    except Exception as e:
        print(f"Error during example usage: {e}")
    finally:
        # Clean up dummy file
        import os
        if os.path.exists(dummy_path):
            os.remove(dummy_path)