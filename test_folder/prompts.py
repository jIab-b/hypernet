import argparse
import os
import json
from openai import OpenAI

def generate_creative_prompts(original_prompt: str, num_prompts: int, api_key: str) -> list[str]:
    """
    Generates a series of creative and expressive prompts based on an original prompt.
    """
    client = OpenAI(api_key=api_key)
    generated_prompts = []

    print(f"Attempting to generate {num_prompts} prompts. This may take some time...")
    for i in range(num_prompts):
        print(f"Generating prompt variation {i+1}/{num_prompts} for original: '{original_prompt}'...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o", # Updated from gpt-4.1 for better availability/reliability
                messages=[
                    {
                        "role": "system",
                        "content": "You are a creative assistant that expands on user prompts to make them more verbose, "
                                   "expressive, and imaginative for image generation. "
                                   "Each new prompt should be a distinct variation or elaboration of the original idea."
                    },
                    {
                        "role": "user",
                        "content": f"Given the prompt: '{original_prompt}', generate a more verbose, creative, and expressive version. do not give a response longer than 77 tokens"
                    }
                ],
                max_tokens=77,
                n=1,
                stop=None,
                temperature=0.99,
                timeout=10.0, # Added timeout for each API call (60 seconds)
            )
            if response.choices:
                print('received one response')
                creative_prompt = response.choices[0].message.content.strip()
                generated_prompts.append(creative_prompt)
            else:
                generated_prompts.append(f"Error: No response for variation {i+1}")
        except Exception as e:
            print(f"Error generating prompt variation {i+1}: {e}")
            generated_prompts.append(f"Error: Could not generate variation {i+1} - {e}")
    
    return generated_prompts

def save_prompts_to_jsonl(prompts: list[str], filename: str):
    """
    Saves a list of prompts to a .jsonl file.
    Each prompt is saved as a JSON object on a new line.
    """
    with open(filename, 'w') as f:
        for i, prompt_text in enumerate(prompts):
            json_record = {"id": i, "prompt": prompt_text}
            f.write(json.dumps(json_record) + '\n')
    print(f"Successfully saved {len(prompts)} prompts to {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate creative prompts using OpenAI and save to a .jsonl file.")
    parser.add_argument("--prompt", type=str, required=True, help="The initial prompt to expand upon.")
    parser.add_argument("--num-images", type=int, required=True, help="The number of creative prompts to generate (corresponds to number of images).")
    parser.add_argument("--output-file", type=str, required=True, help="The name of the .jsonl file to save prompts to.")
    
    args = parser.parse_args()
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: The OPENAI_API_KEY environment variable is not set.")
        return
        
    print(f"Generating {args.num_images} creative prompts based on: '{args.prompt}'...")
    creative_prompts = generate_creative_prompts(args.prompt, args.num_images, api_key)
    
    if creative_prompts:
        save_prompts_to_jsonl(creative_prompts, args.output_file)
    else:
        print("No prompts were generated.")

if __name__ == "__main__":
    main()