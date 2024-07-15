import csv
import math
from transformers import LlamaTokenizer


# Define a function to load prompts from a CSV file and preprocess them
def load_prompts(input_csv="prefix_cache_input.csv", tokenizer_dir=None):
    header = {}
    tokens = []
    prompts = []
    tokenizer = None
    # Check if a tokenizer directory is provided
    if tokenizer_dir is not None:
        # Load the tokenizer from the given directory
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir)

    # Open the input CSV file
    with open(input_csv, "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(spamreader):
            if idx == 0:
                # Skip the first row (file description)
                continue

            if idx == 1:
                # Map header names to their indices
                for h_idx, header_name in enumerate(row):
                    header[header_name] = h_idx
                continue

            # Extract the raw message from the row
            raw_message = row[header["raw_input"]].strip('"').strip(
                """\"}], 'stream': False, 'temperature': 0.5, 'max_length': 
8192, 'top_p': 1.0, 'delete_prompt_from_output': 1}""").strip(
                    "{'messages': [{'role': 'user', 'content': \"")
            # Wrap the raw message with [INST] tags
            prompt = "[INST]%s[/INST]" % raw_message
            if tokenizer is None:
                token_list = []
            else:
                # Tokenize the prompt using the provided tokenizer
                token_list = tokenizer.encode(prompt, add_special_tokens=True)

            # Add the prompt and its tokens to their respective lists
            prompts.append(prompt)
            tokens.append(token_list)

    # Remove specific indices from the prompts and tokens lists
    skip_list = [15, 17, 22]
    for skip_idx in skip_list:
        del prompts[skip_idx]
        del tokens[skip_idx]

    # Calculate the length of the longest common prefix in tokens
    longest_prefix_tokens_num = 0
    tokens_number_list = [len(t) for t in tokens]
    tokens_min_len = min(tokens_number_list)
    if tokenizer is not None:
        for idx in range(tokens_min_len):
            tmp_set = set()
            for prefix_tokens in tokens:
                tmp_set.add(prefix_tokens[idx])
            if len(tmp_set) == 1:
                longest_prefix_tokens_num += 1
                continue
            elif len(tmp_set) > 1:
                break

    # Adjust the number of prompts to be exactly 220
    final_prompts = None
    if len(prompts) < 220:
        final_prompts = prompts * math.floor(
            220 / len(prompts)) + prompts[:220 % len(prompts)]
    elif len(prompts) > 220:
        final_prompts = prompts[:220]
    else:
        final_prompts = prompts

    # Return the length of the longest common prefix and the final list of prompts
    return longest_prefix_tokens_num, final_prompts


# Example usage of the load_prompts function
if __name__ == "__main__":
    longest_prefix_tokens_num, final_prompts = load_prompts(
        input_csv="prefix_cache_input.csv",
        tokenizer_dir="/model/llama2_ksana_13b/hf_fp16")
    print(longest_prefix_tokens_num)
