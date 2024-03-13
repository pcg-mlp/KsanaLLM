import csv
import math
from transformers import LlamaTokenizer


def load_prompts(tokenizer_dir="/dockerdata/karlluo/llama2_ksana_13b/hf_fp16"):
    header = {}
    tokens = []
    prompts = []
    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_dir)

    with open('prefix_cache_input.csv', "r") as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for idx, row in enumerate(spamreader):
            if idx == 0:
                # skip file description
                continue

            if idx == 1:
                # project of header and row index
                for h_idx, header_name in enumerate(row):
                    header[header_name] = h_idx
                continue

            is_match_prefix_token = r"高启强" in row[header["raw_input"]]

            if is_match_prefix_token:
                raw_message = row[header["raw_input"]].strip('"').strip(
                    "\"}], 'stream': False, 'temperature': 0.5, 'max_length': 8192, 'top_p': 1.0, 'delete_prompt_from_output': 1}"
                ).strip("{'messages': [{'role': 'user', 'content': \"")
                prompt = "[INST]%s[/INST]" % raw_message
                token_list = tokenizer.encode(prompt, add_special_tokens=True)
                
                prompts.append(prompt)
                tokens.append(token_list)

    # bad case
    skip_list = [15, 17, 22]
    for skip_idx in skip_list:
        del prompts[skip_idx]
        del tokens[skip_idx]

    # calculate the longest public prefix tokens number
    longest_prefix_tokens_num = 0
    tokens_number_list = [len(t) for t in tokens]
    tokens_min_len = min(tokens_number_list)
    for idx in range(tokens_min_len):
        tmp_set = set()
        for prefix_tokens in tokens:
            tmp_set.add(prefix_tokens[idx])
        if len(tmp_set) == 1:
            longest_prefix_tokens_num += 1
            continue
        elif len(tmp_set) > 1:
            break

    final_prompts = None
    if len(prompts) < 220:
        final_prompts = prompts * math.floor(220 / len(prompts)) + prompts[:(
            220 % len(prompts))]
    elif len(prompts) > 220:
        final_prompts = prompts[:220]
    else:
        final_prompts = prompts

    return longest_prefix_tokens_num, final_prompts


if __name__ == "__main__":
    longest_prefix_tokens_num, final_prompts = load_prompts("/dockerdata/karlluo/llama2_ksana_13b/hf_fp16")
    print(longest_prefix_tokens_num)
