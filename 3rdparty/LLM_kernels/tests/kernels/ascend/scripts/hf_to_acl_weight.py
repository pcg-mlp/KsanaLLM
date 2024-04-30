import os
import torch
import json
import shutil
import glob
# from transformers import LlamaTokenizer, AutoTokenizer

# hf_model_base_path="/cfs/cfs-owwc340l/tangjian/models/llama2-13b-chat-hf"
hf_model_base_path="/cfs/cfs-owwc340l/tangjian/models/llama-hf/13B"
output_dir="../acl_llama13b/"
json_file =os.path.join(hf_model_base_path, "pytorch_model.bin.index.json")
with open(json_file) as f:
  model_config_json = json.load(f)

weight_map_bin_to_tensor = dict() # map["xxx.bin" : torch.tensor]
weight_map_name_to_tensor= dict() # map["weigth_name" : torch.tensor]

weight_map_name_to_bin = model_config_json["weight_map"]
for weight_name in weight_map_name_to_bin:
  bin_filename = weight_map_name_to_bin.get(weight_name)
  if bin_filename not in weight_map_bin_to_tensor:
    weight_map_bin_to_tensor[bin_filename] = torch.load(os.path.join(hf_model_base_path, bin_filename), map_location=torch.device('cpu'))
  weight_tensors_map = weight_map_bin_to_tensor.get(bin_filename)
  weight_tensor = weight_tensors_map.get(weight_name)
  weight_map_name_to_tensor[weight_name] = weight_tensor
  print(f"weight_name: {weight_name}, shape {weight_tensor.shape}, dtype {weight_tensor.dtype}")

os.makedirs(output_dir, exist_ok=True)
for weight_name in weight_map_name_to_tensor:
  np_tensor = weight_map_name_to_tensor[weight_name].to(torch.float16).numpy()
  if len(np_tensor.shape) > 1 and weight_name not in ["model.embed_tokens.weight"]:
    np_tensor = np_tensor.transpose()
  np_tensor.tofile(os.path.join(output_dir, weight_name + ".bin"))


def copy_files_to_directory(file_pattern, target_directory):
    files = glob.glob(file_pattern)
    os.makedirs(target_directory, exist_ok=True)
    for file in files:
        shutil.copy(file, target_directory)

copy_files_to_directory(hf_model_base_path +"/tokenizer*", output_dir)
