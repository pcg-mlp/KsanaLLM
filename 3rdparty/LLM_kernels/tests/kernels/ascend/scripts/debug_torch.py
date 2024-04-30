
from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer
import torch
SEQ_LEN_IN = 2048

# model_path="/cfs/cfs-owwc340l/tangjian/models/llama2-13b-chat-hf"
model_path="/cfs/cfs-owwc340l/tangjian/models/llama-hf/13B"
model = LlamaForCausalLM.from_pretrained(model_path, device_map="cpu") # torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

question = "who are u"
inputs = tokenizer(question, return_tensors="pt")
inputs_ids = inputs.input_ids
# inputs = tokenizer(question, return_tensors="pt", truncation=True, padding='max_length', max_length=SEQ_LEN_IN)
import pdb
pdb.set_trace()
generate_ids = model.generate(inputs_ids, max_length=30)
output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)

# outputs = model(**inputs)
# predicted_token_id = outputs.logits.argmax(-1)
# predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id[0])
# print(predicted_token)

