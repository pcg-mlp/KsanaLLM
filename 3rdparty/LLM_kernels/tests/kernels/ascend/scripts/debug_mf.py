import os
import numpy as np
import mindspore as ms
from mindformers import init_context
from mindformers import MindFormerConfig, LlamaConfig, TransformerOpParallelConfig, AutoTokenizer, LlamaForCausalLM

yaml_file="/cfs/cfs-owwc340l/tangjian/models/llama-hf/MF_13B/checkpoint_network/rank_0/llama2_13b_predict.yaml"
checkpoint_file="/cfs/cfs-owwc340l/tangjian/models/llama-hf/MF_13B/checkpoint_network/rank_0/ckpt.ckpt"
seq_length=2048
model_type="llama2_13b"
config = MindFormerConfig(yaml_file)
config.use_parallel = False
init_context(use_parallel=config.use_parallel,
              context_config=config.context,
              parallel_config=config.parallel)
inputs=["who are u"]
model_config = LlamaConfig(**config.model.model_config)
model_config.parallel_config = TransformerOpParallelConfig(**config.parallel_config)
model_config.batch_size = len(inputs)
model_config.use_past = True
model_config.seq_length = seq_length
model_config.checkpoint_name_or_path = checkpoint_file
print(f"config is: {model_config}")

tokenizer = AutoTokenizer.from_pretrained(os.path.dirname(checkpoint_file))
# inputs_ids = tokenizer(inputs, max_length=model_config.seq_length, padding="max_length")["input_ids"]
inputs = tokenizer(inputs)
inputs_ids = inputs.input_ids

# inputs_ids = ms.Tensor([[1, 1058, 526, 318]], dtype=ms.int64)
model = LlamaForCausalLM(model_config)
# inputs_ids = np.array([[1, 1058, 526, 318]], dtype=np.int64)
import pdb
pdb.set_trace()
generate_ids = model.generate(inputs_ids,
                          max_length=model_config.max_decode_length,
                          do_sample=model_config.do_sample,
                          top_k=model_config.top_k,
                          top_p=model_config.top_p)
output_text = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(output_text)
# for output in generate_ids:
#     print(tokenizer.decode(output))
