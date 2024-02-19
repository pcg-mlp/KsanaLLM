# Demo for LLaMA 13B

## 1. Image & Container

 - mirrors.tencent.com/todacc/venus-numerous-llm:0.1.5

or prepare image with Dockerfile.

## 2. Prepare Model

model format is same as FasterTransformer.

```bash
# download demo model
sudo yum install git-lfs -y
git lfs clone https://git.woa.com/karlluo/llama13b-fastertransformer-model.git
# according to examples/llama13b/llama_13b_bs16_in512_out256.ini's [model_dir] section
# we put model file in /model/llama-ft/13B/2-gpu/
mkdir -p /model/llama-ft/13B/
mv llama13b-fastertransformer-model /model/llama-ft/13B/2-gpu
# the files tree is
# ├── config.ini
# ├── model.final_layernorm.weight.bin
# ├── model.layers.x.attention.dense.weight.0.bin
# ├── model.layers.x.attention.dense.weight.1.bin
# ├── model.layers.x.attention.query_key_value.weight.0.bin
# ├── model.layers.x.attention.query_key_value.weight.1.bin
# ├── model.layers.x.input_layernorm.weight.bin
# ├── model.layers.x.mlp.down_proj.weight.0.bin
# ├── model.layers.x.mlp.down_proj.weight.1.bin
# ├── model.layers.x.mlp.gate_proj.weight.0.bin
# ├── model.layers.x.mlp.gate_proj.weight.1.bin
# ├── model.layers.x.mlp.up_proj.weight.0.bin
# ├── model.layers.x.mlp.up_proj.weight.1.bin
# ├── model.layers.x.post_attention_layernorm.weight.bin
# ├── .........
# ├── model.lm_head.weight.bin
# └── model.wte.weight.bin
```

## 3. Launch Server and Load Model

```bash
# after compile following ${GIT_ROOT_PATH}/README.md, server can be launch with following command line
${GIT_ROOT_PATH}/build/bin/ksana_llm --model_config=${GIT_ROOT_PATH}/examples/llama13b/llama_13b_bs16_in512_out256.ini --host=0.0.0.0 --port=8080
```

## 4. Prepare Client Tokenizor

```bash
pip install requests
sudo yum install git-lfs -y
git lfs clone https://huggingface.co/huggyllama/llama-13b
python llama13b_simple_client.py
```
