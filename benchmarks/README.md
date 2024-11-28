# Test Set Description
 - share_gpt_500.csv: Contains 500 data records, which are randomly selected  from 
  the original [ShareGPT dataset](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered) with seed of 0.

# Download model
```
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python

# download huggingface model for example:
# Note: Make sure git-lfs is installed.
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf

```

# Ksana
## Start server
```
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python

export CUDA_VISIBLE_DEVICES=xx

# launch server
python serving_server.py \
    --config_file ../../../examples/ksana_llm2-7b.yaml \
    --port 8080
```
Change config file when trying other options

## Start benchmark
```
cd ${GIT_PROJECT_REPO_ROOT}/benchmarks

# benchmark
python benchmark_throughput.py --input_csv benchmark_input.csv --port 8080 \
    --backend ksana --model_type llama \
    --perf_csv ksana_perf.csv > ksana_stdout.txt 2>&1
```

# vLLM
## Start server
```
export MODEL_PATH=${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python/Llama-2-7b-hf
export CUDA_VISIBLE_DEVICES=xx

python -m vllm.entrypoints.api_server \
     --model $MODEL_PATH \
     --tokenizer $MODEL_PATH \
     --trust-remote-code \
     --max-model-len 1536 \
     --pipeline-parallel-size 1 \
     --tensor-parallel-size 1 \
     --gpu-memory-utilization 0.94 \
     --disable-log-requests \
     --port 8080 
```

## Start benchmark
```
python benchmark_throughput.py --port 8080  --input_csv benchmark_input.csv  \
    --model_type llama \
    --tokenizer_path $MODEL_PATH  \
    --backend vllm \
    --perf_csv vllm_perf.csv > vllm_stdout.txt 2>&1
```

