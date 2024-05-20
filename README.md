# KsanaLLM

## About

KsanaLLM is a fast and easy-to-use library for LLM inference and serving.

KsanaLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with **PagedAttention**
- Continuous batching of incoming requests
- Fast model execution with CUDA graph
- Optimized CUDA kernels
- Introduce high performance kernel from vllm, TensorRT-LLM, FastTransformers.

KsanaLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support NVIDIA GPUs and Huawei Ascend NPU
- (Experimental) Prefix caching support

KsanaLLM seamlessly supports many Hugging Face models, including the following architectures:

- LLaMA & LLaMA-2 (`meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- Qwen (`Qwen1.5-14B-Chat`)

## Usage

### Create docker container and runtime environment
Notes: Replace ```xxx_container_name``` with real container name.


#### For Nvidia GPU

Option 1: Create and set container for tencent image:
```bash

sudo docker run -itd --name xxx_container_name --network host --shm-size=10g --privileged \
    --device /dev/nvidiaxxx --device /dev/nvidiactl \
    -v /usr/local/nvidia:/usr/local/nvidia mirrors.tencent.com/todacc/venus-numerous-llm:0.1.19 bash

# Set evironment in the container
export LD_LIBRARY_PATH=/data/miniconda3/envs/env-3.9.2/lib/python3.9/site-packages/torch/lib:/usr/local/cuda/compat/lib:/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/lib64/:/usr/local/nvidia/lib64:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/cuda/extras/CUPTI/lib64/:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib/stubs/:/usr/lib/nccl/:$LD_LIBRARY_PATH
```


Option 2: Create and set container for NVIDIA official image
```bash
sudo nvidia-docker run -itd --name xxx_container_name --network host --privileged \
    nvcr.io/nvidia/pytorch:24.03-py3 bash
pip install -r requirements.txt
```
Notes: Replace xxx in ```/dev/nvidiaxxx``` with GPU card index. For multiple cards, add ```--device /dev/nvidiaxxx``` in the command

#### For Huawei Ascend NPU
```bash
sudo docker run -itd --name xxx_container_name --network host --shm-size=10g --privileged \
    mirrors.tencent.com/todacc/venus-std-base-tlinux3-npu-llm:0.1.3 bash
```

### Clone source code

```bash
git clone --recurse-submodules https://xxx/KsanaLLM.git
```
Notes: Replace https://xxx/KsanaLLM.git with real repo url.

### Compile and Test
```bash
mkdir build
cd build
```
For Nvidia
```bash
# SM for A10 is 86， change it when using other gpus. refer to: https://developer.nvidia.cn/cuda-gpus
cmake -DSM=86 -DWITH_TESTING=ON ..
# Set visible devices
export CUDA_VISIBLE_DEVICES=14,15
```

For Huawei Ascend NPU
```bash
cmake -DWITH_TESTING=ON -DWITH_CUDA=OFF -DWITH_ACL=ON ..
```

Build and test
```bash
# Build
make -j

# test
make test
# models_test will use model at /model/llama-hf/7B, add model dir if model exists in local dir.
# llm_kernels_nvidia_kernel_rotary_embedding_test deps on groundtruth data which is not included in this repo, this test will fail.
```

### Run with python serving server

#### Develop

 1. develop with local C++ style (Recommanded)

```bash

mkdir -p ${GIT_PROJECT_REPO_ROOT}/build
cd ${GIT_PROJECT_REPO_ROOT}/build
# SM86 is for A10 GPU, change what you need
cmake -DSM=86 -DWITH_TESTING=ON ..
make -j
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
ln -s ${GIT_PROJECT_REPO_ROOT}/build/lib .
```

then you can go to 3 to launch local serving server

 2. develop with pythonic style

```bash

cd ${GIT_PROJECT_REPO_ROOT}
python setup.py build_ext
python setup.py develop

# when you change C++ code, you need run
python setup.py build_ext
# to re-compile binary code
```

then you can go to 3 to launch local serving server

 3. launch local serving server

```bash
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python

# download huggingface model for example:
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf

# change the model_dir in ${GIT_PROJECT_REPO_ROOT}/examples/llama7b/ksana_llm.yaml

# launch server
# using model default tokenizer
python serving_server.py --config_file ${GIT_PROJECT_REPO_ROOT}/examples/llama7b/ksana_llm.yaml
# or using different tokenizer
python serving_server.py --config_file ${GIT_PROJECT_REPO_ROOT}/examples/llama7b/ksana_llm.yaml \
    --tokenizer_dir ${DIR_PATH_TO_MODEL}/Llama-2-7b-chat-hf

# open another session, request client
cd ${GIT_PROJECT_REPO_ROOT}/examples/llama7b
python serving_client.py
```

#### Distribute

```bash

cd ${GIT_PROJECT_REPO_ROOT}

# for distribute wheel
python setup.py bdist_wheel
# install wheel
pip install dist/ksana_llm-0.1-*-linux_x86_64.whl

# check install success
pip show -f ksana_llm
python -c "import ksana_llm"
```

#### Debug

set environment variable `NLLM_LOG_LEVEL=DEBUG` to get more log info

#### Optional Model Weight Map

You can include an optional weight map JSON file for models that share the same structure as the Llama model but have different weight names.
For more detailed information, please refer to the following link: [Optional Weigth Map Guide](src/ksana_llm/python/weight_map/README.md)
