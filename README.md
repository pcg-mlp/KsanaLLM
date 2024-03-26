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

### Create docker container

```bash
# For NVIDIA GPU
sudo docker run -itd --name xxx_container_name --network host --privileged \
    --device /dev/nvidiaxxx --device /dev/nvidiactl \
    -v /usr/local/nvidia:/usr/local/nvidia mirrors.tencent.com/todacc/venus-numerous-llm:0.1.17 bash

# For Huawei Ascend NPU
sudo docker run -itd --name xxx_container_name --network host --privileged \
    mirrors.tencent.com/todacc/venus-std-base-tlinux3-npu-llm:0.1.1 bash
```
- replace xxx_container_name with real container name.
- replace xxx in ```/dev/nvidiaxxx``` with GPU card index. For multiple cards, add ```--device /dev/nvidiaxxx``` in the command
- A test models_test will use model at /model/llama-hf/7B, add model dir if model exists in local dir.

### Clone source code

```bash
git clone --recurse-submodules https://git.woa.com/RondaServing/LLM/KsanaLLM.git
```

### Compile and Test
```bash
mkdir build
cd build

# SM for A10 is 86， change it when using other gpus. refer to: https://developer.nvidia.cn/cuda-gpus
cmake -DSM=86 -DWITH_TESTING=ON ..

# for Huawei Ascend NPU
cmake -DWITH_TESTING=ON -DWITH_CUDA=OFF -DWITH_ACL=ON ..

# Build
make -j

# Set visible devices
export CUDA_VISIBLE_DEVICES=14,15

# test
make test
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

# download huggingface model for example
wget https://mirrors.tencent.com/repository/generic/pcg-numerous/dependency/numerous_llm_models/llama2_7b_hf.tgz
tar vzxf llama2_7b_hf.tgz

# launch server
python serving_server.py --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm.yaml --tokenizer_dir llama2_7b_hf

# open another session, request client
python serving_client.py
```

#### Distribute

```bash

cd ${GIT_PROJECT_REPO_ROOT}

# for distribute wheel
python setup.py bdist_wheel
# install wheel
pip install dist/ksana_llm-0.1-cp39-cp39-linux_x86_64.whl

# check install success
pip show -f ksana_llm
python -c "import ksana_llm"
```

#### Debug

set environment variable `NLLM_LOG_LEVEL=DEBUG` to get more log info
