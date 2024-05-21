# KsanaLLM

## About

KsanaLLM is a fast and easy-to-use library for LLM inference and serving.

KsanaLLM is fast with:

- State-of-the-art serving throughput.
- Efficient management of attention key and value memory with [PagedAttention](https://arxiv.org/abs/2309.06180).
- Continuous batching of incoming requests.
- Optimized CUDA kernels.
- Introduce high performance kernel from [vllm](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [FastTransformer](https://github.com/NVIDIA/FasterTransformer).

KsanaLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models.
- High-throughput serving with various decoding algorithms, including parallel sampling, beam search, and more.
- Tensor parallelism support for distributed inference.
- Streaming outputs.
- OpenAI-compatible API server.
- Compatible with **NVIDIA GPUs** and **Huawei Ascend NPU**.
- (Experimental) Prefix caching support.

KsanaLLM seamlessly supports many Hugging Face models, including the following architectures:

- LLaMA & LLaMA-2 (`meta-llama/Llama-2-70b-hf`, `lmsys/vicuna-13b-v1.3`, `young-geng/koala`, `openlm-research/open_llama_13b`, etc.)
- Qwen (`Qwen1.5-14B-Chat`)

Supported Hardware

 - Nvidia GPUs: A10, A100
 - Huawei Ascend NPUs: 910B

## Usage

### 1. Create docker container and runtime environment

#### 1.1 For Nvidia GPU

Option 1: Create and set container for **NVIDIA official image**
```bash
# need install nvidia-docker from https://github.com/NVIDIA/nvidia-container-toolkit
sudo nvidia-docker run -itd --network host --privileged \
    nvcr.io/nvidia/pytorch:24.03-py3 bash
pip install -r requirements.txt
# for download huggingface model
apt update && apt install git-lfs -y
```

Option 2: Create and set container for **Tencent image**:
```bash
sudo docker run -it --network host --shm-size=10g --privileged \
    --device /dev/nvidia0 --device /dev/nvidiactl \
    -v /usr/local/nvidia:/usr/local/nvidia mirrors.tencent.com/todacc/venus-numerous-llm:0.1.19 bash
```

#### 1.2 For Huawei Ascend NPU
```bash
sudo docker run -it --network host --shm-size=10g --privileged \
    mirrors.tencent.com/todacc/venus-std-base-tlinux3-npu-llm:0.1.3 bash
```

### 2. Clone source code

```bash
git clone --recurse-submodules https://github.com/Tencent/KsanaLLM
export GIT_PROJECT_REPO_ROOT=`pwd`/KsanaLLM
```

### 3. Compile
```bash
cd ${GIT_PROJECT_REPO_ROOT}
mkdir build && cd build
```

#### 3.1 For Nvidia
```bash
# SM for A10 is 86， change it when using other gpus.
# refer to: https://developer.nvidia.cn/cuda-gpus
cmake -DSM=86 -DWITH_TESTING=ON .. && make -j32
```

#### 3.2 For Huawei Ascend NPU
```bash
cmake -DWITH_TESTING=ON -DWITH_CUDA=OFF -DWITH_ACL=ON .. && make -j32
```

### 4. Run

```bash
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
ln -s ${GIT_PROJECT_REPO_ROOT}/build/lib .

# download huggingface model for example:
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf

# change the model_dir in ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml if needed

# set environment variable `NLLM_LOG_LEVEL=DEBUG` before run to get more log info
# the serving log locate in log/ksana_llm.log

# ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml's tensor_para_size equal the GPUs/NPUs number
export CUDA_VISIBLE_DEVICES=xx

# launch server
python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml \
    --port 8080
```

Inference test with one shot conversation
```bash
# open another session
cd ${GIT_PROJECT_REPO_ROOT}/examples/llama7b
python serving_client.py --port 8080
```

### 5. Distribute

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

### 6. Optional 

#### 6.1 Model Weight Map

You can include an optional weight map JSON file for models that share the same structure as the Llama model but have different weight names.
For more detailed information, please refer to the following link: [Optional Weigth Map Guide](src/ksana_llm/python/weight_map/README.md)
