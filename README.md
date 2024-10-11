# KsanaLLM

[English](README.md) | [中文](README_cn.md)

## About

KsanaLLM is a high performance and easy-to-use engine for LLM inference and serving.

**High Performance and Throughput:**

- Utilizes optimized CUDA kernels, including high performance kernels from [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [FastTransformer](https://github.com/NVIDIA/FasterTransformer)
- Efficient management of attention key and value memory with [PagedAttention](https://arxiv.org/abs/2309.06180)
- Detailed optimization of task-scheduling and memory-uitlization for dynamic batching 
- (Experimental) Prefix caching support
- Sufficient testing has been conducted on GPU cards such as A10, A100, L40, etc

**Flexibility and easy to use:**

- Seamless integration with popular Hugging Face models, and support multiple weight formats, such as pytorch and SafeTensors

- High-throughput serving with various decoding algorithms, including parallel sampling, beam search, and more

- Enables multi-gpu tensor parallelism 

- Streaming outputs

- OpenAI-compatible API server

- Support NVIDIA GPUs and Huawei Ascend NPU

  

**KsanaLLM seamlessly supports many Hugging Face models, including the below models that have been verified:**

- LLaMA 7B/13B & LLaMA-2 7B/13B & LLaMA3 8B/70B
- Baichuan1 7B/13B & Baichuan2 7B/13B
- Qwen 7B/14B & QWen1.5 7B/14B/72B/110B
- Yi1.5-34B 

**Supported Hardware**

 - Nvidia GPUs: A10, A100, L40, L20
 - Huawei Ascend NPUs: 910B2C

## Usage

### 1. Create Docker container and runtime environment

#### 1.1 For Nvidia GPU

```bash
# need install nvidia-docker from https://github.com/NVIDIA/nvidia-container-toolkit
sudo nvidia-docker run -itd --network host --privileged \
    nvcr.io/nvidia/pytorch:24.03-py3 bash
pip install -r requirements.txt
# for download huggingface model
apt update && apt install git-lfs -y
```

#### 1.2 For Huawei Ascend NPU

**Please install Huawei Ascend NPU driver and CANN: [driver download link](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)**

**Recommend version: CANN 8.0RC2**

**Only Support Ascend NPU + X86 CPU**

```bash
cd docker
docker build -f Dockerfile.npu -t ksana-npu .
docker run \
    -u root \
    -itd --privileged \
    --shm-size=50g \
    --network host \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp:unconfined $(find /dev/ -regex ".*/davinci$" | awk '{print " --device "$0}') \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /usr/local/sbin/:/usr/local/sbin/ \
    -v /var/log/npu/conf/slog/slog.conf:/var/log/npu/conf/slog/slog.conf \
    -v /var/log/npu/slog/:/var/log/npu/slog \
    -v /var/log/npu/profiling/:/var/log/npu/profiling \
    -v /var/log/npu/dump/:/var/log/npu/dump \
    -v /var/log/npu/:/usr/slog \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    ksana-npu bash

# install Ascend-cann-toolkit, Ascend-cann-nnal from https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit
# download torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl from https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit
pip3 install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install -r requirements.txt
```

### 2. Clone source code

```bash
git clone --recurse-submodules https://github.com/pcg-mlp/KsanaLLM
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
# Note: Make sure git-lfs is installed.
git clone https://huggingface.co/NousResearch/Llama-2-7b-hf

# change the model_dir in ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml if needed

# set environment variable `KLLM_LOG_LEVEL=DEBUG` before run to get more log info
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
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
python serving_generate_client.py --port 8080
```

Inference test with forward(Single round inference without generate sampling)

```bash
python serving_forward_client.py --port 8080
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

For more detailed information, please refer to the following link: [Optional Weight Map Guide](src/ksana_llm/python/weight_map/README.md)

#### 6.2 Plugin

Custom plugins can perform some special pre-process and post-processing. You need to place ksana_plugin.py in the model directory.
[Example](examples/qwenvl/ksana_plugin.py)

#### 6.3 KV Cache Scaling Factors

When enabling FP8 E4M3 KV Cache quantization, it is necessary to provide scaling factors to ensure inference accuracy.

For more detailed information, please refer to the following link: [Optional KV Scale Guide](src/ksana_llm/python/kv_scale_files/README.md)

#### 7. Contact Us
##### WeChat Group
<img src=doc/img/webchat-github.jpg width="200px">

