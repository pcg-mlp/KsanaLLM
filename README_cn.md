# 一念LLM (KsanaLLM)

[English](README.md) [中文](README_cn.md)

## 介绍

**一念LLM** 是面向LLM推理和服务的高性能和高易用的推理引擎。

**高性能和高吞吐:**

- 使用极致优化的 CUDA kernels, 包括来自 [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [FastTransformer](https://github.com/NVIDIA/FasterTransformer) 等工作的高性能算子
- 基于 [PagedAttention](https://arxiv.org/abs/2309.06180) 实现地对注意力机制中key和value的高效显存管理
- 对任务调度和显存占用精细调优的动态batching
- (实验版) 支持前缀缓存(Prefix caching)
- 在A10, A100和L40等卡上做了较充分的验证测试

**灵活易用:**

- 能够无缝集成流行的 Hugging Face 格式模型，支持 PyTorch 和 SafeTensor 两种权重格式

- 能够实现高吞吐服务，支持多种解码算法，包括并行采样、beam search 等

- 支持多卡间的 tensor 并行 

- 支持流式输出

- 支持 OpenAI-compatible API server

- 支持英伟达 GPU 和华为昇腾 NPU


**一念LLM 支持 Hugging Face 的很多流行模型，下面是经过验证测试的模型:**

- LLaMA 7B/13B & LLaMA-2 7B/13B & LLaMA3 8B/70B
- Baichuan1 7B/13B & Baichuan2 7B/13B
- Qwen 7B/14B & QWen1.5 7B/14B/72B/110B
- Yi1.5-34B

**支持的硬件**

 - Nvidia GPUs: A10, A100, L40
 - Huawei Ascend NPUs: 910B

## 使用

### 1. 创建 Docker 容器和运行时环境

#### 1.1 英伟达 GPU

```bash
# need install nvidia-docker from https://github.com/NVIDIA/nvidia-container-toolkit
sudo nvidia-docker run -itd --network host --privileged \
    nvcr.io/nvidia/pytorch:24.03-py3 bash
pip install -r requirements.txt
# for download huggingface model
apt update && apt install git-lfs -y
```

#### 1.2 华为昇腾 NPU

[https://ascendhub.huawei.com/#/detail/mindie](https://ascendhub.huawei.com/#/detail/mindie)
version: 1.0.RC1-800I-A2-aarch64

### 2. 下载源码

```bash
git clone --recurse-submodules https://github.com/pcg-mlp/KsanaLLM
export GIT_PROJECT_REPO_ROOT=`pwd`/KsanaLLM
```

### 3. 编译

```bash
cd ${GIT_PROJECT_REPO_ROOT}
mkdir build && cd build
```

#### 3.1 英伟达 GPU

```bash
# SM for A10 is 86， change it when using other gpus.
# refer to: https://developer.nvidia.cn/cuda-gpus
cmake -DSM=86 -DWITH_TESTING=ON .. && make -j32
```

#### 3.2 华为昇腾 NPU

```bash
cmake -DWITH_TESTING=ON -DWITH_CUDA=OFF -DWITH_ACL=ON .. && make -j32
```

### 4. 执行

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

基于one shot对话的推理测试 

```bash
# open another session
cd ${GIT_PROJECT_REPO_ROOT}/examples/llama7b
python serving_generate_client.py --port 8080
```

forward推理测试（单轮推理，无采样）

```bash
python serving_forward_client.py --port 8080
```

### 5. 分发

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

### 6. 可选

#### 6.1 模型权重映射

在支持新模型时，如果模型结构与已知模型（例如Llama）相同，只是权重名字不同，可以通过JSON文件来对权重做一个映射，从而能够较简单的支持新模型。想要获取更详细的信息，请参考: [Optional Weigth Map Guide](src/ksana_llm/python/weight_map/README.md)。

#### 6.2 自定义插件

自定义插件可以做特殊预处理和后处理。使用时，你需要把ksana_plugin.py放在模型目录下。
[样例](examples/qwenvl/ksana_plugin.py)

#### 7. 联系我们
##### 微信群
<img src=doc/img/webchat-github.jpg width="200px">
