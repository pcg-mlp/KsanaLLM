# KsanaLLM

[English](README.md) | [中文](README_cn.md)

## About

KsanaLLM is a high performance and easy-to-use engine for LLM inference and serving.

**High Performance and Throughput:**

- Utilizes optimized CUDA kernels, including high performance kernels from [vLLM](https://github.com/vllm-project/vllm), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM), [FastTransformer](https://github.com/NVIDIA/FasterTransformer), [SGLang](https://github.com/sgl-project/sglang), [LightLLM](https://github.com/ModelTC/lightllm)
- Efficient management of attention key and value memory with [PagedAttention](https://arxiv.org/abs/2309.06180)
- Detailed optimization of task-scheduling and memory-uitlization for dynamic batching 
- Prefix caching support
- Sufficient testing has been conducted on GPU/NPU cards such as A10, A100, L20, L40, H20, 910B2C etc

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
- Qwen 7B/14B & QWen1.5 7B/14B/72B/110B Qwen-VL
- Yi1.5-34B 
- DeepSeek V3/R1

**Supported Hardware**

 - Nvidia GPUs: A10, A100, L40, L20, H20
 - Huawei Ascend NPUs: 910B2C
 - Kunlunxin XPU: P800

## Usage

### 1. Create Docker container and runtime environment

#### 1.1 For Nvidia GPU

```bash
# need install nvidia-docker from https://github.com/NVIDIA/nvidia-container-toolkit
cd docker
docker build -f Dockerfile.gpu -t ksanallm-gpu .
docker run \
    -u root \
    -itd --privileged \
    --shm-size=50g \
    --network host \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --gpus all \
    ksanallm-gpu bash

# goto KsanaLLM root directory 
pip install -r requirements.txt
```

#### 1.2 Direct Use of Internal Nvidia GPU Image 

1. Internal Image Name (For H20 / cuda-124):

```bash
mirrors.tencent.com/todacc/venus-std-base-tlinux4-ksana-hopper-gpu-rdma:0.1.2
```


2. Internal Image Name (For A10 / cuda-118):

```bash
mirrors.tencent.com/todacc/venus-numerous-llm:1.0.0
```

3. Container Creation:
```bash
 # For H20 / cuda-124
 sudo docker run -itd --name xxx -v /data1/:/data1 --network host  --shm-size=10g --privileged --gpus all -v /usr/local/nvidia:/usr/local/nvidia mirrors.tencent.com/todacc/venus-std-base-tlinux4-ksana-hopper-gpu-rdma:0.1.2 /bin/bash

 # For A10 / cuda-118
 sudo docker run -itd --name xxx -v /data1/:/data1 --network host  --shm-size=10g --privileged --gpus all -v /usr/local/nvidia:/usr/local/nvidia mirrors.tencent.com/todacc/venus-numerous-llm:1.0.0 /bin/bash
 ```

#### 1.3 Direct Use of Tencent Cloud Nvidia GPU Image

```bash
# need install nvidia-docker from https://github.com/NVIDIA/nvidia-container-toolkit
cd docker
nvidia-docker build -f Dockerfile.tencentos4.gpu -t ksanallm-gpu .
nvidia-docker run \
    -u root \
    -itd --privileged \
    --shm-size=50g \
    --network host \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    ksanallm-gpu bash

# goto KsanaLLM root directory 
pip install -r requirements.txt
```

#### 1.4 For Huawei Ascend NPU

**Please install Huawei Ascend NPU driver and CANN: [driver download link](https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit)**

**Recommend version: CANN 8.0RC2**

**Only Support Ascend NPU + X86 CPU**

```bash
cd docker
docker build -f Dockerfile.npu -t ksanallm-npu .
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
    ksanallm-npu bash

# install Ascend-cann-toolkit, Ascend-cann-nnal from https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit
# download torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl from https://www.hiascend.com/document/detail/zh/canncommercial/80RC2/softwareinst/instg/instg_0000.html?Mode=PmIns&OS=Ubuntu&Software=cannToolKit
pip3 install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
pip install torch_npu-2.1.0.post6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
pip install -r requirements.txt
```

#### 1.5 For Kunlunxin XPU

**Image:**

| Image | Version |
| - | - |
| [http://mirrors.tencent.com/todacc/venus-ksanallm-dev-tlinux4_kunlun:0.1.1](http://mirrors.tencent.com/todacc/venus-ksanallm-dev-tlinux4_kunlun:0.1.1) | 0.1.1 |

Container creation example:
```bash
CONT_NAME="cont_name"
CONT_IMAGE="mirrors.tencent.com/todacc/venus-ksanallm-dev-tlinux4_kunlun:0.1.1"
PATH_MAPPING="-v /data1:/data1"
sudo docker run -d -it \
       --net="host" \
       --user="root" \
       --name="$CONT_NAME" \
       --shm-size=10g \
       --ulimit memlock=-1 \
       --ulimit stack=67108864 \
       --privileged=true \
       --cap-add=ALL \
       $(for i in $(seq 0 7); do echo -n "--device=/dev/xpu${i}:/dev/xpu${i} "; done) \
       --device=/dev/xpuctrl:/dev/xpuctrl \
       --security-opt seccomp:unconfined \
       --security-opt label=disable \
       $PATH_MAPPING \
       $CONT_IMAGE \
       /bin/bash
```

### 2. Clone source code

```bash
git clone --recurse-submodules https://github.com/pcg-mlp/KsanaLLM
export GIT_PROJECT_REPO_ROOT=`pwd`/KsanaLLM
```

### 3. Compile

```bash
cd ${GIT_PROJECT_REPO_ROOT}
pip install -r requirements.txt
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

#### 3.3 For Kunlunxin XPU

```bash
cmake -DWITH_KUNLUN=ON -DWITH_TESTING=OFF -DWITH_STANDALONE_TEST=OFF -DCMAKE_BUILD_TYPE=Release .. && make -j
```

#### 3.4 Enable use of Tencent internal libraries, for example, Tencent nameserver Polaris

```bash
cmake -DSM=86 -DWITH_TESTING=ON -DWITH_INTERNAL_LIBRARIES=ON .. && make -j32
```

#### 3.5 Enable Global Cache Connector (Mooncake Store Backend)

Global Cache Connector enables RDMA-based global KV cache access using [Mooncake Store](https://github.com/kvcache-ai/Mooncake) as the storage backend.

**Prerequisites: Build Mooncake v0.3.9**

```bash
git clone https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
git checkout v0.3.9

mkdir build && cd build
cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DBUILD_TESTING=OFF -DENABLE_TESTS=OFF
make -j$(nproc)
```

**Build KsanaLLM with Mooncake real store enabled**

```bash
cd ${GIT_PROJECT_REPO_ROOT}/build

cmake -DSM=90a -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MOONCAKE_REAL_STORE=ON \
    -DMOONCAKE_SOURCE_ROOT=/path/to/Mooncake \
    .. && make -j$(nproc)
```

| CMake Variable | Required | Description |
|---|---|---|
| `USE_MOONCAKE_REAL_STORE` | Yes | Set `ON` to enable the real Mooncake Store backend (default `OFF`, uses mock backend) |
| `MOONCAKE_SOURCE_ROOT` | Yes | Path to Mooncake source root (for headers). Must be checked out at tag `v0.3.9` — CMake verifies the git tag and will error on mismatch |
| `MOONCAKE_BUILD_ROOT` | No | Path to Mooncake build directory (for libraries). Defaults to `${MOONCAKE_SOURCE_ROOT}/build` |

**Standalone build (without full KsanaLLM)**

To build `libglobal_cache_connector.so` independently:

```bash
mkdir build && cd build

# Mock backend only
cmake ../csrc/global_cache_connector/ \
    -DCMAKE_TOOLCHAIN_FILE= \
    -DSTANDALONE_BUILD=ON
make -j$(nproc)

# With real Mooncake Store backend
cmake ../csrc/global_cache_connector/ \
    -DUSE_MOONCAKE_REAL_STORE=ON \
    -DMOONCAKE_SOURCE_ROOT=/path/to/Mooncake
make -j$(nproc)
```

**Enable Global Cache at runtime**

Build alone is not enough. Serving must turn on the hook, configure the connector in yaml, and align Mooncake ports.

1. **Environment**

```bash
export ENABLE_GLOBAL_CACHE_HOOK=1
# Optional: absolute path to mooncake_master if it is not on PATH
# export MOONCAKE_MASTER_BIN=/path/to/Mooncake/build/mooncake-store/src/mooncake_master
```

When `ENABLE_GLOBAL_CACHE_HOOK=1`, serving auto-starts a local `mooncake_master` if the RPC port is free (skips if already listening).

2. **Yaml** (`setting.global_cache_connector`)

- Set a non-empty `service_endpoint` (empty → connector disabled).
- Use event-driven scheduler: `setting.batch_scheduler.scheduler_type: 1`.
- Keep `service_endpoint` / `master_server_addr` ports consistent with serving CLI (defaults `8080` / `50051`).

Minimal example (see full commented block in `examples/ksana_llm.yaml`):

```yaml
setting:
  batch_scheduler:
    scheduler_type: 1
  global_cache_connector:
    service_endpoint: "http://127.0.0.1:8080/metadata"
    master_server_addr: "127.0.0.1:50051"
    backend_type: "mooncake"   # or mooncake-mock
    protocol: "tcp"            # tcp | rdma
    # rdma_devices: ""         # empty = auto; for RDMA set e.g. mlx5_0
```

3. **Launch serving**

```bash
export ENABLE_GLOBAL_CACHE_HOOK=1
python serving_server.py \
    --config_file /path/to/your_config.yaml \
    --port 8080 \
    --mooncake-http-port 8080 \
    --mooncake-rpc-port 50051
```

| Item | Role |
|---|---|
| `ENABLE_GLOBAL_CACHE_HOOK=1` | Enable query/put hook; may auto-start local master |
| `service_endpoint` non-empty | Construct / enable Global Cache Connector |
| `scheduler_type: 1` | Required with the global-cache hook path |
| `--mooncake-http-port` / `--mooncake-rpc-port` | Must match yaml `service_endpoint` / `master_server_addr` |

More env vars and field notes: [`docs/ksana_env_guide.md`](docs/ksana_env_guide.md).

### 4. Run

#### 4.1 Single

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

# For Kunlunxin XPU, use either CUDA_VISIBLE_DEVICES or XPU_VISIBLE_DEVICES to control the number of cards
# export XPU_VISIBLE_DEVICES=0

# launch server
python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml \
    --port 8080
# KsanaLLM now supports the OpenAI API protocol and has implemented the core /v1/chat/completions API. 
# you can add specific parameters to enable tool invocation parsing and inference content analysis capabilities.
# tool-choice:
--enable-auto-tool-choice     --tool-call-parser deepseek_v3  \
# reasoning-parser:
--reasoning-parser  deepseek_r1 \
# apply specific chat-template:
--chat-template openaiapi/chat_templates/tool_chat_template_deepseekr1.jinja

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

Test performance of the model
```bash
cd ${GIT_PROJECT_REPO_ROOT}/build 
./bin/run_model_performance --runtime-config ${GIT_PROJECT_REPO_ROOT}/examples/llama7b/ksana_llm_tp.yaml --perf-config ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/model_performance/test_config.csv 

# enable nsys when using Cuda
export ENABLE_PROFILE_EVENT=1 # enale profile event like NVTX on Cuda
nsys profile ./bin/run_model_performance --runtime-config ${GIT_PROJECT_REPO_ROOT}/examples/llama7b/ksana_llm_tp.yaml --perf-config ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/model_performance/test_config.csv 
unset ENABLE_PROFILE_EVENT # after using nsys
```

#### 4.2 Distributed

Distributed execution depends on the following environment variables:
WORLD_SIZE: Number of nodes, i.e., number of inference processes, which can be on the same machine or across machines. If undefined or the value is 1, it is not distributed mode.
NODE_RANK: The rank of the current node, starting from 0, with 0 being the master node.
MASTER_HOST: The IP address of the master node in the inference cluster.
MASTER_PORT: The management port of the master node in the inference cluster.

Below, using IP1 and IP2, with the master node deployed on IP1 and listening on port_1, demonstrates the command for dual-machine execution.
```bash
# on IP1
export WORLD_SIZE=2
export NODE_RANK=0
export MASTER_HOST=IP1
export MASTER_PORT=port_1

python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml \
    --port 8080


# on IP2
export WORLD_SIZE=2
export NODE_RANK=1
export MASTER_HOST=IP1
export MASTER_PORT=port_1

python serving_server.py \
    --config_file ${GIT_PROJECT_REPO_ROOT}/examples/ksana_llm2-7b.yaml \
    --port 8080
```

Note: By default, NCCL communication is used. If you want to force TCP communication, you can add the following environment variable:
export USE_TCP_DATA_CHANNEL=1

#### 4.3 Example of Running the DeepSeek Model on H20

**Suggestion**: Use the internal NVIDIA image from section 1.2

##### 4.3.1 Compilation for NVIDIA H20

```bash
cmake -DSM=90a -DCMAKE_BUILD_TYPE=Release ..  && make -j

cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
ln -s ${GIT_PROJECT_REPO_ROOT}/build/lib .
```

##### 4.3.2 Dual-Node 16-GPU Execution Example (Using the [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) Model)
Parallelization Strategy: Inter-node Pipeline Parallelism and Intra-node Tensor Parallelism—Optimal Performance Configuration as Follows

1. Execution on IP1 Node (Master Node):
```bash
# Set the IP1 node as the master node.
export WORLD_SIZE=2
export NODE_RANK=0
export MASTER_HOST=master_node_ip
export MASTER_PORT=master_node_port
# Optimal environment variable configuration
export ENABLE_COMPRESSED_KV=2
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export USE_TCP_DATA_CHANNEL=1
export MASTER_OFFLOAD_LAYER_NUM=0
# Service Startup
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_fp8_perf.yaml \
--port service_port
```

2. Execution on IP2 Node (Work Node):
```bash
# Set the IP2 node as the work node.
export WORLD_SIZE=2
export NODE_RANK=1
export MASTER_HOST=master_node_ip
export MASTER_PORT=master_node_port
# Optimal environment variable configuration
export ENABLE_COMPRESSED_KV=2
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
export USE_TCP_DATA_CHANNEL=1
# Service Startup
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_fp8_perf.yaml \
--port service_port
```
Note: In the current version, when using the multi-batch feature (i.e., setting max_pp_batch_num=2 in deepseek_fp8_perf.yaml), 
inter-node communication must be conducted via TCP (by exporting USE_TCP_DATA_CHANNEL=1). NCCL-based communication for multi-batch will be supported in future releases.

3. Improving Service Startup Speed (Optional):

If you find that the service startup is too slow, you can accelerate the process by configuring the following environment variables to generate a cached model. 
Upon subsequent service startups, the cached model will be loaded, thereby reducing startup latency.
```bash
export ENABLE_MODEL_CACHE=1
export MODEL_CACHE_PATH=/xxx_cache_model_dir/
```
Note: Both the generation and utilization of the cached model require the above environment variables to be set. Additionally, these configurations must be applied on every node.

##### 4.3.3 Single-Node 8-GPU Execution of the DeepSeek-R1-0528-GPTQ-int4 Model

```bash
# Optimal environment variable configuration
export ENABLE_COMPRESSED_KV=2
export SELECT_ALL_REDUCE_BY_SIZE=1
export PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync
# (Optional) Further Performance Enhancement with Slight Accuracy Degradation
export W4AFP8_MOE_BACKEND=1
# Service Startup
python serving_server.py \
--config_file ${GIT_PROJECT_REPO_ROOT}/examples/deepseek_int4_perf.yaml \
--port service_port
```
Note: The startup of int4 models can also be accelerated by configuring the environment variables for generating cached models as described in Section 4.3.2.

##### 4.3.4 Performance Stress Testing (General)
```bash
python ${GIT_PROJECT_REPO_ROOT}/benchmarks/benchmark_throughput.py \
--host master_node_ip \
--port service_port \
--prompt_num 512 \
--input_csv xxx_dataset.csv \
--stream \
--backend ksana \
--model_type deepseek_r1/deepseek_v3 \
--mode async \
--request_rate xx_qps \
--output_csv output_res.csv \
--perf_csv perf_res.csv
# The request_rate parameter controls the rate at which requests are sent. By default, it is set to "inf" (all requests are sent simultaneously).
```

#### 4.4 Starting in PD Disaggregation Mode

###### PD Node Configuration
Add the following content after ```endpoint_type: python``` in the `examples/ksana_llm.yaml` configuration file:
- Make sure ```connector``` is aligned with ```endpoint_type```
- Set ```group_role``` to ```decode``` or ```prefill``` according to the service role
- Note: chunk prefill must be disabled, i.e. set (split_fuse_token_num: 0)

**Decode configuration example**
```
# Disable chunk prefill
split_fuse_token_num: 0
```

```
connector:
    backend: "v2"      # Indicates the v2 version of PD disaggregation
    group_role: decode # Role: decode or prefill
    cluster_name: dev_cluster # The cluster_name must be the same in a cluster with N prefill and M decode nodes
    inference_addr: 127.0.0.1:8089  # Inference address of the current node
    redis_password: "XXXXXXX"       # Redis password
    metadata_addr: "redis://127.0.0.1:6379" # Redis address
```

**Prefill configuration example**
```
# Disable chunk prefill
split_fuse_token_num: 0
```

```
connector:
    backend: "v2"      # Indicates the v2 version of PD disaggregation
    group_role: prefill # Role: decode or prefill
    cluster_name: dev_cluster # The cluster_name must be the same in a cluster with N prefill and M decode nodes
    inference_addr: 127.0.0.1:8090  # Inference address of the current node
    redis_password: "XXXXXXX"       # Redis password
    metadata_addr: "redis://127.0.0.1:6379" # Redis address
```

###### Node Startup Commands
- Note: Different node roles should point to the corresponding configuration files prepared above
- Note: When starting multiple roles on a single node, configure a different Mooncake RPC address for each node

**Decode node startup command example**
```
# Use the Mooncake Transfer Engine Next version
export MC_USE_TENT=1
# When starting both roles on the same machine, set a different Mooncake RPC address for each node
export MC_TENT_CONF='{"rpc_server_hostname": "127.0.0.1", "rpc_server_port":17071}'
# Service startup command. Make sure it points to the configuration file for the corresponding role
python ../python/serving_server.py \
  --config_file ksana_llm_decode.yaml \
  --host 0.0.0.0 \
  --port 8089
```

**Prefill node startup command example**
```
# Use the Mooncake Transfer Engine Next version
export MC_USE_TENT=1
# When starting both roles on the same machine, set a different Mooncake RPC address for each node
export MC_TENT_CONF='{"rpc_server_hostname": "127.0.0.1", "rpc_server_port":17072}'
# Service startup command. Make sure it points to the configuration file for the corresponding role
python ../python/serving_server.py \
  --config_file ksana_llm_prefill.yaml \
  --host 0.0.0.0 \
  --port 8090
```

### 5. Distribute

```bash
cd ${GIT_PROJECT_REPO_ROOT}

# for distribute wheel
# By default the wheel version is a fixed base "0.0.0" plus the latest short git
# commit id as a PEP 440 local version (e.g. 0.0.0+g26efdb17). It does NOT derive
# from git tags, so a stale tag set on the build machine can never leak a wrong
# version. The local version does not affect installation or dependency
# resolution, but keeps the exact commit visible via `pip show ksana_llm`.
python setup.py bdist_wheel

# equivalent to the default: fixed base version + commit id (no git tag)
python setup.py bdist_wheel --with-commit-id

# use the git tag version only, without the commit id (e.g. v0.8.1.post3)
python setup.py bdist_wheel --with-tag

# override the CUDA target architectures (defaults to 80,86,89,90a)
python setup.py bdist_wheel --with-sm 90a,89

# or build with other cmake args
export CMAKE_ARGS="
  -DWITH_CUDA=ON
  -DWITH_ACL=OFF
" && python setup.py bdist_wheel

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

Custom plugins can perform some special pre-processing and post-processing tasks. You need to place your `ksana_plugin.py` in the
model directory.

You should implement a `KsanaPlugin` class with three optional methods:
`init_plugin(self, **kwargs)`, `preprocess(self, **kwargs)` and `postprocess(self, **kwargs)`.

- `init_plugin` is called during plugin initialization
- `preprocess` is called at the start of each request (e.g., ViT inference)
- `postprocess` is called at the end of each request (e.g., PPL calculation)

See [Example](src/ksana_llm/python/ksana_plugin/qwen_vl/ksana_plugin.py) for more details.

#### 6.3 KV Cache Scaling Factors

When enabling FP8 E4M3 KV Cache quantization, it is necessary to provide scaling factors to ensure inference accuracy.

For more detailed information, please refer to the following link: [Optional KV Scale Guide](src/ksana_llm/python/kv_scale_files/README.md)

#### 7. Contact Us
##### WeChat Group
<img src=docs/img/webchat-github.jpg width="200px">

