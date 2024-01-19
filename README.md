# Usage
## Create docker container
```
sudo docker run -itd --name xxx_container_name --network host --privileged   --device /dev/nvidiaxxx --device /dev/nvidiactl   -v /usr/local/nvidia:/usr/local/nvidia mirrors.tencent.com/todacc/venus-numerous-llm:0.1.14 bash
```
- replace xxx_container_name with real container name.
- replace xxx in ```/dev/nvidiaxxx``` with GPU card index. For multiple cards, add ```--device /dev/nvidiaxxx``` in the command
- A test models_test will use model at /model/llama-ft/7B/1-gpu/, add model dir if model exists in local dir.

## Clone source code

```
git clone --recurse-submodules https://git.woa.com/RondaServing/LLM/NumerousLLM.git
```

## Compile and Test
```bash
mkdir build
cd build

# SM for A10 is 86， change it when using other gpus. refer to: https://developer.nvidia.cn/cuda-gpus
cmake -DSM=86 -DWITH_TESTING=ON ..

# Build
make -j

# Set visible devices
export CUDA_VISIBLE_DEVICES=14,15

# install missing package in docker image, TODO: remove after image is fixed
pip install numpy

# create temp dir for test models_test, TODO: remove after test is fixed
mkdir -p /model/llama-ft/7B/nllm
mkdir -p /model/llama-ft/7B/nllm_decode

# test
make test
```

## Run

```bash
# run standalone demo using llama7b
./bin/numerous_llm --model_config ../examples/llama7b/config.ini
# check runing status
# open another terminal or session
cat ./log/numerous_llm.log
```

Start client
```
python ../examples/llama13b/llama13b_simple_client.py
```

## Code format command line

please format your code before submit a merge request

```bash
# prepare clang-format
# pip install clang-format
# prefer clang-format 17.0.5
cd ${GIT_PROJECT_ROOT_DIR}
clang-for
clang-format -i ${CODE_YOUR_EDIT}
```