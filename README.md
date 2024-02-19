# Usage

## Create docker container

```bash
sudo docker run -itd --name xxx_container_name --network host --privileged \
    --device /dev/nvidiaxxx --device /dev/nvidiactl \
    -v /usr/local/nvidia:/usr/local/nvidia mirrors.tencent.com/todacc/venus-numerous-llm:0.1.17 bash
```
- replace xxx_container_name with real container name.
- replace xxx in ```/dev/nvidiaxxx``` with GPU card index. For multiple cards, add ```--device /dev/nvidiaxxx``` in the command
- A test models_test will use model at /model/llama-ft/7B/1-gpu/, add model dir if model exists in local dir.

## Clone source code

```bash
git clone --recurse-submodules https://git.woa.com/RondaServing/LLM/KsanaLLM.git
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

# create temp dir for test models_test, TODO: remove after test is fixed
mkdir -p /model/llama-ft/7B/nllm
mkdir -p /model/llama-ft/7B/nllm_decode

# test
make test
```

## Run with python serving server

### Develop

 1. develop with local C++ style (Recommanded)

```bash

mkdir -p ${GIT_PROJECT_REPO_ROOT}/build
cd ${GIT_PROJECT_REPO_ROOT}/build
cmake -DSM=86 -DWITH_TESTING=ON ..
make -j
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python
ln -s ${GIT_PROJECT_REPO_ROOT}/build/lib .
```

 2. develop with pythonic style

```bash

cd ${GIT_PROJECT_REPO_ROOT}
python setup.py build_ext
python setuppy develop

# when you change C++ code, you need run
python setup.py build_ext
# to re-compile binary code
```

 3. launch local serving server

```bash
cd ${GIT_PROJECT_REPO_ROOT}/src/ksana_llm/python

# download model: for example 7b
wget https://mirrors.tencent.com/repository/generic/pcg-numerous//dependency/numerous_llm_models/llama2_7b_fp16_1_gpu.tgz
tar vzxf llama2_7b_fp16_1_gpu.tgz
# change llama2_7b_fp16_1_gpu/config.ini
# ...
# model_dir=/path/to/llama2_7b_fp16_1_gpu
# ...

# download tokenizer: for example 7b
wget https://mirrors.tencent.com/repository/generic/pcg-numerous/dependency/numerous_llm_models/llama2_7b_hf.tgz
tar vzxf llama2_7b_hf.tgz

# launch server
python serving_server.py --model_dir llama2_7b_fp16_1_gpu --tokenizer_dir llama2_7b_hf

# open another session, request client
python serving_client.py
```

### Distribute

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

## Run with http raw server (Deprecated)

```bash
# run standalone demo using llama7b
./bin/ksana_llm --model_config ../examples/llama7b/config.ini
# check runing status
# open another terminal or session
cat ./log/ksana_llm.log
```

Start client
```bash
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
