#!/bin/bash
set -ex
script_path="$( cd "$(dirname "$BASH_SOURCE")" ; pwd -P )"
root_path="${script_path}/../../../.."
function build()
{
  
  if [ -d ${root_path}/build ];then
    rm -rf ${root_path}/build
  fi

  mkdir -p ${root_path}/build
  cd ${root_path}/build

  cmake .. -DWITH_CUDA=OFF -DWITH_ACL=ON -DWITH_TESTING=ON
  make -j
  cd -
}

function download_weight()
{
  if [ ! -d ${script_path}/../data ]; then
    wget http://mirrors.tencent.com/repository/generic/venus_repo/llm_finetune/common_tools/npu/ksana/data.zip
    unzip data.zip
  fi
  if [ ! -d ${script_path}/../llama_weight ]; then
    wget http://mirrors.tencent.com/repository/generic/venus_repo/llm_finetune/common_tools/npu/ksana/llama_weight.zip
    tar xf llama_weight.zip
  fi
}

function main()
{
  rm -rf ${script_path}/../output ${script_path}/../input
  mkdir -p ${script_path}/../input
  mkdir -p ${script_path}/../output
  touch ${script_path}/../input/.keep
  if [ $# -lt 2 ]; then
    build
  fi
  model_type="7b"
  if [ $# -ge 1 ]; then
    model_type=$1
  fi
  if [ "$model_type" == "7b" ]; then
    download_weight
  fi
  cp ${root_path}/build/bin/llama_run ${script_path}/../output/
  cd ${script_path}/../output
  ./llama_run ${model_type} > log 2>&1 &
  cd ${script_path}
  python3 input_trans.py ${model_type}
}

main $@
