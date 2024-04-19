# LLM Kernels

## Models

  - [Llama 7B/13B/33B/65B](https://ai.meta.com/blog/large-language-model-llama-meta-ai/)
  - [Llama 2 7B/13B/70B](https://ai.meta.com/llama/)

## Devices

  - Nvidia GPU
  - Ascend NPU (only 910B2C)

## Kernels

  - masked_multihead_attention
  - gemm_wrapper of cublas and cutlass
  - attention
  - layernorm
  - embedding, position embedding
  - paged_attention
  - rotary_embedding
  - cast
    - float16 to float32
  - assemble last token
  - paged attention
  - activation
    - silu

## Compile and run on nVidia GPU
  0. Create container with image mirrors.tencent.com/todacc/venus-numerous-llm:0.1.19
  1. `git clone https://xxx/LLM_kernels.git`
  2. `cd LLM_kernels`
  3. `mkdir build`
  4. `cmake ..` or you know sm code of current Nvidia GPU for example A10 `cmake -DSM=86 ..`. Using -DSM can reduce the compile time.
  5. `make test`

## Compile and run on Huawei Ascend NPU
  0. Create container with image mirrors.tencent.com/todacc/venus-std-base-tlinux3-npu-llm:0.1.3
  1. `git clone https://xxx/LLM_kernels.git`
  2. `cd LLM_kernels`
  3. `mkdir build`
  4. `cmake .. -DWITH_CUDA=OFF -DWITH_ACL=ON -DWITH_TEST=ON`
  5. `make test`