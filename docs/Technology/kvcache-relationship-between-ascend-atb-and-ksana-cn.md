# 昇腾ATB KVCache与一念KVCache之间的转换关系

### 背景知识

- Ksana kv cache由num_blocks个独立存储空间的block组成，block与block之间不保证是连续的存储空间。每个block的形状为[2, layer_num, block_token_num, head_num, head_dim]，其中2代表key和value。

- Ascend ATB的kvcache由kcache和vcache组成，kcache和vcache是独立且连续的存储空间。kcache和vcache的形状为[num_blocks * layer_num, block_token_num, head_num, head_dim]。每个block的大小为[block_token_num, head_num, head_dim]。

### 背景

- 接入NPU需要使用Ascend ATB（以下简称ATB），为了让NPU的self/paged attention能够使用Ksana的kvcache，并共享底层内存/显存管理能力，需要将Ksana kv cache转换为Ascend ATB kvcache的模式。

### 兼容和转换方式

1. 更改blocks的分配方式，使得blocks在物理上是连续的，而上层指针指向不同的存储空间。Ksana kvcache原来是每个block调用一次malloc，现在改为预先分配一个连续的存储空间，大小为[num_blocks, 2, layer_num, block_token_num, head_num, head_dim]。原先每个block的指针指向cache_base_ptr + (block index * 2 * layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE))。

2. 在每次推理过程中，每个prompt会带有一个block id数组，可以通过block id获取存储空间的指针。对于ATB，需要进行转换才能使用。转换过程如下：
   - 有block id数组[b0, b1, b2, b3, b4]，以及Ksana上kvcache经过第1步改造的kvcache基础地址指针cache_base_ptr。
   - 对于ATB来说，Ksana的kvcache总共有num_blocks * 2 * layer_num个block。
   - 那么ATB的block id数组为[b0 * layer_num * 2, b1 * layer_num * 2, b2 * layer_num * 2, b3 * layer_num * 2, b4 * layer_num * 2]。
   - Ksana的kvcache以block为单位交换显存/内存，因此为了复用Ksana的kvcache底层内存/显存管理能力，ATB的kcache和vcache共用同一个Ksana kvcache。
   - 由于Ksana的block内部分为K和V两部分，每部分大小为[layer_num, block_token_num, head_num, head_dim]。
   - 为了让ATB的kcache和vcache可以共用同一个block id数组，kcache的指针为cache_base_ptr，vcache的指针为cache_base_ptr + (layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE))。
   - 那么kcache/vcache的block id数组为[b0 * layer_num * 2 + layer_idx, b1 * layer_num * 2 + layer_idx, b2 * layer_num * 2 + layer_idx, b3 * layer_num * 2 + layer_idx, b4 * layer_num * 2 + layer_idx]。