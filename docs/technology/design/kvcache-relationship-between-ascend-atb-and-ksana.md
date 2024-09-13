# KVCache Relationship Between Ascend ATB And Ksana

### Background Knowledge

- The Ksana kv cache consists of blocks, each of which is an independent storage space. The blocks are not guaranteed to be contiguous in memory. Each block has a shape of [2, layer_num, block_token_num, head_num, head_dim], where 2 represents key and value.

- The Ascend ATB kv cache consists of kcache and vcache, which are independent contiguous storage spaces. The shapes of kcache and vcache are [num_blocks * layer_num, block_token_num, head_num, head_dim]. Each block has a size of [block_token_num, head_num, head_dim].

### Background

- To interface with the NPU, Ascend ATB (hereinafter referred to as ATB) needs to be used. In order for the NPU's self/paged attention to utilize Ksana's kv cache and share the underlying memory/GPU memory management capabilities, the Ksana kv cache needs to be converted to the Ascend ATB kv cache format.

### Compatibility and Conversion Method

1. Change the block allocation method so that the blocks are contiguous in physical memory, while the upper-level pointers point to different storage spaces. Originally, each block in the Ksana kv cache called malloc once. This should be changed to pre-allocate a contiguous storage space of size [num_blocks, 2, layer_num, block_token_num, head_num, head_dim]. The pointers of each block should then point to cache_base_ptr + (block index * 2 * layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).

2. During each inference process, each prompt will carry an array of block IDs, which can be used to obtain the pointers to the storage space. For ATB, conversion is required to use these pointers. The conversion process is as follows:
   - Given a block ID array [b0, b1, b2, b3, b4] and the base address pointer of the Ksana kv cache after the modification in step 1, cache_base_ptr.
   - For ATB: The Ksana kv cache has a total of num_blocks * 2 * layer_num blocks.
   - Therefore, the block ID array for ATB is [b0 * layer_num * 2, b1 * layer_num * 2, b2 * layer_num * 2, b3 * layer_num * 2, b4 * layer_num * 2].
   - Ksana's kv cache swaps memory/GPU memory at the block level, so to reuse Ksana's kv cache's underlying memory/GPU memory management capabilities, ATB's kcache and vcache share the same Ksana kv cache.
   - Since each block in Ksana is divided into K and V parts, each part having a size of [layer_num, block_token_num, head_num, head_dim].
   - To allow ATB's kcache and vcache to share the same block ID array, the kcache pointer is cache_base_ptr, and the vcache pointer is cache_base_ptr + (layer_num * block_token_num * head_num * head_dim * sizeof(DTYPE)).
   - Therefore, the block ID array for kcache/vcache is [b0 * layer_num * 2 + layer_idx, b1 * layer_num * 2 + layer_idx, b2 * layer_num * 2 + layer_idx, b3 * layer_num * 2 + layer_idx, b4 * layer_num * 2 + layer_idx].