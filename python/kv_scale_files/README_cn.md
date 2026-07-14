# 标定KV Cache缩放因子

## kv_cache_scales.json 文件说明

- kv_cache_scales.json 用于存储每个TP并行下模型每一层对应的KV Cache缩放因子，缩放因子用于FP8 E4M3的计算
- 如果使用FP8 E4M3格式的kv cache，请在您的模型目录下准备对应的kv_cache_scales.json，src/ksana_llm/python/kv_scale_files/下提供了部分模型的kv_cache_scales.json示例
- 一念暂时仅支持tp_size = 1的KV Cache量化

## 如何标定模型的 kv_cache_scales.json

- 参考 https://github.com/vllm-project/vllm/blob/main/examples/fp8/README.md
- 注释 PYTHON_PATH/site-packages/ammo/torch/export/layer_utils.py 346-347
  '''
    # if get_kv_cache_dtype(qkv_modules) == KV_CACHE_FP8:
    #     return torch.tensor([1.0], dtype=torch.float)
  '''
- 修改 quantize.py
  '''
    quantize.py:29   from ammo.torch.export import export_tensorrt_llm_checkpoint
    quantize.py:306              export_tensorrt_llm_checkpoint(
    quantize.py:314                  # export_tensorrt_llm_config=False,
  '''
- 修改 extract_scales.py.py
  '''
    extract_scales.py:271         "model_type": lambda json_dict: json_dict["decoder"],
    extract_scales.py:272         "tp_size": lambda json_dict: int(json_dict["mapping"]["tp_size"]),
  '''
