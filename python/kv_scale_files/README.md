# Calibrate KV Cache Scaling Factors

## kv_cache_scales.json File Description

- kv_cache_scales.json is used to store the scaling factors for the KV Cache corresponding to each layer of the model under each TP parallel. The scaling factors are used for the calculation of FP8 E4M3.
- To enable FP8 E4M3 KV Cache, please prepare the corresponding kv_cache_scales.json file in your model directory. Some examples of kv_cache_scales.json are provided in src/ksana_llm/python/kv_scale_files/.
- Ksana currently only supports quantization of KV Cache with tp_size = 1.

## How to Calibrate kv_cache_scales.json

- reference to: https://github.com/vllm-project/vllm/blob/main/examples/fp8/README.md
- Annotation: PYTHON_PATH/site-packages/ammo/torch/export/layer_utils.py 346-347
  '''
    # if get_kv_cache_dtype(qkv_modules) == KV_CACHE_FP8:
    #     return torch.tensor([1.0], dtype=torch.float)
  '''
- Modify: quantize.py
  '''
    quantize.py:29   from ammo.torch.export import export_tensorrt_llm_checkpoint
    quantize.py:306              export_tensorrt_llm_checkpoint(
    quantize.py:314                  # export_tensorrt_llm_config=False,
  '''
- Modify: extract_scales.py.py
  '''
    extract_scales.py:271         "model_type": lambda json_dict: json_dict["decoder"],
    extract_scales.py:272         "tp_size": lambda json_dict: int(json_dict["mapping"]["tp_size"]),
  '''
