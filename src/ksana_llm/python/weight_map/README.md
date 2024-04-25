# 自定义模型 weight map

## weight_map.json 文件说明

- weight_map.json 用于存储用户模型中,与 **Llama** 模型 weight_name !!#ff0000  **不同**!! 的自定义 weight 名称的映射表。
- key 将作为正则匹配替换时的 `re` 字符串(`std::basic_regex`)
	- key 将作为正则匹配替换时的 `re` 字符串(`std::basic_regex`)
	- value 将作为正则匹>配替换时的 `fmt` 字符串(替换格式字符串)
	- 关于 `regex_replace` 更多定义，可查看 [链接](https://zh.cppreference.com/w/cpp/regex/regex_replace)
## 如何实现一个自定义模型 weight map

- 对于以 **Llama** 为基座，且实际模型结构改动量不大的模型 ，可以通过扩展自定义 weight_map 集成进**一念**引擎中。
- 以 **Qwen1** 模型为例:
  - **Qwen1** 模型中包含参数名 `transformer.wte.weight`,对应 **Llama** 模型中的 `model.embed_tokens.weight`,则  key-value 键值对应写为：
  ``` 
  "transformer.wte.weight": "model.embed_tokens.weight"`
  ```
  - **Qwen1** 模型中包含参数名 `transformer.h.LAYER_ID.ln_1.weight` (`LAYER_ID`表示所在层数),对应 **Llama** 模型中的 `model.layers.LAYER_ID.input_layernorm.weight`。这个参
数中包含了一个数字 `LAYER_ID`,且在前后的 weight_name 中均有体现,因此键值对应这么写：
```
`"transformer.h.(\\d+).ln_1.weight": "model.layers.$1.input_layernorm.weight"`
// 使用 `()` 来标记 key 中需要保留的部分
// 使用 \\d+ 表示匹配一个整型数字
// 使用 `$1 $2 $3 $...` 将他们依次填写入 value 合适的位置 
```
  -** Qwen1** 与 **Llama** 完全相同的 weight_name: `lm_head.weight`, **!!#ff0000 不需要!!**添加 key-value 对.
