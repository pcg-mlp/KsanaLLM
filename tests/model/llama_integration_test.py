# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================
import sys
import argparse
import logging
import threading
import queue
import time

# for C++ lib python wrapper
sys.path.insert(0, '../../build/lib')
# for python interface
sys.path.insert(0, '../../src/ksana_llm/python')
sys.path.insert(0, '.')

import ksana_llm

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)
# create formatter
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
stream_handler.setFormatter(formatter)
logging.getLogger().addHandler(stream_handler)

from transformers import GenerationConfig, LlamaTokenizer


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        default="examples/ksana_llm.yaml",
                        help='serving config file')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="/model/llama-hf/13B",
                        help='tokenizer dir')
    args = parser.parse_args()
    return args


def infer(prompt, tokenizer, generation_config, model, queue=None, idx=0):
    input_tokens = tokenizer.encode(prompt, add_special_tokens=True)
    generation_config = GenerationConfig(num_beams=1,
                                         top_k=1,
                                         top_p=0.0,
                                         temperature=0.0)
    result = model.generate(model_name="llama",
                            inputs=input_tokens,
                            generation_config=generation_config,
                            streamer=None)
    if queue is None:
        return tokenizer.decode(result)
    else:
        queue.put((idx, tokenizer.decode(result)))
        return


if __name__ == "__main__":
    args = args_config()

    logging.info("start load tokenizer")
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir)
    logging.info("finish load tokenizer")

    logging.info("start load model")
    start_time = time.perf_counter()
    model = ksana_llm.AutoModel.from_config(args.config_file)
    end_time = time.perf_counter()
    model_load_time = end_time - start_time
    logging.info("finish load model")

    warmup_prompts = [
        "[INST]作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何向地面控制团队传达您的观察结果和建议？[/INST]",
        "[INST]作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何在太空吃饭？[/INST]"
    ]

    ref_result = [
        "如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即向地面控制团队发出紧急通信。我会使用空间站上的通信设备，向地面控制团队报告我的观察结果和建议。首先，我会向地面控制团队报告我所看到的外星实体的位置、形状、大小和颜色等详细信息。我会尽可能提供准确的描述，以便地面控制团队能够更好地了解外星实体的情况。其次，我会向地面控制团队建议采取适当的措施来保护空间站和宇航员的安全。这可能包括减速空间站的速度，改变空间站的方向，或者采取其他措施来避免与外星实体发生碰撞。最后，我会向地面控制团队提供任何其他有用的信息，以便他们能够更好地了解外星实体的情况。这可能包括外星实体的运动轨迹、速度和方向等信息。总之，如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即向地面控制团队发出紧急通信，并提供尽可能详细的观察结果和建议，以便地面控制团队能够采取适当的措施来保护空间站和宇航员的安全。</s>",
        "作为一名国际空间站上的宇航员，如果我意外地目睹了外星实体接近空间站，我的第一反应可能是惊恐和惊讶。但是，我们的训练和准备使我们能够在任何情况下保持冷静和专业。在太空中吃饭是一项非常重要的任务，因为我们需要保持身体健康和精神状态良好，以便完成我们的任务。在太空中，我们使用特殊的食品和餐具，以确保我们的食物和饮料不会漂浮到空间站的其他部分或飘出太空舱。我们的食物通常是预包装的，并且在太空中使用特殊的加热器和冷却器进行加热和冷却。我们还使用特殊的餐具，例如吸管和勺子，以避免食物和液体漂浮到空间站的其他部分。在太空中，我们通常会吃固体食物，例如面包、饼干和坚果，以及液体食物，例如果汁和牛奶。我们还会吃一些特殊的食物，例如蛋白质棒和能量棒，以帮助我们保持身体健康和精神状态良好。在太空中，我们通常会在特殊的餐桌上吃饭，这些餐桌可以固定在空间站的墙壁上，以避免食物和餐具漂浮到空间站的其他部分。我们还会在特殊的食品袋中储存食物和饮料，以便在需要时使用。总之，在太空中吃饭需要特殊的准备和技巧，以确保我们的食物和饮料不会漂浮到空间站的其他部分或飘出太空舱。我们的训练和准备使我们能够在任何情况下保持冷静和专业，以确保我们的身体健康和精神状态良好。</s>"
    ]

    none_prefix_prompt = ["[INST]你好。[/INST]"]
    none_prefix_prompt_ref = ["你好，有什么我可以帮助你的吗？</s>"]

    generation_config = GenerationConfig(num_beams=1,
                                         top_k=1,
                                         top_p=0.0,
                                         temperature=0.0)

    # prefix_len should be 32
    multi_thread_list = []
    multi_thread_queue = queue.Queue()

    logging.info("start warmup")
    start_time = time.perf_counter()
    for idx, prompt in enumerate(warmup_prompts):
        result = infer(prompt, tokenizer, generation_config, model)
        assert result.replace("\n", "") == ref_result[idx]
    end_time = time.perf_counter()
    warmup_time = end_time - start_time
    logging.info("finish warmup")

    logging.info("start infer")
    start_time = time.perf_counter()
    for idx, prompt in enumerate(warmup_prompts):
        thread = threading.Thread(target=infer,
                                  args=(
                                      prompt,
                                      tokenizer,
                                      generation_config,
                                      model,
                                      multi_thread_queue,
                                      idx,
                                  ))
        thread.start()
        multi_thread_list.append(thread)

    for thread in multi_thread_list:
        thread.join()
    end_time = time.perf_counter()
    total_infer_time = end_time - start_time
    logging.info("finish infer")
    assert total_infer_time < warmup_time

    while not multi_thread_queue.empty():
        idx, result = multi_thread_queue.get(block=False)
        assert result.replace("\n", "") == ref_result[idx]

    print(f"Total model load time: {model_load_time:.2f} s")
    print(f"Total warmup time: {warmup_time:.2f} s")
    print(f"Total infer time: {total_infer_time:.2f} s")

    result = infer(none_prefix_prompt[0], tokenizer, generation_config, model)
    assert result.replace("\n", "") == none_prefix_prompt_ref[0]
