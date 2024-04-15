# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import requests
import multiprocessing
import argparse
import json
import time


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default="localhost",
                        help='server host address')
    parser.add_argument('--port', type=int, default=8888, help='server port')
    args = parser.parse_args()
    return args


def post_request(serv, data, queue=None):
    resp = requests.post(serv, json=data, timeout=600000)
    if queue is None:
        return json.loads(resp.content)
    else:
        queue.put(json.loads(resp.content))


def show_response(data, result):
    print("result:", result)


if __name__ == "__main__":
    args = args_config()
    serv = "http://" + args.host + ":" + str(args.port) + "/generate"

    text_list = [
        #"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你是谁<|im_end|>\n<|im_start|>assistant\n", # qwen2
        "[INST]作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何向地面控制团队传达您的观察结果和建议？[/INST]",
        "[INST]想象一下您是夏洛克·福尔摩斯，您被要求解开一个涉及失踪传家宝的谜团。请解释一下您找到该物品的策略。[/INST]"
    ]

    multi_proc_list = []
    multi_proc_queue = multiprocessing.Queue()
    start_time = time.time()
    for i in range(len(text_list)):
        prompt = "%s" % text_list[i]

        data = {
            "prompt": prompt,
            "sampling_config": {
                "temperature": 0.0,
                "topk": 1,
                "topp": 0.0,
                "max_new_tokens" : 1024,
                "repetition_penalty": 1.0
            },
            "stream": False,
        }

        proc = multiprocessing.Process(target=post_request,
                                       args=(
                                           serv,
                                           data,
                                           multi_proc_queue,
                                       ))
        proc.start()
        multi_proc_list.append(proc)

    for i in range(len(text_list)):
        show_response(text_list[i], multi_proc_queue.get())

    for proc in multi_proc_list:
        proc.join()

    end_time = time.time()
    print("{} requests duration: {:.3f}s".format(len(text_list),
                                                 end_time - start_time))
