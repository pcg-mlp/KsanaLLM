# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import requests
import multiprocessing
import json
import time


def post_request(data, queue=None):
    resp = requests.post("http://localhost:8888/generate", data=data, timeout=30)
    if queue is None:
        return json.loads(resp.content)
    else:
        queue.put(json.loads(resp.content))


def show_response(data, result):
    print("result:", result["texts"])


text_list = ["您好!"]
# text_list = ["您好!", "你是谁？"]

multi_proc_list = []
multi_proc_queue = multiprocessing.Queue()

for i in range(len(text_list)):
    prompt = text_list[i]

    data = {"model_name": "llama",
            "prompts": [prompt],
            "sampling_configs": [{"temperature": 1.0, "topk": 1, "topp": 0.0}],
            }

    proc = multiprocessing.Process(
        target=post_request, args=(json.dumps(data), multi_proc_queue,))
    proc.start()
    multi_proc_list.append(proc)


for i in range(len(text_list)):
    show_response(text_list[i], multi_proc_queue.get())

for proc in multi_proc_list:
    proc.join()
