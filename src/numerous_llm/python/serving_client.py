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
                        default="0.0.0.0",
                        help='server host address')
    parser.add_argument('--port',
                        type=int,
                        default=8888,
                        help='server port')
    args = parser.parse_args()
    return args


def post_request(serv, data, queue=None):
    resp = requests.post(serv, data=data, timeout=600000)
    if queue is None:
        return json.loads(resp.content)
    else:
        queue.put(json.loads(resp.content))


def show_response(data, result):
    print("result:", result)


if __name__ == "__main__":
    args = args_config()
    serv = "http://" + args.host + ":" + str(args.port) + "/generate"

    # text_list = ["您好！"]
    text_list = ["您好!", "您好!", "您好!", "您好!", "您好!", "您好!", "您好!", "您好!", "您好!", "您好!"]

    multi_proc_list = []
    multi_proc_queue = multiprocessing.Queue()

    for i in range(len(text_list)):
        # prompt = "USER:%s\nASSISTANT:" % text_list[i]
        prompt = "[INST]%s[/INST]" % text_list[i]

        data = {
            "model_name": "llama",
            "prompt": prompt,
            "sampling_config": {
                "temperature": 0.0,
                "topk": 1,
                "topp": 0.0
            },
        }

        proc = multiprocessing.Process(
            target=post_request, args=(
                serv, json.dumps(data), multi_proc_queue,))
        proc.start()
        multi_proc_list.append(proc)

    for i in range(len(text_list)):
        show_response(text_list[i], multi_proc_queue.get())

    for proc in multi_proc_list:
        proc.join()
