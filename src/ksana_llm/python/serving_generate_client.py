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
        "[INST]作为国际空间站上的宇航员，您意外地目睹了外星实体接近空间站。您如何向地面控制团队传达您的观察结果和建议？[/INST]",
        "[INST]想象一下您是夏洛克·福尔摩斯，您被要求解开一个涉及失踪传家宝的谜团。请解释一下您找到该物品的策略。[/INST]"
        # "USER:你好\nASSISTANT:",
    ]

    multi_proc_list = []
    multi_proc_queue = multiprocessing.Queue()
    start_time = time.time()
    # Iterate over the list of text prompts
    for i in range(len(text_list)):
        # Create a prompt string from the current text element
        prompt = "%s" % text_list[i]

        # Create a data dictionary to pass to the post_request function
        data = {
            # Set the prompt for the request
            "prompt": prompt,
            # "input_refit_embedding": {
            #     "pos": [20,285,550],
            #     "embeddings": [[1.0,2.0,...],[3.0,4.0,...],[5.0,6.0,...]],
            # },
            # Configure the sampling parameters
            "sampling_config": {
                "temperature": 0.0,  # temperature for sampling
                "topk": 1,  # top-k sampling
                "topp": 0.0,  # top-p sampling
                "logprobs": 0, # Return the n token log probabilities for each position.
                "max_new_tokens":
                128,  # maximum number of new tokens to generate
                "repetition_penalty": 1.0,  # penalty for repetitive responses
                "stop_token_ids": [] # list of tokens that stop the generation
            },
            # Set stream mode to False
            "stream": False,
        }

        # Create a new process to handle the post request
        proc = multiprocessing.Process(
            target=post_request,  # function to call
            args=(  # arguments to pass
                serv,  # server object
                data,  # data dictionary
                multi_proc_queue,  # queue for communication
            ))
        # Start the process
        proc.start()
        # Add the process to the list of processes
        multi_proc_list.append(proc)

    # Wait for the responses from the processes
    for i in range(len(text_list)):
        # Get the response from the queue and display it
        show_response(text_list[i], multi_proc_queue.get())

    # Wait for all processes to finish
    for proc in multi_proc_list:
        # Join the process to ensure it has finished
        proc.join()

    end_time = time.time()
    print("{} requests duration: {:.3f}s".format(len(text_list),
                                                 end_time - start_time))
