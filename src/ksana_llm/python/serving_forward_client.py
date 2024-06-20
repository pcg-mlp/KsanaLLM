# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import requests
import multiprocessing
import argparse
import json
import time
import msgpack


def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host',
                        type=str,
                        default="localhost",
                        help='server host address')
    parser.add_argument('--port', type=int, default=8888, help='server port')
    args = parser.parse_args()
    return args


def post_request_msgpack(serv, data, queue=None):
    packed_data = msgpack.packb(data)
    response = requests.post(serv, data=packed_data, headers={'Content-Type': 'application/msgpack'})
    if queue is None:
        return msgpack.unpackb(response.content)
    else:
        queue.put(msgpack.unpackb(response.content))

def show_response(data, result):
    print("result:", result)


if __name__ == "__main__":
    args = args_config()
    serv = "http://" + args.host + ":" + str(args.port) + "/forward"

    text_list = [
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n你好。<|im_end|>\n<|im_start|>assistant",
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
            # "subinput_pos": [6,6,6],
            # "subinput_embedding": [[1.0,2.0],[3.0,4.0]],
        }

        # Create a new process to handle the post request

        proc = multiprocessing.Process(
            target=post_request_msgpack,  # function to call
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
