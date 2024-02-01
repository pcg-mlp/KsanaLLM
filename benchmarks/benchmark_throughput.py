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
    parser.add_argument('--port', type=int, default=8888, help='server port')
    parser.add_argument('--input_csv',
                        type=str,
                        default="benchmark_input.csv",
                        help='input data for benchmark')
    args = parser.parse_args()
    return args


def post_request(serv, data, queue=None):
    while not queue.empty():
        prompt = queue.get(False)
        if prompt is None:
            break
        data = {
            "model_name": "llama",
            "prompt": prompt,
            "sampling_config": {
                "temperature": 0.0,
                "topk": 1,
                "topp": 0.0
            },
        }
        resp = requests.post(serv, data=json.dumps(data), timeout=6000000)
        result_str = json.loads(resp.content)
        print(result_str)


def read_from_csv(csv_file, col_idx=0, remove_head=True):
    import csv
    csv_reader = csv.reader(open(csv_file))
    if remove_head:
        next(csv_reader)
    for line in csv_reader:
        yield line[col_idx]


if __name__ == "__main__":
    args = args_config()
    serv = "http://" + args.host + ":" + str(args.port) + "/generate"
    input_generator = read_from_csv(args.input_csv)
    multi_proc_list = []
    multi_proc_queue = multiprocessing.Queue()
    input_data_list = []
    thread_num = 16

    start_time = time.time()
    for input_idx, input_str in enumerate(input_generator):
        prompt = "USER: %s\nASSISTANT:" % input_str
        input_data_list.append(input_str)
        multi_proc_queue.put(prompt)

    for _ in range(thread_num):
        proc = multiprocessing.Process(target=post_request,
                                       args=(
                                           serv,
                                           None,
                                           multi_proc_queue,
                                       ))
        proc.start()
        multi_proc_list.append(proc)

    for proc in multi_proc_list:
        proc.join()

    end_time = time.time()
    print("{} requests duration: {:.3f}s".format(len(input_data_list),
                                                 end_time - start_time))
