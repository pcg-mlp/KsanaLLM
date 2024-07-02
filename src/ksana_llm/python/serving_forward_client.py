# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import requests
import multiprocessing
import argparse
import json
import time
import msgpack
import numpy as np
import base64


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

def python_tensor_to_numpy(python_tensor):
    """
    Converts a custom PythonTensor object into a NumPy array.

    This function is designed to transform a PythonTensor, which is a custom object for this example,
    into a NumPy array. The conversion process takes into account the specified shape and
    data type of the input tensor to ensure accurate representation in the NumPy array format.

    Parameters:
    - python_tensor: A dictionary containing the data buffer from the PythonTensor object,
                     the desired shape for the resulting NumPy array, and the desired data type.
                     The data buffer should be a buffer or a similar object that
                     holds the raw data.

    Returns:
    - numpy.ndarray: A NumPy array constructed from the PythonTensor's data, adhering to the
                     specified shape and data type requirements.

    Raises:
    - ValueError: If the specified data type is not recognized. Currently, only 'float32',
                  'float16', 'uint16', and 'int32' data types are supported, represented as
                  'float32', 'float16', 'bfloat16', and 'int32' respectively.
    """
    # Extract data, shape, and dtype from the python_tensor dictionary
    data, shape, dtype = python_tensor["data"], python_tensor["shape"], python_tensor["dtype"]
    
    # Map the specified string data type to its corresponding NumPy data type
    if dtype == "float32":
        np_dtype = np.float32
    elif dtype == "float16":
        np_dtype = np.float16
    elif dtype == "bfloat16":
        np_dtype = np.uint16
    elif dtype == "int32":
        np_dtype = np.int32
    else:
        # Throw an exception for unsupported data types
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Create a NumPy array from the raw data buffer using the specified data type
    data_array = np.frombuffer(base64.b64decode(data), dtype=np_dtype)
    
    # Reshape the NumPy array according to the specified shape
    numpy_array = data_array.reshape(shape)

    # Special handling for 'bfloat16' dtype to convert uint16 to float32
    if dtype == "bfloat16":
        numpy_array = numpy_array.astype(np.uint32) << 16
        numpy_array = numpy_array.view(np.float32)

    return numpy_array

def show_response(data, result):
    if isinstance(result, dict) and "response" in result:
        for response in result["response"]:
            target = response["target_name"]
            python_tensor = response["tensor"]
            print(
                f"input_token_ids : {result['input_token_ids']}, target : {target}, tensor : \n{python_tensor_to_numpy(python_tensor)}")
    else:
        print(result)

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
            # "input_refit_embedding": {
            #     "pos": [20,285,550],
            #     "embeddings": [[1.0,2.0,...],[3.0,4.0,...],[5.0,6.0,...]],
            # },
            "request_target": [
                {
                    "target_name": "layernorm",
                    "token_id": [13],
                    # "slice_pos" : [[0,0],[2,5],[7,7]],
                    "token_reduce_mode": "GATHER_ALL",
                    # "token_reduce_mode" : "GATHER_TOKEN_ID",
                },
            ],
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
