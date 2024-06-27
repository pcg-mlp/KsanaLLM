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
    Converts a custom PythonTensor object to a NumPy array.

    This function is designed to take a PythonTensor, a custom object for this example,
    and convert it into a NumPy array. The conversion respects the specified shape and
    data type of the input tensor.

    Parameters:
    - python_tensor: A tuple containing the data buffer from the PythonTensor object,
                     the desired shape of the NumPy array, and the desired data type.
                     The data buffer is expected to be a buffer or similar object that
                     contains the raw data.

    Returns:
    - numpy.ndarray: A NumPy array created from the PythonTensor's data, with the
                     specified shape and data type.

    Raises:
    - ValueError: If the specified data type is not supported. Currently, only 'float32'
                  and 'int32' data types are supported.
    """
    # Unpack the python_tensor tuple into data, shape, and dtype
    data, shape, dtype, *_ = python_tensor
    
    # Map the specified string data type to the corresponding NumPy data type
    if dtype == "f4":
        np_dtype = np.float32
    elif dtype == "f2":
        np_dtype = np.float16
    elif dtype == "bf2":
        np_dtype = np.uint16
    elif dtype == "i4":
        np_dtype = np.int32
    else:
        # Raise an exception if an unsupported data type is specified
        raise ValueError(f"Unsupported dtype: {dtype}")

    # Convert the raw data buffer to a NumPy array with the specified data type
    data_array = np.frombuffer(data, dtype=np_dtype)
    
    # Reshape the NumPy array to the desired shape
    numpy_array = data_array.reshape(shape)

    if dtype == "bf2":
        numpy_array = numpy_array.astype(np.uint32) << 16
        numpy_array = numpy_array.view(np.float32)

    return numpy_array

def show_response(data, result):
    if "response" in result:
        for target, python_tensor in result["response"].items():
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
            "request_target" : {
                "layernorm" : {
                    "token_id" : [13],
                    #"slice_pos" : [[0,0],[2,5],[7,7]],
                    "token_reduce_mode" : "GATHER_ALL",
                    #"token_reduce_mode" : "GATHER_TOKEN_ID",
                },
            },
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
