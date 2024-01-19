import requests
import multiprocessing
import json


def request(data, queue=None):
    resp = requests.post("http://localhost:8080/generate", data=data, timeout=30)
    if queue is None:
        return json.loads(resp.content)
    else:
        queue.put(json.loads(resp.content))

def check_result(data, result):
    pass

if __name__ == "__main__":
    data = {"input_tokens": [233, 1681], "tokens_len": 2, "model_name": "llama"}
    # timeout in 3 seconds
    resp_data = request(data)
    print(resp_data)
    check_result(data, resp_data)
    pass
    multi_proc_queue = multiprocessing.Queue()
    multi_proc_list = []
    # concurrent 10 proc
    for _ in range(10):
        proc = multiprocessing.Process(target=request, args=(data, multi_proc_queue,))
        proc.start()
        multi_proc_list.append(proc)

    for _ in range(10):
        check_result(data, multi_proc_queue.get())

    for proc in multi_proc_list:
        proc.join()
