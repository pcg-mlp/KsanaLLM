import requests
import multiprocessing
import json

def request(data, queue=None):
    resp = requests.post("http://localhost:8080/generate", data=data, timeout=3)
    if queue is None:
        return json.loads(resp.content)
    else:
        queue.put(json.loads(resp.content))

def check_result(data, result):
    for token_outter_idx in range(len(data["tokens"])):
        token = data["tokens"][token_outter_idx]
        for token_idx in range(len(token)):
            assert result["tokens"][token_outter_idx][token_idx] == token[token_idx]
        assert result["tokens"][token_outter_idx][-1] == 2

if __name__ == "__main__":
    data = {"tokens": [[1,2,3],[4,5,6,7,8]], "tokens_len":[3, 5], "model_name": "llama"}
    # timeout in 3 seconds
    resp_data = request(data)
    print(resp_data)
    check_result(data, resp_data)

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

