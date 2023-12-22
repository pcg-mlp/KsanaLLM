import requests

if __name__ == "__main__":
    data = {"tokens": [[1,2,3],[4,5,6,7,8]], "tokens_len":[3, 5], "model_name": "llama"}
    # timeout in 3 seconds
    resp = requests.post("http://localhost:8080/generate", data=data, timeout=3)
    print(resp.content)