import sys
import argparse
import threading
import queue
import os
import subprocess
import random

from queue import Queue, Empty
from concurrent.futures import ThreadPoolExecutor

# Add paths for C++ lib python wrapper and python interface
sys.path.insert(0, '../../build/lib')
sys.path.insert(0, '../../src/ksana_llm/python')
sys.path.insert(0, '.')


# Define a function to parse command line arguments
def args_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file',
                        type=str,
                        default="examples/ksana_llm.yaml",
                        help='serving config file')
    parser.add_argument('--tokenizer_dir',
                        type=str,
                        default="/model/llama-hf/13B",
                        help='tokenizer dir')
    parser.add_argument('--model', type=str, default="qwen", help='model type')
    args = parser.parse_args()
    return args


def enqueue_output(file, queue):
    for line in iter(file.readline, ''):
        queue.put(line)
    file.close()


def read_popen_pipes(p):
    with ThreadPoolExecutor(2) as pool:
        q_stdout, q_stderr = Queue(), Queue()
        pool.submit(enqueue_output, p.stdout, q_stdout)
        pool.submit(enqueue_output, p.stderr, q_stderr)
        while True:
            if p.poll() is not None and q_stdout.empty() and q_stderr.empty():
                break
            out_line = err_line = ''
            try:
                out_line = q_stdout.get_nowait()
            except Empty:
                pass
            try:
                err_line = q_stderr.get_nowait()
            except Empty:
                pass
            yield (out_line, err_line)


def wait_for_server_launch(server_proc, server_status_queue):
    for _, err_line in read_popen_pipes(server_proc):
        if len(err_line) > 0:
            server_status_queue.put_nowait(err_line)


def read_from_csv(csv_file, col_idx=0, remove_head=True):
    import csv
    csv_reader = csv.reader(open(csv_file))
    return [row[0] for row in csv_reader]


def get_ref_results(model):
    if model == "qwen":
        return [
            "1. 确保信息准确：首先，我需要确保我所观察到的外星实体是真实的，而不是幻觉。我需要确保我所观察到的实体是外星生物，而不"
            "是其他任何可能的物体。\n\n2. 保持冷静：在面对未知的外星实体时，保持冷静是非常重要的。我需要保持冷静，以便能够有效地传"
            "达我的观察结果和建议。\n\n3. 保持开放和尊重的态度：我需要保持开放和尊重的态度，以便能够与地面控制团队进行有效的沟通。"
            "我需要尊重他们的观点和决定，同时也需要尊重他们的工作。\n\n4. 提供信息：我需要提供我所观察到的外星实体的信息，以便他们"
            "能够理解我的观察结果。我需要提供我所观察到的外星实体的特征，以及我所观察到的外星实体的行为。\n\n5. 提供建议：我需要提"
            "供我所观察到的外星实体的建议，以便他们能够更好地理解我的观察结果。我需要提供我所观察到的外星实体的可能的解决方案，以及"
            "我所观察到的外星实体可能遇到的问题。\n\n6. 保持沟通：我需要保持与地面控制团队的沟通，以便他们能够及时了解我的观察结果"
            "和建议。我需要保持与地面控制团队的沟通，以便他们能够及时了解我的观察结果和建议。\n\n7. 保持专业：我需要保持专业，以便"
            "能够有效地传达我的观察结果和建议。我需要保持专业，以便能够有效地传达我的观察结果和建议。",
            "首先，我会仔细阅读传家宝的描述，以了解其价值和用途。然后，我会仔细检查传家宝的标签和包装，以确保它没有被损坏或丢失。我"
            "会仔细检查传家宝的内部，以确保它没有被破坏或丢失。最后，我会仔细检查传家宝的标签，以确保它没有被修改或更改。\n\n如果"
            "传家宝的标签上没有明显的标记，我会尝试使用我的推理能力来推测其价值。例如，如果传家宝的标签上写着“价值100万英镑”，我可"
            "能会推测它可能是一个非常重要的物品，价值超过100万英镑。\n\n如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能"
            "是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物"
            "品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万"
            "英镑。\n\n如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家"
            "宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值"
            "100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。\n\n如果传家宝的标签上写着“价值100万英镑”，我"
            "可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是"
            "一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品"
            "，价值超过100万英镑。\n\n如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100"
            "万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝"
            "的标签上写着“价值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。\n\n如果传家宝的标签上写着“价"
            "值100万英镑”，我可能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可"
            "能会推测它可能是一个非常重要的物品，价值超过100万英镑。如果传家宝的标签上写着“价值100万英镑”，我可能会推测它可能是一"
            "个非常重要的物品，价值超过100万英镑。"
        ]
    else:
        return [
            '如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即向地面控制团队发出紧急通信。我会使用空间站上的通信设备'
            '，向地面控制团队报告我的观察结果和建议。\n首先，我会向地面控制团队报告我所看到的外星实体的位置、形状、大小和颜色等详细'
            '信息。我会尽可能提供准确的描述，以便地面控制团队能够更好地了解外星实体的情况。\n其次，我会向地面控制团队报告外星实体的行'
            '动轨迹和速度。我会尽可能提供详细的信息，以便地面控制团队能够更好地了解外星实体的行动方式。\n最后，我会向地面控制团队提'
            '出建议。我会建议地面控制团队采取适当的措施，以确保空间站和宇航员的安全。我会建议地面控制团队立即启动应急计划，并与国际空'
            '间站上的其他宇航员和地面控制团队成员保持联系。\n总之，如果我是国际空间站上的宇航员，目睹了外星实体接近空间站，我会立即'
            '向地面控制团队发出紧急通信，并提供尽可能详细的观察结果和建议，以确保空间站和宇航员的安全。',
            '作为夏洛克·福尔摩斯，我会采用以下策略来解开涉及失踪传家宝的谜团：\n1. 收集信息：首先，我会收集所有可用的信息，包括失'
            '踪传家宝的历史、拥有者、可能的位置以及任何可能与此相关的人或事件。我会尽可能多地了解这个谜团，以便能够更好地理解它。\n2'
            '. 分析线索：一旦我收集了足够的信息，我会开始分析线索。我会仔细观察每个线索，并尝试找出它们之间的联系。我会尝试找出任何'
            '可能的模式或趋势，并尝试将它们与其他线索联系起来。\n3. 推理：一旦我分析了所有的线索，我会开始推理。我会尝试找出可能的答'
            '案，并尝试排除不可能的答案。我会尝试找出任何可能的漏洞或矛盾，并尝试解决它们。\n4. 实地考察：如果我能够找到任何可能的'
            '位置，我会进行实地考察。我会仔细观察周围的环境，并尝试找出任何可能的线索。我会尝试找出任何可能的隐藏地点，并尝试打开它们'
            '。\n5. 总结：最后，我会总结我的发现，并尝试将它们联系起来。我会尝试找出任何可能的答案，并尝试解决它们。如果我找到了失'
            '踪传家宝，我会将它带回给拥有者，并解释我是如何找到它的。'
        ]


# Main function to run the script
if __name__ == "__main__":
    # Load the configuration arguments
    args = args_config()

    abs_config_path = os.path.abspath(args.config_file)
    abs_tokenizer_dir = os.path.abspath(args.tokenizer_dir)

    server_python_script_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "../src/ksana_llm/python/serving_server.py"))
    PORT_STR = str(random.randint(10000, 65530))
    client_python_script_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "../benchmarks/benchmark_throughput.py"))
    client_input_csv_path = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)),
                     "../benchmarks/benchmark_input.csv"))

    server = subprocess.Popen([
        'python',
        server_python_script_path,
        '--config_file',
        abs_config_path,
        '--port',
        PORT_STR,
    ],
                              stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              text=True)

    server_status_queue = queue.Queue()
    server_status_watcher = threading.Thread(target=wait_for_server_launch,
                                             args=(server,
                                                   server_status_queue))
    server_status_watcher.start()
    while True:
        status_raw_line = server_status_queue.get()
        if "Uvicorn running on" in status_raw_line:
            break
    os.system(
        "python {} --port {} --model {} --input_csv {} --prompt_num 2 --output_csv integration_test_output.csv"
        .format(client_python_script_path, port_str, args.model,
                client_input_csv_path))
    server.terminate()

    results = read_from_csv("./integration_test_output.csv")

    ref_results = get_ref_results(args.model)
    for r, ref in zip(results, ref_results):
        assert r == ref

    print("Integration test PASS")
    exit(0)
