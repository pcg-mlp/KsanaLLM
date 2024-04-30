import os
import time
import torch
import numpy as np
from transformers import LlamaTokenizer, AutoTokenizer
import sys
import pdb

MAX_PROMPT_LEN = 1024

model_type = "7b"
if len(sys.argv) > 1 :
  model_type = sys.argv[1]
assert model_type in ["7b", "13b"]

model_type_to_path_map = {"7b": "../llama_weight", "13b" : "../acl_llama13b" }
model_path=model_type_to_path_map.get(model_type)

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print('''Welcome to use the llama7b inference application based on aclnn. 
The application is Common sense questions and answers, You can ask your own question or use the questions below.
- Who was the first president of the United States
- Why should we learn math
- What is the capital of France
- How to learn a new language
- Should I take notes during class
If you need to exit, you can enter "exit" or directly ctrl+c
''')
try:
  i = 0
  exit_data = np.random.randint(10, size=[1,1])
  while(1):
    print("*****************************************")
    question_input = input("Question Input: ")
    if question_input == "exit":
      output_path = "../input/exit.bin"
      exit_data.tofile(output_path)
      print("The program is over, welcome to use next time")
      break
    # question = "Common sense questions and answers\n\nQuestion: " + question_input
    question = question_input
    # question = questions[i]
    inputs = tokenizer(question, return_tensors="pt")
    # inputs = tokenizer(question, return_tensors="pt", truncation=True, padding='max_length', max_length=MAX_PROMPT_LEN)
    input_ids = inputs.input_ids.numpy()
    input_idx = input_ids[:][:MAX_PROMPT_LEN]
    print(f"input ids: {input_idx}")

    # save prompt_len
    prompt_len = np.int64(len(input_idx[0]))
    prompt_len.tofile("../input/prompt_len.bin")
    # save input_idx
    input_path = "../input/input_ids_" + str(i) + ".bin"
    input_idx.tofile(input_path)
    start_time = time.time()
    output_path = "../output/fin_result_" + str(i) + ".bin"

    while(1):
      if not os.path.isfile(output_path):
        time.sleep(1)
        continue
      else:
        break
    stop_time = time.time()
    print(f"it takes {(stop_time - start_time):.2f} seconds ")
    anser = [np.fromfile(output_path,dtype=np.int64)]
    print(f"input_idx shape {input_idx.shape}")
    for i_an in anser:
      print(f"anser shape {i_an.shape}")
    all_senes = np.hstack((input_idx, anser))
    res1 = tokenizer.batch_decode(all_senes, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_list = list(filter(None, "".join(res1).split("\n")))
    print("\n-----The actual Q&A this time is as follows-----")
    for x in output_list:
      print(x)    
    print("*****************************************\n")
    pdb.set_trace()
    i += 1
except KeyboardInterrupt:
  output_path = "../input/exit.bin"
  exit_data.tofile(output_path)
  print("\nThe program is over, welcome to use next time")
