import os
import time
import torch
import numpy as np
from transformers import LlamaTokenizer, AutoTokenizer

SEQ_LEN_IN = 32
tokenizer = AutoTokenizer.from_pretrained("../llama_weight", use_fast=False)
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
  exit_data = np.random.randint(10,size=[1,1])
  while(1):
    print("*****************************************")
    question_input = input("Question Input: ")
    if question_input == "exit":
      output_path = "../input/exit.bin"
      exit_data.tofile(output_path)
      print("The program is over, welcome to use next time")
      break
    question = "Common sense questions and answers\n\nQuestion: " + question_input
    # question = questions[i]
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding='max_length', max_length=SEQ_LEN_IN)
    input_0 = inputs.input_ids.numpy()
    # delete pad from question
    target=32000
    index=SEQ_LEN_IN-1
    for j in range(0, SEQ_LEN_IN):
      if target == input_0[0][j]:
        index = j
        break
    input_1 = input_0[:,:index]
    # save max_input_length
    max_input_length = np.int64(index)
    max_input_length.tofile("../input/prompt_len.bin")
    # save aclnngather_index
    aclnngather_index = np.arange(index).astype(np.int64)
    aclnngather_index.tofile("../input/aclnngather_index.bin")
    # save input_1
    input_path = "../input/input_ids_" + str(i) + ".bin"
    input_1.tofile(input_path)
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
    print(f"input_1 shape {input_1.shape}")
    for i_an in anser:
      print(f"anser shape {i_an.shape}")
    all_senes = np.hstack((input_1,anser))

    res1 = tokenizer.batch_decode(all_senes, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    output_list = list(filter(None, "".join(res1).split("\n")))
    print("\n-----The actual Q&A this time is as follows-----")
    print(output_list[1])
    print(output_list[2])
    print("*****************************************\n")
    i += 1
except KeyboardInterrupt:
  output_path = "../input/exit.bin"
  exit_data.tofile(output_path)
  print("\nThe program is over, welcome to use next time")
