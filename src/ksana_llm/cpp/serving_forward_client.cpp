/* Copyright 2024 Tencent Inc.  All rights reserved.
==============================================================================*/

#include <getopt.h>
#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <vector>

#include "base64.hpp"
#include "httplib.h"
#include "ksana_llm/utils/request_serial.h"
#include "msgpack.hpp"

struct Args {
  // server host address
  const char *host = "localhost";
  // server port
  int port = 8888;
  // server api
  const char *api = "forward";
};

void PrintUsage(char *program) {
  std::cout << "usage: " << program << " [-h, --help] [-s, --host HOST] [-p, --port PORT] [-a, --api API]\n\n";
  std::cout << "optional arguments:\n";
  std::cout << "  -h, --help       show this help message and exit\n";
  std::cout << "  -s, --host HOST  server host address\n";
  std::cout << "  -p, --port PORT  server port\n";
  std::cout << "  -a, --api  API   server api\n";
  exit(1);
}

// Parse the command line arguments.
Args ParseArgs(int argc, char *argv[]) {
  Args args;
  const option options[] = {{"host", required_argument, nullptr, 's'},
                            {"port", required_argument, nullptr, 'p'},
                            {"api", required_argument, nullptr, 'a'},
                            {"help", no_argument, nullptr, 'h'},
                            {nullptr, 0, nullptr, 0}};

  int opt;
  while ((opt = getopt_long(argc, argv, "s:p:a:h", options, nullptr)) != -1) {
    switch (opt) {
      case 's':  // -s or --host
        args.host = optarg;
        break;
      case 'p':  // -p or --port
        args.port = std::stoi(optarg);
        break;
      case 'a':  // -a or --api
        args.api = optarg;
        break;
      case 'h':  // -h or --help
      case '?':  // unrecognized option
      default:
        PrintUsage(argv[0]);
    }
  }
  return args;
}

// Send request packed in msgpack to the server.
bool PostRequestMsgPack(httplib::Client &cli, const std::string &api, const ksana_llm::BatchRequestSerial &request,
                        ksana_llm::BatchResponseSerial &response) {
  msgpack::sbuffer sbuf;
  msgpack::pack(sbuf, request);

  if (auto rsp = cli.Post("/" + api, sbuf.data(), sbuf.size(), "application/x-msgpack"); !rsp || rsp->status != 200) {
    std::cerr << "Failed to get response: " << rsp << std::endl;
    return false;
  } else {
    auto handle = msgpack::unpack(rsp->body.data(), rsp->body.size());
    auto object = handle.get();
    object.convert(response);
    return true;
  }
}

int main(int argc, char *argv[]) {
  std::cout << std::fixed << std::setprecision(5);

  Args args = ParseArgs(argc, argv);

  // Create an HTTP client to the server.
  httplib::Client cli(args.host, args.port);

  auto start = std::chrono::high_resolution_clock::now();
  // Prepare the request data.
  ksana_llm::BatchRequestSerial request;
  request.requests.push_back(
      ksana_llm::RequestSerial{.prompt = "Hello, world!",  // Set the prompt.
                               .request_target = {ksana_llm::TargetRequestSerial{
                                   // Set a list of targets.
                                   .target_name = "logits",  // Request logits output.
                                   .slice_pos = {{0, 1}},    // Provide a set of sorted intervals of the result.
                                   .token_reduce_mode = "GATHER_TOKEN_ID"}}});
  request.requests.push_back(
      ksana_llm::RequestSerial{.input_tokens = {1, 22, 13},  // Set the input tokens.
                               .request_target = {ksana_llm::TargetRequestSerial{
                                   // Set a list of targets.
                                   .target_name = "logits",  // Request logits output.
                                   .slice_pos = {{-3, -2}},  // Indices can be negative to count from the end.
                                   .token_reduce_mode = "GATHER_TOKEN_ID"}}});

  // Send request to the server and get the response.
  ksana_llm::BatchResponseSerial response;
  PostRequestMsgPack(cli, args.api, request, response);
  auto end = std::chrono::high_resolution_clock::now();

  // Parse the response and show the result.
  for (auto &each_response : response.responses) {
    for (auto &target_response : each_response.response) {
      auto &tensor = target_response.tensor;
      // C++ does not have primitive half precision float type.
      assert(tensor.dtype == "float32");
      auto data_bytes = base64::decode_into<std::vector<uint8_t>>(tensor.data.begin(), tensor.data.end());
      assert(tensor.shape[0] * sizeof(float) == data_bytes.size());
      std::vector<float> data(tensor.shape[0]);
      memcpy(data.data(), data_bytes.data(), data_bytes.size());

      std::cout << "input_token_ids : [ ";
      for (auto token : each_response.input_token_ids) {
        std::cout << token << " ";
      }
      std::cout << "], target_name : " << target_response.target_name << ", tensor : \n";
      std::cout << "[ ";
      for (auto value : data) {
        std::cout << value << " ";
      }
      std::cout << "]" << std::endl;
    }
  }

  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "request duration: " << duration << " ms" << std::endl;

  return 0;
}
