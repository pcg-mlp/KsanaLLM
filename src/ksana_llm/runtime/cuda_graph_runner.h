/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include <cuda_runtime.h>
#include <unordered_map>
#include <vector>

namespace ksana_llm {

// The class used to build a cuda graph.
class CudaGraphBuilder {
  public:
    // Begin and end the graph capture.
    void BeginCapture(cudaStream_t stream);
    void EndCapture(cudaStream_t stream);

    // Get the captured graph.
    cudaGraphExec_t GetGraphExec();

  private:
    // Whether has been instantiated.
    bool initialized_ = false;

    // The graph object.
    cudaGraph_t graph_;

    // The executable graph instance.
    cudaGraphExec_t exec_graph_;
};

// Used to run inference forward with cuda graph.
class CudaGraphRunner {
  public:
    // Launch cuda graph with specified batch_size.
    void LaunchGraph(size_t batch_size, cudaStream_t stream);

    // Capture graphs for batch size 1, 2, 4, 8, 16, 24, 32, 40, ..., 256.
    void GetGraphBatchSizes(std::vector<size_t>& batch_sizes);

    // Whether the cuda graph is available.
    bool CheckGraphAvailable(size_t batch_size);

    // Get the paded batch size.
    size_t GetPaddedBatchSize(size_t batch_size);

  private:
    // batch_size => graph instance.
    std::unordered_map<size_t, cudaGraphExec_t> graph_instances_;

    // The batch size config for cuda graph.
    size_t max_batch_size_ = 256;
    size_t batch_size_step_ = 8;
};

}  // namespace ksana_llm
