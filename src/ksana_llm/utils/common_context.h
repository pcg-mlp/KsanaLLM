/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#pragma once

#include <vector>
#include "ksana_llm/utils/device_utils.h"

namespace ksana_llm {

// The extension type define.
template <int T>
struct ExtensionTypeTraits {
    typedef DummyClass value_type;
};

// The global context, like cuda stream, nccl handler.
template <int T>
class ContextT {
  public:
    ContextT(const int tensor_parallel_size, const int pipeline_parallel_size);
    ~ContextT();

    int GetTensorParallelSize() { return tensor_parallel_size_; }

    int GetPipeLineParallelSize() { return pipeline_parallel_size_; }

    inline bool IsRunContextDecodeAndDecodeSerially() { return is_contextdecode_and_decode_run_serially_; }

    std::vector<Stream>& GetMemoryManageStreams() { return memory_manage_streams_; }

    std::vector<Stream>& GetComputeStreams() { return compute_streams_; }

    std::vector<Stream>& GetH2DStreams() { return h2d_streams_; }

    std::vector<Stream>& GetD2HStreams() { return d2h_streams_; }

    std::vector<Stream>& GetD2DStreams() { return d2d_streams_; }

    std::vector<Stream>& GetNCCLStreams() { return nccl_streams_; }

  public:
    friend class ExtensionTypeTraits<T>::value_type;
    typename ExtensionTypeTraits<T>::value_type* ext = nullptr;

  private:
    // init streams
    void InitStreams(const int worker_id);

  private:
    int device_num_{0};
    int tensor_parallel_size_{0};
    int pipeline_parallel_size_{0};
    const int defalt_device_num_{0};
    int driver_version_;
    // if true, only one thread execute context_decode/decode and context_decode decode run in sync
    // TODO(karlluo): load from environment
    bool is_contextdecode_and_decode_run_serially_{true};

    // streams
    std::vector<Stream> memory_manage_streams_;
    std::vector<Stream> compute_streams_;
    std::vector<Stream> h2d_streams_;
    std::vector<Stream> d2h_streams_;
    std::vector<Stream> d2d_streams_;
    std::vector<Stream> nccl_streams_;

  private:
    // Initialize and destroy extension, implemented by device.
    void InitializeExtension();
    void DestroyExtension();
};

}  // namespace ksana_llm
