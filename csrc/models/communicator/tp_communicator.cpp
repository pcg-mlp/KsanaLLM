/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/

#include "models/communicator/tp_communicator.h"

#include "device/device_utils.h"

namespace ksana_llm {

Status TpCommunicator::AllReduce(std::vector<Tensor>& reduce_buffer_tensors,
                                 std::vector<Tensor>& hidden_buffer_tensors_0, const bool is_multi_token_forward,
                                 ForwardingContext& forwarding_context) {
  // NOTE(karlluo): multiple event in nccl will cause preformance regression
  // nccl multiple event just enable when context.IsRunContextDecodeAndDecodeSerially() == false
  PROFILE_EVENT_SCOPE(AllReduce, fmt::format("TpCommunicator_AllReduce_seq{}", reduce_buffer_tensors[0].shape[0]));
  if (!forwarding_context.GetContext()->global->IsRunContextDecodeAndDecodeSerially()) {
    EventRecord(forwarding_context.GetModelOutput()->compute_ready_event,
                forwarding_context.GetContext()->device->GetPrimaryComputeStream());
    StreamWaitEvent(forwarding_context.GetContext()->device->GetCommStream(),
                    forwarding_context.GetModelOutput()->compute_ready_event);
  }

  // AllReduceSum
  if (forwarding_context.GetModelCommunicator()) {
    forwarding_context.GetModelCommunicator()->ReduceSum(reduce_buffer_tensors, hidden_buffer_tensors_0,
                                                         is_multi_token_forward, /*use_custom*/ true);
  }
  return Status();
}

Status TpCommunicator::AllGather(Tensor& gather_tensor, Tensor& buffer, ForwardingContext& forwarding_context) {
  if (!forwarding_context.GetModelCommunicator()) {
    return Status();
  }

  std::vector<Tensor> input{gather_tensor, buffer};
  std::vector<Tensor> output{gather_tensor};
  forwarding_context.GetModelCommunicator()->AllGather(input, output);
  gather_tensor = std::move(output[0]);
  return Status();
}

}  // namespace ksana_llm
