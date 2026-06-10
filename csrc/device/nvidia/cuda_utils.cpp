/* Copyright 2024 Tencent Inc.  All rights reserved.

==============================================================================*/
#include "device/nvidia/cuda_utils.h"

#include "device/nvidia/nccl_utils.h"

namespace ksana_llm {

#ifdef ENABLE_MULTICAST
NvlsMcastMemory* NvlsMcastMemory::GetInstance() {
  static NvlsMcastMemory instance;
  return &instance;
}

// [device_count * unicast_ipc_handle][1 * multicast_ipc_handle]
void NvlsMcastMemory::Initialize(const size_t device_count) { ipc_handles_.resize(device_count + 1); }

std::shared_ptr<NcclIpcSocket>& NvlsMcastMemory::GetNcclIpcSocket() { return nccl_ipc_socket_; }

NvlsHandle* NvlsMcastMemory::GetNvlsHandle() { return nvls_handle_; }

std::vector<IpcMemHandle>& NvlsMcastMemory::GetIpcHandles() { return ipc_handles_; }

uint32_t volatile& NvlsMcastMemory::GetAbortFlag() { return abort_flag_; }

void** NvlsMcastMemory::GetSignalPadsDev() { return signal_pads_dev_; }

#define ROUND_UP(n, multiple) (((n + multiple - 1) / multiple) * multiple)

// Adapted from
// https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/tensorrt_llm/runtime/ipcNvlsMemory.cu
CUmemAllocationHandleType NvlsMcastMemory::GetMemHandleType(const int device_id) {
  // Check if fabric handle support is available.
  int fabric_supported = 0;
  CU_CHECK(cuDeviceGetAttribute(&fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device_id));
  if (!fabric_supported) {
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }

  nvmlDevice_t nvml_device;
  nvmlGpuFabricInfo_t fabric_info;
  NVML_CHECK(nvmlInit_v2());
  NVML_CHECK(nvmlDeviceGetHandleByIndex(device_id, &nvml_device));
  NVML_CHECK(nvmlDeviceGetGpuFabricInfo(nvml_device, &fabric_info));
  NVML_CHECK(nvmlShutdown());

  // Check if the fabric is fully initialized.
  if (fabric_info.state != NVML_GPU_FABRIC_STATE_COMPLETED || fabric_info.status != NVML_SUCCESS) {
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }

  // Check that fabric handles can be created.
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(CUmemAllocationProp));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_FABRIC;

  size_t alloc_size = 1024;  // anything > 0
  size_t min_gran = 0;
  CU_CHECK(cuMemGetAllocationGranularity(&min_gran, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  alloc_size = ROUND_UP(alloc_size, min_gran);

  CUmemGenericAllocationHandle handle;
  CUresult err = cuMemCreate(&handle, alloc_size, &prop, 0);
  if (err == CUDA_ERROR_NOT_PERMITTED || err == CUDA_ERROR_NOT_SUPPORTED) {
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }

  // Check if fabric handles can be exported & imported by IMEX (Internode Memory Exchange)
  CUmemFabricHandle fh;
  err = cuMemExportToShareableHandle(&fh, handle, CU_MEM_HANDLE_TYPE_FABRIC, 0);
  if (err != CUDA_SUCCESS ||
      (err = cuMemImportFromShareableHandle(&handle, &fh, CU_MEM_HANDLE_TYPE_FABRIC)) != CUDA_SUCCESS) {
    CU_CHECK(cuMemRelease(handle));
    return CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
  }

  CU_CHECK(cuMemRelease(handle));
  // If we get here, fabric handles are supported.
  return CU_MEM_HANDLE_TYPE_FABRIC;
}

void NvlsMcastMemory::AllocMcastMemory(const size_t size) {
  // Prevent duplicate execution
  if (nvls_handle_ != nullptr) {
    return;
  }

  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  const size_t device_count = ipc_handles_.size() - 1;

  // Get runtime and driver device IDs.
  int CU_dev;
  CU_CHECK(cuDeviceGet(&CU_dev, device_id));

  nvls_handle_ = new NvlsHandle();

  constexpr size_t kSignalPadAlignment = 16UL;
  nvls_handle_->signal_pad_offset = (size + kSignalPadAlignment - 1) & ~(kSignalPadAlignment - 1);
  nvls_handle_->size = nvls_handle_->signal_pad_offset + kSignalPadSize;

  // Get handle type used to share memory handles between devices.
  nvls_handle_->mem_handle_type = GetMemHandleType(device_id);

  // Define allocation access permissions (same for unicast and multicast).
  memset(&nvls_handle_->access_desc, 0, sizeof(CUmemAccessDesc));
  nvls_handle_->access_desc.location.id = device_id;
  nvls_handle_->access_desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  nvls_handle_->access_desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  // Define unicast allocation properties.
  CUmemAllocationProp prop;
  memset(&prop, 0, sizeof(CUmemAllocationProp));
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_id;
  prop.requestedHandleTypes = nvls_handle_->mem_handle_type;

  // Define multicast allocation properties.
  memset(&nvls_handle_->mcprop, 0, sizeof(CUmulticastObjectProp));
  nvls_handle_->mcprop.numDevices = device_count;
  nvls_handle_->mcprop.handleTypes = nvls_handle_->mem_handle_type;
  nvls_handle_->mcprop.flags = 0;

  // Round up allocation size to the nearest multiple of the unicast allocation granularity.
  CU_CHECK(cuMemGetAllocationGranularity(&nvls_handle_->granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  nvls_handle_->size = ROUND_UP(nvls_handle_->size, nvls_handle_->granularity);

  // Round up allocation size to the nearest multiple of the multicast allocation granularity.
  CU_CHECK(cuMulticastGetGranularity(&nvls_handle_->mc_granularity, &nvls_handle_->mcprop,
                                     CU_MULTICAST_GRANULARITY_MINIMUM));
  nvls_handle_->size = ROUND_UP(nvls_handle_->size, nvls_handle_->mc_granularity);
  nvls_handle_->mcprop.size = nvls_handle_->size;

  // Allocate physical pages of memory on GPU.
  CU_CHECK(cuMemCreate(&nvls_handle_->uc_handle, nvls_handle_->size, &prop, 0));
  // Reserve unicast virtual address space for the memory.
  CU_CHECK(cuMemAddressReserve(&nvls_handle_->uc_va, nvls_handle_->size, nvls_handle_->granularity, 0U, 0));
  // Map the unicast virtual address space to the physical pages.
  CU_CHECK(cuMemMap(nvls_handle_->uc_va, nvls_handle_->size, 0, nvls_handle_->uc_handle, 0));
  // Set the access permissions for the unicast memory.
  CU_CHECK(cuMemSetAccess(nvls_handle_->uc_va, nvls_handle_->size, &nvls_handle_->access_desc, 1));
  nvls_handle_->uc_ptr = reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(nvls_handle_->uc_va));

  // Export unicast pointer.
  CU_CHECK(cuMemExportToShareableHandle(&ipc_handles_[device_id], nvls_handle_->uc_handle,
                                        nvls_handle_->mem_handle_type,
                                        /*flags*/ 0));

  if (device_id == 0) {
    CU_CHECK(cuMulticastCreate(&nvls_handle_->mc_handle, &nvls_handle_->mcprop));
    // Export multicast pointer.
    CU_CHECK(cuMemExportToShareableHandle(&ipc_handles_.back(), nvls_handle_->mc_handle, nvls_handle_->mem_handle_type,
                                          /*flags*/ 0));
  }
}

void NvlsMcastMemory::InitMcastMemory() {
  // Prevent duplicate execution
  if (!nvls_handle_->peer_uc_vas.empty()) {
    return;
  }
  KLLM_CHECK_WITH_INFO(nvls_handle_->mem_handle_type == CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                       "Only support posix file descriptor");

  int device_id;
  CUDA_CHECK(cudaGetDevice(&device_id));
  const size_t device_count = ipc_handles_.size() - 1;

  // Get runtime and driver device IDs.
  int CU_dev;
  CU_CHECK(cuDeviceGet(&CU_dev, device_id));

  nvls_handle_->peer_uc_vas.resize(device_count);
  nvls_handle_->peer_uc_handles.resize(device_count);

  // Import unicast addresses from all ranks.
  for (size_t i = 0; i < device_count; i++) {
    if (i == static_cast<size_t>(device_id)) {
      nvls_handle_->peer_uc_vas[i] = nvls_handle_->uc_va;
      nvls_handle_->peer_uc_handles[i] = nvls_handle_->uc_handle;
    } else {
      IpcMemHandle peer_ipc_handle = ipc_handles_[i];
      void* os_handle = nvls_handle_->mem_handle_type == CU_MEM_HANDLE_TYPE_FABRIC
                            ? &peer_ipc_handle
                            : reinterpret_cast<void*>(peer_ipc_handle.fd);
      CU_CHECK(
          cuMemImportFromShareableHandle(&nvls_handle_->peer_uc_handles[i], os_handle, nvls_handle_->mem_handle_type));
      // Reserve peer unicast virtual address space for the memory.
      CU_CHECK(
          cuMemAddressReserve(&nvls_handle_->peer_uc_vas[i], nvls_handle_->size, nvls_handle_->granularity, 0U, 0));
      // Map the peer unicast virtual address space to the physical pages.
      CU_CHECK(cuMemMap(nvls_handle_->peer_uc_vas[i], nvls_handle_->size, 0, nvls_handle_->peer_uc_handles[i], 0));
      // Set the access permissions for the peer unicast memory.
      CU_CHECK(cuMemSetAccess(nvls_handle_->peer_uc_vas[i], nvls_handle_->size, &nvls_handle_->access_desc, 1));
    }
  }

  if (device_id != 0) {
    // Import multicast address from rank 0.
    IpcMemHandle root_ipc_handle = ipc_handles_.back();
    void* os_handle = nvls_handle_->mem_handle_type == CU_MEM_HANDLE_TYPE_FABRIC
                          ? &root_ipc_handle
                          : reinterpret_cast<void*>(root_ipc_handle.fd);
    CU_CHECK(cuMemImportFromShareableHandle(&nvls_handle_->mc_handle, os_handle, nvls_handle_->mem_handle_type));
  }

  // Add device to multicast object
  CU_CHECK(cuMulticastAddDevice(nvls_handle_->mc_handle, CU_dev));
  // Bind physical memory to the Multicast group.
  // Note: It will block until all ranks have been added to the group.
  CU_CHECK(cuMulticastBindMem(nvls_handle_->mc_handle, 0, nvls_handle_->uc_handle, 0, nvls_handle_->size, 0));
  // Reserve multicast virtual address space for the memory.
  CU_CHECK(cuMemAddressReserve(&nvls_handle_->mc_va, nvls_handle_->size, nvls_handle_->mc_granularity, 0U, 0));
  // Map the multicast virtual address space to the physical pages.
  CU_CHECK(cuMemMap(nvls_handle_->mc_va, nvls_handle_->size, 0, nvls_handle_->mc_handle, 0));
  // Set the access permissions for the multicast memory.
  CU_CHECK(cuMemSetAccess(nvls_handle_->mc_va, nvls_handle_->size, &nvls_handle_->access_desc, /*count*/ 1));
  nvls_handle_->mc_ptr = reinterpret_cast<uintptr_t>(reinterpret_cast<void*>(nvls_handle_->mc_va));

  // Initialize signal pads
  std::vector<CUdeviceptr> signal_pads(device_count);
  for (size_t i = 0; i < device_count; i++) {
    signal_pads[i] = nvls_handle_->peer_uc_vas[i] + nvls_handle_->signal_pad_offset;
    if (i == static_cast<size_t>(device_id)) {
      CU_CHECK(cuMemsetD8(signal_pads[i], 0, kSignalPadSize));
    }
  }
  // Copy host array of pointers to device array
  CUDA_CHECK(cudaMalloc(&signal_pads_dev_, device_count * sizeof(CUdeviceptr)));
  CUDA_CHECK(
      cudaMemcpy(signal_pads_dev_, signal_pads.data(), device_count * sizeof(CUdeviceptr), cudaMemcpyHostToDevice));
}

void NvlsMcastMemory::FreeMcastMemory() {
  if (nvls_handle_ != nullptr) {
    CU_CHECK(cuMemUnmap(nvls_handle_->mc_va, nvls_handle_->size));
    CU_CHECK(cuMemRelease(nvls_handle_->mc_handle));
    CU_CHECK(cuMemAddressFree(nvls_handle_->mc_va, nvls_handle_->size));
    for (size_t i = 0; i < nvls_handle_->peer_uc_vas.size(); ++i) {
      CU_CHECK(cuMemUnmap(nvls_handle_->peer_uc_vas[i], nvls_handle_->size));
      CU_CHECK(cuMemRelease(nvls_handle_->peer_uc_handles[i]));
      CU_CHECK(cuMemAddressFree(nvls_handle_->peer_uc_vas[i], nvls_handle_->size));
    }
    delete nvls_handle_;
  }
  CUDA_CHECK(cudaFree(signal_pads_dev_));
}
#endif

}  // namespace ksana_llm
