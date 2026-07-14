# Copyright 2024 Tencent Inc.  All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from collections import OrderedDict
import time

import tensorrt as trt
import torch

import numpy as np
# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}
if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool


class Engine():

    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.buffers = OrderedDict()
        self.tensors = OrderedDict()
        self.cuda_graph_instance = None  # cuda graph

        self._cache_shape_dict = None
        self._binding_infos = OrderedDict()

    def __del__(self):
        del self.engine
        del self.context
        del self.buffers
        del self.tensors
        del self.cuda_graph_instance

    def load(self):
        print(f"[W] Loading TensorRT engine: {self.engine_path}")
        logger = trt.Logger(trt.Logger.INFO)
        with open(self.engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self._set_binding_infos()

    def _set_binding_infos(self):
        for binding in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(binding)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            self._binding_infos[name] = {
                'binding': binding,
                'shape': shape,
                'dtype': dtype}

    def _allocate_buffers(self, shape_dict=None, device='cuda:0', binding=None):
        def allocate_buffer_for(binding):
            name = self.engine.get_tensor_name(binding)
            if shape_dict and name in shape_dict:
                shape = shape_dict[name]
            else:
                shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.context.set_input_shape(name, shape)
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=device)
            self.tensors[name] = tensor

        if binding is not None:
            allocate_buffer_for(binding)
        else:
            for binding in range(self.engine.num_io_tensors):
                allocate_buffer_for(binding)

    def allocate_buffers(self, cur_shape_dict=None, device='cuda:0'):
        if cur_shape_dict == self._cache_shape_dict:
            return

        if self._cache_shape_dict is None:
            self._allocate_buffers(shape_dict=cur_shape_dict, device=device)
            self._cache_shape_dict = cur_shape_dict
            return

        for tensor_name, cur_shape in cur_shape_dict.items():
            prev_shape = self._cache_shape_dict[tensor_name]
            if prev_shape != cur_shape:
                binding_idx = self._binding_infos[tensor_name]['binding']
                self._allocate_buffers(shape_dict=cur_shape_dict, device=device, binding=binding_idx)
        self._cache_shape_dict = cur_shape_dict


    def infer(self, feed_dict, stream):
        for name, buf in feed_dict.items():
            self.tensors[name].copy_(buf)

        for name, tensor in self.tensors.items():
            self.context.set_tensor_address(name, tensor.data_ptr())

        noerror = self.context.execute_async_v3(stream)
        if not noerror:
            raise ValueError(f"[E] TRT engine inference failed.")

        return self.tensors


    def export_onnx(self, model, sample_input, onnx_path, input_list, output_list, dynamic_list):
        print(f"[W] Start converting ONNX!")
        t0 = time.time()
        torch.onnx.export(model,
                          sample_input,
                          onnx_path,
                          opset_version=17,
                          do_constant_folding=True,
                          input_names=input_list,
                          output_names=output_list,
                          dynamic_axes=dynamic_list
                          )
        t1 = time.time()
        print(f"[W] Build ONNX time : {t1-t0}")

    def build_trt(self, onnx_path, enable_fp16=True, enable_refit=False, input_profile=None):
        print(f"[W] Start converting TRT engine!")
        t0 = time.time()

        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        builder_config = builder.create_builder_config()

        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read(), "/".join(onnx_path.split("/"))):
                print('[E] ONNX file parse error:')
                for error in range(parser.num_errors):
                    print('Error %d: %s', error, parser.get_error(error))
                raise Exception('[E] ONNX file parse error!')

            print("[W] Succeeded parsing %s" % onnx_path)

        if enable_fp16:
            builder_config.set_flag(trt.BuilderFlag.FP16)
        
        if enable_refit:
            builder_config.set_flag(trt.BuilderFlag.REFIT)

        if input_profile:
            profile = builder.create_optimization_profile()
            for name, dims in input_profile.items():
                assert len(dims) == 3
                profile.set_shape(name, dims[0], dims[1], dims[2])

            builder_config.add_optimization_profile(profile)

        builder_config.add_optimization_profile(profile)

        engine = builder.build_serialized_network(network, builder_config)

        if engine is None:
            raise Exception('[E] TensorRT build engine fail')

        with open(self.engine_path, 'wb') as f:
            f.write(engine)

        t1 = time.time()
        print(f"[W] Build TRT engine time : {t1-t0}")
