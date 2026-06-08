# pylint: disable=broad-except
# Copyright 2024 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from qwen_vl_utils.vision_process import (
    extract_vision_info,
    fetch_image,
    smart_resize,
)
from transformers import AutoProcessor, AutoConfig

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir: ./KsanaLLM/python/ksana_plugin
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# grandparent_dir: ./KsanaLLM/python
grandparent_dir = os.path.abspath(os.path.join(current_dir, os.pardir, os.pardir))
sys.path.append(grandparent_dir)
# add current_dir to import modeling files alongside this file
if current_dir not in sys.path:
    sys.path.append(current_dir)

from plugin_utils import free_cache
from plugin_utils import load_safetensors, get_weight_map
from plugin_utils import check_input
from ksana_encoder.encoder_client import EncoderClient


TIMING_KEYS = (
    "encoder_send_time",
    "encoder_queue_time",
    "encoder_encode_time",
    "encoder_recv_time",
)


def load_state_dict_from_safetensors(model_path):
    weight_map_files = get_weight_map(model_path, "visual.")
    visual_weights = {}
    for weight_map_file in weight_map_files:
        weight_file = os.path.join(model_path, weight_map_file)
        if os.path.splitext(weight_file)[1] == ".safetensors":
            weights = load_safetensors(weight_file)
        else:
            weights = torch.load(
                weight_file, map_location=torch.device("cpu"), weights_only=True
            )
        for name, tensor in weights.items():
            if "visual." in name:
                # Strip everything up to and including "visual." prefix.
                # Qwen2-VL / Qwen2.5-VL weights: "visual.X.Y" -> "X.Y"
                key = name[name.index("visual.") + len("visual.") :]
                visual_weights[key] = tensor
    return visual_weights


class KsanaPlugin:
    """
    Qwen2-VL / Qwen2.5-VL plugin for KsanaLLM.
    Handles vision preprocessing and encoder inference.
    """

    def __init__(self):
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        self.mm_encoder = None
        self.using_remote_encoder = False
        self.multimodal_cache = None

    # Plugin initialization is automatically invoked upon service startup.
    def init_plugin(self, **kwargs):
        if "preprocess" in kwargs:
            self.encoder_url = kwargs.get("encoder_url", None)
            if self.encoder_url is not None:
                self.using_remote_encoder = True
                self.encoder_timeout = 1000 * kwargs["encoder_timeout"]

            self.multimodal_cache = kwargs.get("multimodal_cache", None)

            self.model_path = kwargs["model_path"]

            self.config = AutoConfig.from_pretrained(
                self.model_path, trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(self.model_path)
            self.tokenizer = self.processor.tokenizer
            self.image_processor = self.processor.image_processor
            self.video_processor = getattr(self.processor, "video_processor", None)
            self.image_resize_factor = (
                self.image_processor.patch_size * self.image_processor.merge_size
            )
            self.precision = getattr(self.config, "torch_dtype", None) or torch.bfloat16

            # 预计算 rescale 和 normalize 的融合系数
            # 1/255 用于将 uint8 [0, 255] 缩放到 float [0, 1] 范围
            # 除以 ip_std 则是应用 Normalize 的标准差
            # 两者融合后的 scale系数 = (1/255) / std_dev
            ip_mean = np.array(self.image_processor.image_mean, dtype=np.float32)
            ip_std = np.array(self.image_processor.image_std, dtype=np.float32)
            self._fused_scale = torch.tensor(
                1.0 / (255.0 * ip_std), dtype=torch.float32
            )
            self._fused_bias = torch.tensor(-ip_mean / ip_std, dtype=torch.float32)
            self._patch_size = self.image_processor.patch_size
            self._merge_size = self.image_processor.merge_size
            self._temporal_patch_size = self.image_processor.temporal_patch_size

            self.init_encoder()

            model_type = self.config.model_type
            print(f"[I] Initializing the {model_type} Torch model successfully!")

            free_cache()

            return {"plugin_trt": False}

        if "postprocess" in kwargs:
            return

    @staticmethod
    def _fetch_image_cv(
        ele: dict, size_factor: int, return_numpy: bool = False
    ) -> "Image.Image | np.ndarray":
        """OpenCV-accelerated image loading for local files. Falls back to PIL for others."""

        def _fallback():
            pil_img = fetch_image(ele, size_factor=size_factor)
            return np.array(pil_img) if return_numpy else pil_img

        image = ele.get("image") or ele.get("image_url")

        is_local_path = isinstance(image, str) and not image.startswith(
            ("http://", "https://", "data:image")
        )
        if image is None or not is_local_path:
            return _fallback()

        path = image[7:] if image.startswith("file://") else image

        img_bgr = cv2.imread(path)
        if img_bgr is None:
            return _fallback()

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img_rgb.shape[:2]

        if "resized_height" in ele and "resized_width" in ele:
            rh, rw = smart_resize(
                ele["resized_height"], ele["resized_width"], factor=size_factor
            )
        else:
            min_pixels = ele.get("min_pixels", None)
            max_pixels = ele.get("max_pixels", None)
            rh, rw = smart_resize(
                h_orig,
                w_orig,
                factor=size_factor,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
            )

        if (h_orig, w_orig) != (rh, rw):
            img_rgb = cv2.resize(img_rgb, (rw, rh), interpolation=cv2.INTER_CUBIC)

        if return_numpy:
            return img_rgb

        return Image.fromarray(img_rgb)

    def _process_images_torch(self, images_np: list) -> dict:
        """Torch-native image processor with bf16-early conversion."""
        patch_size = self._patch_size
        merge_size = self._merge_size
        temporal_patch_size = self._temporal_patch_size

        all_patches = []
        all_grid_thw = []

        for img_np in images_np:
            h, w = img_np.shape[:2]
            grid_t = 1
            grid_h = h // patch_size
            grid_w = w // patch_size

            t = torch.from_numpy(img_np).to(dtype=torch.float32)
            t.mul_(self._fused_scale).add_(self._fused_bias)
            t = t.to(dtype=torch.bfloat16)
            t = t.permute(2, 0, 1).contiguous()

            t = t.unsqueeze(0).expand(temporal_patch_size, -1, -1, -1).contiguous()

            t = t.reshape(
                grid_t,
                temporal_patch_size,
                3,
                grid_h // merge_size,
                merge_size,
                patch_size,
                grid_w // merge_size,
                merge_size,
                patch_size,
            )
            # 重排维度，为 ViT 模型输入打包时空 patch：
            # (grid_t, grid_h // merge_size, grid_w // merge_size) -> 输出 token 所在的空间网格位置
            # (merge_size, merge_size) -> 2x2 的合并块
            # 3 -> RGB 通道
            # temporal_patch_size -> 时间深度
            # (patch_size, patch_size) -> 单个 14x14 的图像块
            t = t.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).contiguous()
            t = t.reshape(
                grid_t * grid_h * grid_w,
                3 * temporal_patch_size * patch_size * patch_size,
            )

            all_patches.append(t)
            all_grid_thw.append([grid_t, grid_h, grid_w])

        pixel_values = (
            torch.cat(all_patches, dim=0) if len(all_patches) > 1 else all_patches[0]
        )
        image_grid_thw = torch.tensor(all_grid_thw, dtype=torch.long)

        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    def _init_local_encoder(self):
        if self.config.model_type == "qwen2_5_vl":
            from Qwen2_5VL.modeling import KsanaQwen25VLVisionModel, _HAS_VIT_FA3
            model_cls = KsanaQwen25VLVisionModel
        else:
            from Qwen2VL.modeling import KsanaQwen2VLVisionModel, _HAS_VIT_FA3
            model_cls = KsanaQwen2VLVisionModel

        visual = None
        attn_impl_used = None
        for attn_impl in ["flash_attention_3", "flash_attention_2", "sdpa"]:
            try:
                visual = model_cls._from_config(
                    self.config.vision_config,
                    attn_implementation=attn_impl,
                    dtype=self.precision,
                )
                attn_impl_used = attn_impl
                break
            except Exception as e:
                print(f"[W] {attn_impl} unavailable ({e}), trying next...")
        if visual is None:
            raise RuntimeError(
                "No available attention implementation for vision encoder."
            )
        visual = visual.to(dtype=self.precision)
        visual.load_state_dict(load_state_dict_from_safetensors(self.model_path))

        # Recompute rotary embedding inv_freq in FP32
        for module in visual.modules():
            if (
                hasattr(module, "inv_freq")
                and hasattr(module, "dim")
                and hasattr(module, "theta")
            ):
                inv_freq = 1.0 / (
                    module.theta
                    ** (torch.arange(0, module.dim, 2, dtype=torch.float) / module.dim)
                )
                module.inv_freq = inv_freq

        visual = visual.to(device=self.device).eval()
        self.mm_encoder = visual

        if _HAS_VIT_FA3:
            runtime_attn = "sgl_kernel FA3"
        else:
            runtime_attn = f"transformers {attn_impl_used} dispatch"
        print(
            f"[I] Vision encoder: model={model_cls.__name__}, "
            f"attn_implementation={attn_impl_used}, runtime_attention={runtime_attn}"
        )

    def _init_remote_encoder(self):
        streaming = os.environ.get("KSANA_ENABLE_ENCODER_STREAMING", "0") == "1"
        if streaming:
            from ksana_encoder.encoder_client import StreamingEncoderClient

            self.mm_encoder = StreamingEncoderClient(
                encoder_url=self.encoder_url,
                timeout=self.encoder_timeout,
                streaming=True,
            )
        else:
            self.mm_encoder = EncoderClient(
                encoder_url=self.encoder_url, timeout=self.encoder_timeout
            )

    def init_encoder(self):
        if not self.using_remote_encoder:
            self._init_local_encoder()
        else:
            self._init_remote_encoder()

    @torch.no_grad()
    def _local_encode(self, inputs: Dict) -> Dict:
        pixel_values = inputs["pixel_values"].to(self.device)
        grid_thw = inputs["grid_thw"].to(self.device)
        output = self.mm_encoder(pixel_values, grid_thw)
        if not isinstance(output, torch.Tensor):
            embeds = output.pooler_output
        else:
            embeds = output
        return {"embeds": embeds.cpu()}

    def _remote_encode(self, inputs: Dict) -> Dict:
        return self.mm_encoder(inputs)

    def _encode_single(self, inputs: Dict) -> Dict:
        cache_key = inputs.get("mm_item_key", None)
        use_cache = cache_key is not None and self.multimodal_cache is not None

        if use_cache:
            cached_result = self.multimodal_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        if not self.using_remote_encoder:
            result = self._local_encode(inputs)
        else:
            result = self._remote_encode(inputs)

        if use_cache:
            self.multimodal_cache.put(cache_key, result)
        return result

    def encode(self, inputs: Dict) -> Dict:
        if "pixel_values" in inputs and "grid_thw" in inputs:
            return self._encode_single(inputs)

        result = {}
        for modality in ("image", "video"):
            pv_key = f"{modality}_pixel_values"
            if pv_key not in inputs:
                continue
            encode_input = {
                "pixel_values": inputs[pv_key],
                "grid_thw": inputs[f"{modality}_grid_thw"],
            }
            output = self._encode_single(encode_input)
            result[f"{modality}_embeds"] = output["embeds"]
            for key in TIMING_KEYS:
                if key in output:
                    result[key] = result.get(key, 0) + output[key]
        return result

    def _expand_image_tokens(self, text, image_grid_thw):
        merge_length = self.image_processor.merge_size ** 2
        index = 0
        image_token = self.processor.image_token
        for i in range(len(text)):
            while image_token in text[i]:
                num_tokens = image_grid_thw[index].prod() // merge_length
                text[i] = text[i].replace(
                    image_token, "<|placeholder|>" * num_tokens, 1
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", image_token)
        return text

    def _prepare_inputs(self, messages, additional_params):
        """Tokenize messages and extract pixel values and grid info."""
        config = self.config
        messages = KsanaPlugin.convert_openai_messages(messages, additional_params)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        vision_infos = extract_vision_info(messages)
        image_vision_infos = [
            vi for vi in vision_infos if "image" in vi or "image_url" in vi
        ]
        has_video = any("video" in vi for vi in vision_infos)

        if has_video:
            image_inputs_pv, video_inputs_raw = process_vision_info(messages)

            image_inputs = None
            if image_vision_infos:
                image_inputs = [
                    self._fetch_image_cv(vi, size_factor=self.image_resize_factor)
                    for vi in image_vision_infos
                ] or None

            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else image_inputs_pv,
                videos=video_inputs_raw if video_inputs_raw else None,
                padding=True,
                return_tensors="pt",
            )
        else:
            image_data = {}
            if image_vision_infos:
                image_np_inputs = [
                    self._fetch_image_cv(
                        vi, size_factor=self.image_resize_factor, return_numpy=True
                    )
                    for vi in image_vision_infos
                ]
                image_data = self._process_images_torch(image_np_inputs)

            text_list = [text]
            if image_data:
                text_list = self._expand_image_tokens(
                    text_list, image_data["image_grid_thw"]
                )

            text_inputs = self.tokenizer(text_list, padding=True, return_tensors="pt")

            inputs = dict(text_inputs)
            inputs.update(image_data)

        input_tokens = inputs["input_ids"][0].tolist()

        vision_srt = [
            int(pos + 1)
            for pos, id in enumerate(input_tokens)
            if id == config.vision_start_token_id
        ]
        vision_end = [
            int(pos - 1)
            for pos, id in enumerate(input_tokens)
            if id == config.vision_end_token_id
        ]

        image_grid_thw = inputs.get("image_grid_thw", [])
        video_grid_thw = inputs.get("video_grid_thw", [])

        return (
            inputs,
            input_tokens,
            vision_srt,
            vision_end,
            image_grid_thw,
            video_grid_thw,
        )

    def _scatter_embeds(
        self,
        encode_output,
        input_tokens,
        vision_srt,
        vision_end,
        image_grid_thw,
        video_grid_thw,
        inputs,
    ):
        """Scatter encoder outputs back into vision span order."""
        config = self.config

        permute = {}

        def _scatter_modality(embeds, token_id):
            sections, idx = [], []
            for i, srt in enumerate(vision_srt):
                if input_tokens[srt] == token_id:
                    sections.append(vision_end[i] - srt + 1)
                    idx.append(i)
            split_embeds = torch.split(embeds, sections, dim=0)
            for i in range(len(split_embeds)):
                permute[idx[i]] = split_embeds[i]

        if "image_embeds" in encode_output:
            _scatter_modality(encode_output["image_embeds"], config.image_token_id)
        if "video_embeds" in encode_output:
            _scatter_modality(encode_output["video_embeds"], config.video_token_id)

        if "embeds" in encode_output:
            for token_id in [config.image_token_id, config.video_token_id]:
                sections = []
                idx = []
                for i, srt in enumerate(vision_srt):
                    if input_tokens[srt] == token_id:
                        sections.append(vision_end[i] - srt + 1)
                        idx.append(i)
                if sections and sum(sections) == encode_output["embeds"].shape[0]:
                    _scatter_modality(encode_output["embeds"], token_id)
                    break

        embeds = []
        for i in range(len(permute)):
            embeds.append(permute[i])

        position_ids, mrope_position_deltas = self._get_input_positions(
            input_tokens, image_grid_thw, video_grid_thw
        )
        position_ids = position_ids.squeeze(0).contiguous()
        mrope_position_deltas = torch.LongTensor([mrope_position_deltas.item()])

        additional_tensors = [position_ids, mrope_position_deltas]

        return input_tokens, vision_srt, embeds, additional_tensors

    @staticmethod
    def _write_to_ksana_input(
        ksana_python_input, input_tokens, vision_srt, embeds, additional_tensors
    ):
        ksana_python_input.input_tokens = input_tokens
        ksana_python_input.input_refit_embedding.pos = vision_srt
        ksana_python_input.input_refit_embedding.embedding_tensors = embeds
        ksana_python_input.input_refit_embedding.additional_tensors = additional_tensors

    # Method for pre-processing
    def preprocess(self, **kwargs):
        if not check_input(["ksana_python_input", "messages"], **kwargs):
            raise RuntimeError("Plugin preprocess wrong input.")

        encode_input, context = self.preprocess_before_encode(**kwargs)
        if context is None:
            return
        encode_output = self.encode(encode_input)
        self.preprocess_after_encode(encode_output, context, **kwargs)

    def preprocess_before_encode(self, **kwargs):
        """Phase 1: CPU work before encoder I/O."""
        messages: Optional[List[Dict]] = kwargs["messages"]
        if messages is None:
            return {}, None
        additional_params = kwargs.get("additional_params", {})

        inputs, input_tokens, vision_srt, vision_end, image_grid_thw, video_grid_thw = (
            self._prepare_inputs(messages, additional_params)
        )

        encode_input = {}
        if "pixel_values" in inputs:
            encode_input["image_pixel_values"] = inputs["pixel_values"].to(
                dtype=self.precision
            )
            encode_input["image_grid_thw"] = inputs["image_grid_thw"]
        if "pixel_values_videos" in inputs:
            encode_input["video_pixel_values"] = inputs["pixel_values_videos"].to(
                dtype=self.precision
            )
            encode_input["video_grid_thw"] = inputs["video_grid_thw"]

        context = {
            "input_tokens": input_tokens,
            "vision_srt": vision_srt,
            "vision_end": vision_end,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
            "inputs": inputs,
        }
        return encode_input, context

    def preprocess_after_encode(self, encode_output, context, **kwargs):
        """Phase 2: post-encode scatter and write."""
        if context is None:
            return

        ksana_python_input = kwargs["ksana_python_input"]
        input_tokens = context["input_tokens"]
        vision_srt = context["vision_srt"]
        vision_end = context["vision_end"]
        image_grid_thw = context["image_grid_thw"]
        video_grid_thw = context["video_grid_thw"]
        inputs = context["inputs"]

        input_tokens, vision_srt, embeds, additional_tensors = self._scatter_embeds(
            encode_output,
            input_tokens,
            vision_srt,
            vision_end,
            image_grid_thw,
            video_grid_thw,
            inputs,
        )

        self._write_to_ksana_input(
            ksana_python_input, input_tokens, vision_srt, embeds, additional_tensors
        )

    # Method for post-processing
    def postprocess(self, **kwargs):
        return

    def _get_input_positions(
        self,
        input_tokens: List[int],
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
    ):
        """Get mrope input positions and delta value.

        Adapted from
        https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py#L840
        """
        config = self.config
        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id
        spatial_merge_size = config.vision_config.spatial_merge_size

        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = \
                t, h // spatial_merge_size, w // spatial_merge_size
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).repeat_interleave(3) + st_idx)

            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(
                -1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(
                llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(
                llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(
                (torch.tensor(list(zip(t_index, h_index, w_index)))
                             + text_len + st_idx).view(-1))
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(
                llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).repeat_interleave(3) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list)
        mrope_position_delta = llm_positions.max() + 1 - len(input_tokens)

        return llm_positions, mrope_position_delta

    @staticmethod
    def convert_openai_messages(messages: List[Dict], additional_params: Dict) -> List[Dict]:
        for i, message in enumerate(messages):
            for j, part in enumerate(message["content"]):
                if "image_url" in part:
                    messages[i]["content"][j]["image"] = part["image_url"]["url"]
                    for param in ["min_pixels", "max_pixels",
                                  "resized_height", "resized_width"]:
                        if param in additional_params:
                            messages[i]["content"][j][param] = additional_params[param]
                    messages[i]["content"][j].pop("image_url")
                elif "video_url" in part:
                    messages[i]["content"][j]["video"] = part["video_url"]["url"]
                    for param in ["fps", "nframes", "min_frames", "max_frames",
                                  "resized_height", "resized_width"]:
                        if param in additional_params:
                            messages[i]["content"][j][param] = additional_params[param]
                    messages[i]["content"][j].pop("video_url")

        for message in messages:
            if message['role'] == 'user' and 'content' in message:
                multi_modal_elements = [elem for elem in message['content'] if elem['type'] != 'text']
                text_elements = [elem for elem in message['content'] if elem['type'] == 'text']
                message['content'] = multi_modal_elements + text_elements
        return messages
