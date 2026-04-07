# pylint: disable=broad-except
# Copyright 2026 Tencent Inc.  All rights reserved.
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
# add current_dir to import modeling.py alongside this file
if current_dir not in sys.path:
    sys.path.append(current_dir)

from modeling import KsanaQwen3VLVisionModel
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
                # Qwen3-VL weights: "model.visual.X.Y" -> "X.Y"
                key = name[name.index("visual.") + len("visual.") :]
                visual_weights[key] = tensor
    return visual_weights


class KsanaPlugin:
    """
    Qwen3-VL plugin for KsanaLLM.
    Handles vision preprocessing and encoder inference for Qwen3-VL models.
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
            self.video_processor = self.processor.video_processor
            # image_resize_factor: patch_size * merge_size = 32, aligned with HF image_processor
            self.image_resize_factor = (
                self.image_processor.patch_size * self.image_processor.merge_size
            )
            self.precision = (
                getattr(self.config.text_config, "dtype", None) or torch.bfloat16
            )

            # Pre-compute fused rescale+normalize coefficients: pixel * scale + bias ≡ (pixel/255 - mean) / std
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

            print("[I] Initializing the Qwen3-VL Torch model successfully!")

            free_cache()

            # Ensure the result is a dictionary
            return {"plugin_trt": False}

        if "postprocess" in kwargs:
            return

    @staticmethod
    def _fetch_image_cv(
        ele: dict, size_factor: int, return_numpy: bool = False
    ) -> "Image.Image | np.ndarray":
        """OpenCV-accelerated image loading (~2x speedup over PIL) for local files.

        Falls back to fetch_image for HTTP URLs, base64, or PIL objects.
        If return_numpy=True, returns (H, W, 3) uint8 array to avoid PIL round-trip.
        """

        def _fallback():
            pil_img = fetch_image(ele, size_factor=size_factor)
            return np.array(pil_img) if return_numpy else pil_img

        image = ele.get("image") or ele.get("image_url")

        # Only accelerate local file paths; fall back for other input types.
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

        # Compute target size (same logic as qwen_vl_utils.fetch_image)
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

        # INTER_CUBIC ≈ PIL BICUBIC
        if (h_orig, w_orig) != (rh, rw):
            img_rgb = cv2.resize(img_rgb, (rw, rh), interpolation=cv2.INTER_CUBIC)

        if return_numpy:
            return img_rgb

        return Image.fromarray(img_rgb)

    def _process_images_torch(self, images_np: list) -> dict:
        """Torch-native image processor with bf16-early conversion (~4x speedup over HF numpy path).

        Args:
            images_np: List of (H, W, 3) uint8 arrays pre-resized by _fetch_image_cv.

        Returns:
            dict with 'pixel_values' (total_patches, patch_dim) bf16 and 'image_grid_thw' int64.
        """
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

            # uint8 numpy → torch, fused rescale+normalize
            t = torch.from_numpy(img_np).to(dtype=torch.float32)  # (H, W, 3)
            t.mul_(self._fused_scale).add_(self._fused_bias)

            # Convert to bf16 early to halve data size for subsequent ops
            t = t.to(dtype=torch.bfloat16)

            t = t.permute(2, 0, 1).contiguous()  # HWC → CHW

            # Temporal padding: duplicate single image for temporal_patch_size
            t = t.unsqueeze(0).expand(temporal_patch_size, -1, -1, -1).contiguous()

            # Reshape into patch structure (same as HF _preprocess)
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
        from modeling import _HAS_VIT_FA3

        visual = None
        for attn_impl in ["flash_attention_3", "flash_attention_2", "sdpa"]:
            try:
                visual = KsanaQwen3VLVisionModel._from_config(
                    self.config.vision_config,
                    attn_implementation=attn_impl,
                    dtype=self.precision,
                )
                break
            except Exception as e:
                print(f"[W] {attn_impl} unavailable ({e}), trying next...")
        if visual is None:
            raise RuntimeError(
                "No available attention implementation for vision encoder."
            )
        visual = visual.to(dtype=self.precision)
        visual.load_state_dict(load_state_dict_from_safetensors(self.model_path))
        # Recompute rotary embedding inv_freq in FP32 to match HF from_pretrained behavior.
        # inv_freq is a non-persistent buffer (not in state_dict), so from_pretrained()
        # keeps its original FP32 value from __init__. But _from_config() + .to(dtype)
        # converts it to BF16, losing precision and causing positional encoding errors.
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

        # Print final attention configuration
        if _HAS_VIT_FA3:
            runtime_attn = "sgl_kernel FA3"
        else:
            runtime_attn = f"transformers {attn_impl} dispatch"
        print(
            f"[I] Vision encoder: model=KsanaQwen3VLVisionModel, "
            f"attn_implementation={attn_impl}, runtime_attention={runtime_attn}"
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
    def _local_encode(self, inputs: Dict, on_intermediate_ready=None) -> Dict:
        pixel_values = inputs["pixel_values"].to(self.device)
        grid_thw = inputs["grid_thw"].to(self.device)

        if on_intermediate_ready is not None:

            def model_callback(ds_idx, cpu_feature):
                on_intermediate_ready(
                    {
                        "deepstack_feature": cpu_feature,
                        "deepstack_index": ds_idx,
                    }
                )

            output = self.mm_encoder(
                pixel_values, grid_thw, on_deepstack_ready=model_callback
            )
        else:
            output = self.mm_encoder(pixel_values, grid_thw)

        if not isinstance(output, torch.Tensor):
            embeds = output.pooler_output
            deepstack_features = getattr(output, "deepstack_features", None)
        else:
            embeds = output
            deepstack_features = None
        result = {"embeds": embeds.cpu()}
        if deepstack_features is not None and on_intermediate_ready is None:
            result["deepstack_features"] = [
                f if f.device.type == "cpu" else f.cpu() for f in deepstack_features
            ]
        return result

    def _remote_encode(self, inputs: Dict) -> Dict:
        return self.mm_encoder(inputs)

    def _encode_single(self, inputs: Dict, on_intermediate_ready=None) -> Dict:
        """Encode a single modality (image or video) with optional caching."""
        cache_key = inputs.get("mm_item_key", None)
        use_cache = cache_key is not None and self.multimodal_cache is not None

        if use_cache:
            cached_result = self.multimodal_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

        if not self.using_remote_encoder:
            result = self._local_encode(
                inputs, on_intermediate_ready=on_intermediate_ready
            )
        else:
            result = self._remote_encode(inputs)

        if use_cache:
            self.multimodal_cache.put(cache_key, result)
        return result

    def encode(self, inputs: Dict, on_intermediate_ready=None) -> Dict:
        """Encode image and/or video modalities.

        Coupled path: inputs has pixel_values + grid_thw (single modality).
        Decoupled path: inputs has image_pixel_values/image_grid_thw and/or
        video_pixel_values/video_grid_thw.

        Args:
            inputs: 编码输入数据
            on_intermediate_ready: 流式回调函数，用于立即发送deepstack中间结果。
                                   None表示非流式模式。
        """
        # Coupled path: single modality with pixel_values + grid_thw
        if "pixel_values" in inputs and "grid_thw" in inputs:
            return self._encode_single(
                inputs, on_intermediate_ready=on_intermediate_ready
            )

        # Decoupled path: combined image + video
        result = {}
        for modality in ("image", "video"):
            pv_key = f"{modality}_pixel_values"
            if pv_key not in inputs:
                continue
            encode_input = {
                "pixel_values": inputs[pv_key],
                "grid_thw": inputs[f"{modality}_grid_thw"],
            }
            output = self._encode_single(
                encode_input, on_intermediate_ready=on_intermediate_ready
            )
            result[f"{modality}_embeds"] = output["embeds"]
            if "deepstack_features" in output:
                result[f"{modality}_deepstack_features"] = output["deepstack_features"]
            for key in TIMING_KEYS:
                if key in output:
                    result[key] = result.get(key, 0) + output[key]
        return result

    def _expand_image_tokens(self, text, image_grid_thw):
        """Expand <|image_pad|> placeholders based on grid dimensions and merge size."""
        merge_length = self.image_processor.merge_size**2
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
        """Tokenize messages and extract pixel values and grid info.

        Returns:
            (inputs, input_tokens, vision_srt, vision_end, image_grid_thw, video_grid_thw)
        """
        config = self.config
        messages = KsanaPlugin.convert_openai_messages(messages, additional_params)

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Use OpenCV-accelerated loading with HF-aligned resize factor (patch_size * merge_size)
        vision_infos = extract_vision_info(messages)
        image_vision_infos = [
            vi for vi in vision_infos if "image" in vi or "image_url" in vi
        ]
        has_video = any("video" in vi for vi in vision_infos)

        # Video path: use qwen_vl_utils 0.0.14+ with return_video_metadata=True
        # to get correct frame sampling and metadata, then pass to HF processor
        # with do_sample_frames=False to avoid double-sampling.
        if has_video:
            image_inputs_pv, video_inputs_raw, video_kwargs = process_vision_info(
                messages,
                image_patch_size=self.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            # Separate (tensor, metadata) tuples from video_inputs
            video_tensors = []
            video_metadata = []
            for v in video_inputs_raw:
                if isinstance(v, tuple):
                    video_tensors.append(v[0])
                    video_metadata.append(v[1])
                else:
                    video_tensors.append(v)

            # Image inputs: use OpenCV-accelerated loading if available
            image_inputs = None
            if image_vision_infos:
                image_inputs = [
                    self._fetch_image_cv(vi, size_factor=self.image_resize_factor)
                    for vi in image_vision_infos
                ] or None

            inputs = self.processor(
                text=[text],
                images=image_inputs if image_inputs else image_inputs_pv,
                videos=video_tensors if video_tensors else None,
                video_metadata=video_metadata if video_metadata else None,
                **video_kwargs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # Optimized path: torch-native image processor with bf16-early conversion
            # Step 1: Image processing
            image_data = {}
            if image_vision_infos:
                image_np_inputs = [
                    self._fetch_image_cv(
                        vi, size_factor=self.image_resize_factor, return_numpy=True
                    )
                    for vi in image_vision_infos
                ]
                image_data = self._process_images_torch(image_np_inputs)

            # Step 2: Expand placeholder tokens
            text_list = [text]
            if image_data:
                text_list = self._expand_image_tokens(
                    text_list, image_data["image_grid_thw"]
                )

            # Step 3: Tokenize
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
        """Scatter encoder outputs back into vision span order and build final tensors.

        Handles both coupled (embeds) and decoupled (image_embeds/video_embeds) paths.

        Returns:
            (input_tokens, vision_srt, embeds_list, additional_tensors)
        """
        config = self.config

        # Reassemble embeddings in visual token order
        permute = {}
        # DeepStack: {span_idx: {layer_idx: tensor}}
        deepstack_permute = {}

        def _scatter_modality(embeds, deepstack_features, token_id):
            """Split embeds by vision span sections and populate permute/deepstack_permute."""
            sections, idx = [], []
            for i, srt in enumerate(vision_srt):
                if input_tokens[srt] == token_id:
                    sections.append(vision_end[i] - srt + 1)
                    idx.append(i)
            split_embeds = torch.split(embeds, sections, dim=0)
            for i in range(len(split_embeds)):
                permute[idx[i]] = split_embeds[i]

            if deepstack_features is not None:
                for layer_idx, feat in enumerate(deepstack_features):
                    split_feats = torch.split(feat, sections, dim=0)
                    for i in range(len(split_feats)):
                        if idx[i] not in deepstack_permute:
                            deepstack_permute[idx[i]] = {}
                        deepstack_permute[idx[i]][layer_idx] = split_feats[i]

        # Decoupled path keys
        if "image_embeds" in encode_output:
            _scatter_modality(
                encode_output["image_embeds"],
                encode_output.get("image_deepstack_features", None),
                config.image_token_id,
            )
        if "video_embeds" in encode_output:
            _scatter_modality(
                encode_output["video_embeds"],
                encode_output.get("video_deepstack_features", None),
                config.video_token_id,
            )

        # Coupled path: single modality (backward compat)
        if "embeds" in encode_output:
            for token_id in [config.image_token_id, config.video_token_id]:
                sections = []
                idx = []
                for i, srt in enumerate(vision_srt):
                    if input_tokens[srt] == token_id:
                        sections.append(vision_end[i] - srt + 1)
                        idx.append(i)
                if sections and sum(sections) == encode_output["embeds"].shape[0]:
                    _scatter_modality(
                        encode_output["embeds"],
                        encode_output.get("deepstack_features", None),
                        token_id,
                    )
                    break

        embeds = []
        for i in range(len(permute)):
            embeds.append(permute[i])

        # Assemble deepstack features in visual span order, concatenated per layer
        deepstack_features_ordered = []
        if deepstack_permute:
            num_deepstack_layers = max(len(v) for v in deepstack_permute.values())
            for layer_idx in range(num_deepstack_layers):
                layer_feats = []
                for span_idx in range(len(permute)):
                    if (
                        span_idx in deepstack_permute
                        and layer_idx in deepstack_permute[span_idx]
                    ):
                        layer_feats.append(deepstack_permute[span_idx][layer_idx])
                if layer_feats:
                    deepstack_features_ordered.append(torch.cat(layer_feats, dim=0))

        position_ids, mrope_position_deltas = self._get_input_positions(
            input_tokens, image_grid_thw, video_grid_thw
        )
        position_ids = position_ids.squeeze(0).contiguous()
        mrope_position_deltas = torch.LongTensor([mrope_position_deltas.item()])

        # Build additional_tensors: [mrope_pos, mrope_delta, ...deepstack_tensors...]
        additional_tensors = [position_ids, mrope_position_deltas]

        if deepstack_features_ordered:
            visual_segment_starts = torch.tensor(vision_srt, dtype=torch.int32)
            visual_segment_lengths = torch.tensor(
                [vision_end[i] - vision_srt[i] + 1 for i in range(len(vision_srt))],
                dtype=torch.int32,
            )
            num_deepstack_layers = len(deepstack_features_ordered)
            # deepstack_visual_indexes = ViT extraction layers; text decoder injects at 0..N-1
            deepstack_layer_ids = torch.arange(num_deepstack_layers, dtype=torch.int32)
            additional_tensors.extend(
                [
                    visual_segment_starts,
                    visual_segment_lengths,
                    deepstack_layer_ids,
                ]
            )
            for feat in deepstack_features_ordered:
                additional_tensors.append(feat.to(dtype=self.precision))

        return input_tokens, vision_srt, embeds, additional_tensors

    @staticmethod
    def _write_to_ksana_input(
        ksana_python_input, input_tokens, vision_srt, embeds, additional_tensors
    ):
        """Write preprocessing results to ksana_python_input."""
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
        """Phase 1 of decoupled preprocessing: CPU work before encoder I/O.

        Returns:
            (encode_input, context): encode_input for encoder; context for phase 2.
        """
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
        """Phase 2 of decoupled preprocessing: post-encode scatter and write."""
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

        Computes 3D (T/H/W) position IDs in interleaved layout [t0,h0,w0, t1,h1,w1, ...].
        The interleave-to-rotary-dim mapping is handled by the C++ MROPE_INTERLEAVED kernel.

        Adapted from
        https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py#L840
        """
        config = self.config
        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id

        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        # Qwen3-VL wraps each video frame in its own <|vision_start|>...<|vision_end|>
        # pair (separated by timestamp text tokens), so each frame produces a separate
        # video_token_id occurrence.  Expand video_grid_thw from per-video [T, H, W]
        # to per-frame [1, H, W] * T so the loop below can consume one entry per frame.
        expanded_video_grid_thw = []
        for thw in video_grid_thw:
            t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
            for _ in range(t):
                expanded_video_grid_thw.append([1, h, w])
        video_grid_thw = expanded_video_grid_thw

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)
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
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).repeat_interleave(3) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t)
                .view(-1, 1)
                .expand(-1, llm_grid_h * llm_grid_w)
                .flatten()
            )
            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                (
                    torch.stack([t_index, h_index, w_index], dim=1) + text_len + st_idx
                ).view(-1)
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).repeat_interleave(3) + st_idx
            )

        # We set the layout of mrotary_embedding_pos be [t1, h1, w1, t2, h2, w2, ...],
        # instead of the conventional [t1, t2, ..., h1, h2, ..., w1, w2, ...].
        # This is to facilitate the implementation of our MRotaryEmbedding kernel.
        llm_positions = torch.cat(llm_pos_ids_list)
        mrope_position_delta = llm_positions.max() + 1 - len(input_tokens)

        return llm_positions, mrope_position_delta

    @staticmethod
    def convert_openai_messages(
        messages: List[Dict], additional_params: Dict
    ) -> List[Dict]:
        # Convert `messages` in OpenAI Chat Completion format to Qwen VL format,
        # so that it can be processed by qwen_vl_utils.
        for i, message in enumerate(messages):
            for j, part in enumerate(message["content"]):
                if "image_url" in part:
                    messages[i]["content"][j]["image"] = part["image_url"]["url"]
                    for param in [
                        "min_pixels",
                        "max_pixels",
                        "resized_height",
                        "resized_width",
                    ]:
                        if param in additional_params:
                            messages[i]["content"][j][param] = additional_params[param]
                    messages[i]["content"][j].pop("image_url")
                elif "video_url" in part:
                    messages[i]["content"][j]["video"] = part["video_url"]["url"]
                    for param in [
                        "fps",
                        "nframes",
                        "min_frames",
                        "max_frames",
                        "resized_height",
                        "resized_width",
                    ]:
                        if param in additional_params:
                            messages[i]["content"][j][param] = additional_params[param]
                    messages[i]["content"][j].pop("video_url")

        for message in messages:
            if message["role"] == "user" and "content" in message:
                multi_modal_elements = [
                    elem for elem in message["content"] if elem["type"] != "text"
                ]
                text_elements = [
                    elem for elem in message["content"] if elem["type"] == "text"
                ]
                message["content"] = multi_modal_elements + text_elements
        return messages
