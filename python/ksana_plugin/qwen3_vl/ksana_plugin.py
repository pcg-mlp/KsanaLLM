# pylint: disable=broad-except
# Copyright 2026 Tencent Inc.  All rights reserved.
#
# ==============================================================================

import os
import sys
from typing import Dict, List, Optional, Union

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, AutoConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir: ./KsanaLLM/src/ksana_llm/python/ksana_plugin
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)
# parent2_dir: ./KsanaLLM/csrc/python
parent2_dir = os.path.abspath(os.path.join(parent_dir, os.pardir))
sys.path.append(parent2_dir)

from plugin_utils import free_cache
from plugin_utils import load_safetensors, get_weight_map
from plugin_utils import check_input
from ksana_encoder.encoder_client import EncoderClient


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
            self.precision = (
                getattr(self.config.text_config, "dtype", None) or torch.bfloat16
            )

            self.init_encoder()

            print("[I] Initializing the Qwen3-VL Torch model successfully!")

            free_cache()

            # Ensure the result is a dictionary
            return {"plugin_trt": False}

        if "postprocess" in kwargs:
            return

    def _init_local_encoder(self):
        try:
            visual = Qwen3VLVisionModel._from_config(
                self.config.vision_config,
                attn_implementation="flash_attention_2",
                dtype=self.precision,
            )
        except Exception as e:
            print(f"[W] flash_attention_2 unavailable ({e}), falling back to sdpa.")
            visual = Qwen3VLVisionModel._from_config(
                self.config.vision_config,
                attn_implementation="sdpa",
                dtype=self.precision,
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

    def _init_remote_encoder(self):
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
            deepstack_features = getattr(output, "deepstack_features", None)
        else:
            embeds = output
            deepstack_features = None
        result = {"embeds": embeds.cpu()}
        if deepstack_features is not None:
            result["deepstack_features"] = [f.cpu() for f in deepstack_features]
        return result

    def _remote_encode(self, inputs: Dict) -> Dict:
        return self.mm_encoder(inputs)

    def encode(self, inputs: Dict) -> Dict:
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

    # Method for pre-processing
    def preprocess(self, **kwargs):
        if not check_input(["ksana_python_input", "messages"], **kwargs):
            raise RuntimeError("Plugin preprocess wrong input.")

        messages: Optional[List[Dict]] = kwargs["messages"]
        if messages is None:
            return
        additional_params = kwargs.get("additional_params", {})
        messages = KsanaPlugin.convert_openai_messages(messages, additional_params)

        ksana_python_input = kwargs["ksana_python_input"]

        config = self.config

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
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

        # Used to reassemble image embeddings and video embeddings in the order of visual tokens
        permute = {}
        # DeepStack: collect per-span deepstack features {span_idx: {layer_idx: tensor}}
        deepstack_permute = {}

        def infer_embeds(permute, deepstack_permute, tag="image"):
            pixel_values_tag = (
                "pixel_values_videos" if tag == "video" else "pixel_values"
            )
            grid_thw_tag = "video_grid_thw" if tag == "video" else "image_grid_thw"
            token_id = (
                config.video_token_id if tag == "video" else config.image_token_id
            )

            grid_thw = []
            if pixel_values_tag in inputs:
                pixel_values = inputs[pixel_values_tag].to(dtype=self.precision)
                grid_thw = inputs[grid_thw_tag]

                # Vision encoder
                encode_input = {
                    "pixel_values": pixel_values,
                    "grid_thw": grid_thw,
                }
                encode_output = self.encode(encode_input)
                embeds = encode_output["embeds"]
                deepstack_features = encode_output.get("deepstack_features", None)

                sections, idx = [], []
                for i, srt in enumerate(vision_srt):
                    if input_tokens[srt] == token_id:
                        sections.append(vision_end[i] - srt + 1)
                        idx.append(i)
                embeds = torch.split(embeds, sections, dim=0)
                for i in range(len(embeds)):
                    permute[idx[i]] = embeds[i]

                # Split deepstack features by the same sections
                if deepstack_features is not None:
                    for layer_idx, feat in enumerate(deepstack_features):
                        split_feats = torch.split(feat, sections, dim=0)
                        for i in range(len(split_feats)):
                            if idx[i] not in deepstack_permute:
                                deepstack_permute[idx[i]] = {}
                            deepstack_permute[idx[i]][layer_idx] = split_feats[i]
            return grid_thw

        image_grid_thw = infer_embeds(permute, deepstack_permute, tag="image")
        video_grid_thw = infer_embeds(permute, deepstack_permute, tag="video")

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
            # DeepStack injects into the first N text decoder layers (0, 1, ..., N-1),
            # where N = len(vision_config.deepstack_visual_indexes).
            # Note: deepstack_visual_indexes specifies Vision Encoder extraction layers,
            # NOT text decoder injection layers. The text decoder always injects at layers
            # 0..N-1, matching the HF Qwen3VLTextModel behavior.
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

        # Write results to ksana_python_input
        ksana_python_input.input_tokens = input_tokens
        ksana_python_input.input_refit_embedding.pos = vision_srt
        ksana_python_input.input_refit_embedding.embedding_tensors = embeds
        ksana_python_input.input_refit_embedding.additional_tensors = additional_tensors

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
