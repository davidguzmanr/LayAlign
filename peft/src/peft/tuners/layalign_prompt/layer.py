# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TRANSFORMERS_MODEL_CONFIG


class AdaptedAttention(nn.Module):
    """This module wraps a LLamaAttention module and injects adaption prompts."""

    def __init__(self, model_type: str, model):
        """
        Initialize object.

        Args:
            model_type: The transformer model type. This is used to retrieve the right method to
                compute query states.
            adapter_len: The length of the adaption prompt to insert.
            model: The original transformer attention module that is being wrapped.
        """
        assert not isinstance(model, AdaptedAttention)
        super().__init__()
        self.model_type = model_type
        self.model = model
        #self.adapter_states = adapter_states
        #self.prefix_vec = prefix_vec
        # Assume all parameters of the attention model we are wrapping are on the same device.
        device = next(model.parameters()).device
        # Don't think this was specified in the paper, but we follow the official repo which used an Embedding
        # which initializes the tokens with standard normal values.
        # https://github.com/ZrrSkywalker/LLaMA-Adapter/blob/41c3546fe1997ab8a65809dc8d8f9252b19d9faf/llama/model.py#L234
        # (bsz, adapter_len, hidden_size)
        target_dtype = (
            model.q_proj.weight.dtype if model.q_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        )
        self.adapter_states =  torch.empty(1)
        # self.adaption_prompt_k = prefix_states[layer_index, 0, :, :, :, :]
        # self.adaption_prompt_v = prefix_states[layer_index, 1, :, :, :, :]
        # Initialize the gate to 0 as this is "zero-init".
        self.adaption_gate = nn.Parameter(torch.zeros(1))
        # self.adaption_prompt = nn.Parameter(
        #     torch.empty(1, 1, self.model.hidden_size)
        # )

    def update_adapter_states(self, adapter_states):
        self.adapter_states = adapter_states

    def forward(self, **kwargs):
        """
        Forward pass for the adapter which wraps the original LlamaAttention module.

        Notes:
        - Supports both 2-item and 3-item returns from the wrapped attention module
            (HF Transformers versions may return (attn_output, past_key_value) or
            (attn_output, attn_weights, past_key_value), or a dataclass with fields).
        - We do not support returning attention weights here.
        """
        # HF uses 'output_attentions' (plural), but handle either just in case.
        if kwargs.get("output_attentions", False) or kwargs.get("output_attention", False):
            raise NotImplementedError("output_attentions is not currently supported.")

        # Ask the attention module to provide/use cache when supported.
        kwargs.setdefault("use_cache", True)

        # Call the wrapped attention module
        res = self.model(**kwargs)

        # Normalize outputs across HF versions
        if isinstance(res, tuple):
            if len(res) == 3:
                # (attn_output, attn_weights, past_key_value)
                output, _, past_key_value = res
            elif len(res) == 2:
                # (attn_output, past_key_value)  OR  (attn_output, attn_weights)
                output, second = res
                # Heuristic: cache is usually a tuple/list or Cache-like object
                if isinstance(second, (tuple, list)) or getattr(second, "get_seq_length", None) is not None:
                    past_key_value = second
                else:
                    past_key_value = None
            elif len(res) >= 1:
                output = res[0]
                past_key_value = None
            else:
                raise RuntimeError("Wrapped attention returned an empty tuple.")
        else:
            # Dataclass-style outputs
            output = getattr(res, "last_hidden_state", res)
            past_key_value = getattr(res, "past_key_values", None)

        # Sanity check for adapter states
        if not isinstance(self.adapter_states, (tuple, list)) and not torch.is_tensor(self.adapter_states):
            raise RuntimeError("Adapter states not initialized. Call update_adapter_states(...) before forward().")
        if torch.is_tensor(self.adapter_states) and self.adapter_states.numel() == 1:
            raise RuntimeError("Adapter states are empty. Call update_adapter_states(...) before forward().")

        bsz = output.shape[0]
        q_len = output.shape[1]

        # Resolve adapter states for K and V
        adapter_states_k = self.adapter_states[0]
        adapter_states_v = self.adapter_states[1]
        prefix_len = adapter_states_k.shape[1]

        # Ensure dtype/device match attention projections
        proj_dtype = self.model.q_proj.weight.dtype if self.model.q_proj.weight.dtype not in [torch.int8, torch.uint8] else torch.float32
        device = output.device
        adapter_states_k = adapter_states_k.to(device=device, dtype=proj_dtype)
        adapter_states_v = adapter_states_v.to(device=device, dtype=proj_dtype)

        # Figure out projection layer names from config
        k_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].k_proj_layer
        v_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].v_proj_layer
        o_proj_layer = TRANSFORMERS_MODEL_CONFIG[self.model_type].o_proj_layer

        # Some models (e.g., Mistral) have different in/out dims for k/v proj
        factor = self.model.k_proj.in_features // self.model.k_proj.out_features

        # Project adapter states into K/V spaces
        key = getattr(self.model, k_proj_layer)(adapter_states_k)     # (bsz, prefix_len, d_model_k)
        value = getattr(self.model, v_proj_layer)(adapter_states_v)   # (bsz, prefix_len, d_model_v)

        # Reshape to (bsz, num_heads//factor, prefix_len, head_dim) then transpose to (bsz, num_heads//factor, prefix_len, head_dim)
        adapter_k = key.view(bsz, prefix_len, (self.model.num_heads // factor), self.model.head_dim).transpose(1, 2)
        adapter_v = value.view(bsz, prefix_len, (self.model.num_heads // factor), self.model.head_dim).transpose(1, 2)

        # Repeat heads if model uses group-query attention style
        adapter_k = torch.repeat_interleave(adapter_k, repeats=factor, dim=1)  # (bsz, num_heads, prefix_len, head_dim)
        adapter_v = torch.repeat_interleave(adapter_v, repeats=factor, dim=1)  # (bsz, num_heads, prefix_len, head_dim)

        # Recompute query states with the same kwargs used by attention
        compute_query_states = TRANSFORMERS_MODEL_CONFIG[self.model_type].compute_query_states
        query_states = compute_query_states(model=self.model, **kwargs)  # (bsz, num_heads, q_len, head_dim)

        previous_dtype = query_states.dtype

        # Compute adapter attention scores: (bsz, num_heads, q_len, prefix_len)
        scores = torch.matmul(query_states, adapter_k.transpose(2, 3).to(previous_dtype)) / math.sqrt(self.model.head_dim)
        # Upcast for softmax stability, then gate
        scores = self.adaption_gate * torch.softmax(scores, dim=-1, dtype=torch.float32).to(previous_dtype)

        # Weighted sum over adapter values -> (bsz, q_len, num_heads*head_dim)
        adapter_output = torch.matmul(scores, adapter_v).transpose(1, 2).reshape(bsz, q_len, -1)

        # Optional output projection
        if o_proj_layer is not None:
            adapter_output = getattr(self.model, o_proj_layer)(adapter_output)

        # Add adaption prompt output to original attention output
        output = (output + adapter_output).to(previous_dtype)

        # Keep original API: (output, attn_weights=None, past_key_value)
        return output, None, past_key_value

