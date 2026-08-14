import sys
import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(parent_dir)
# sys.path.append(os.path.join(parent_dir, 'utils'))

import types

import torch
import torch.nn as nn

from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
)

# from vortex.teal.utils.utils_vq import ActivationModule, Distribution, SparsifyFn, get_module_device

from transformers.modeling_flash_attention_utils import _flash_attention_forward

def _monkeypatch_self_attn(self_attn, file_path, grabbing_mode=False, \
        dist_args=None, sparse_fn_args=None):
    from vortex.teal.utils.utils_activations import \
        ActivationModule, Distribution, SparsifyFn
    from vortex.teal.utils.utils_model import get_module_device
    assert hasattr(self_attn, "vq_ctx")
    vq_ctx = self_attn.vq_ctx

    self_attn.forward_old = self_attn.forward

    self_attn.forward = types.MethodType(_FA2_forward, self_attn)

    self_attn.file_path = file_path
    self_attn.grabbing_mode = grabbing_mode

    if not grabbing_mode:
        self_attn.distrs = {}
        self_attn.distrs['h1'] = Distribution(file_path, hidden_type='h1', **(dist_args or {}))
        self_attn.distrs['h2'] = Distribution(file_path, hidden_type='h2', **(dist_args or {}))

        if vq_ctx["uniform_sparse_fn"]:
            self_attn.sparse_fns = nn.ModuleDict({
                'q': SparsifyFn(self_attn.distrs['h1'], **(sparse_fn_args or {})).to(get_module_device(self_attn)),
                'k': SparsifyFn(self_attn.distrs['h1'], **(sparse_fn_args or {})).to(get_module_device(self_attn)),
                'v': SparsifyFn(self_attn.distrs['h1'], **(sparse_fn_args or {})).to(get_module_device(self_attn)),
                'o': SparsifyFn(self_attn.distrs['h2'], **(sparse_fn_args or {})).to(get_module_device(self_attn))
            })
        else:
            tmp_dict = {}
            for proj_name in ['q', 'k', 'v', 'o']:
                # num_codebooks = vq_ctx['config_num_codebooks'][proj_name]
                num_codebooks = vq_ctx['config_num_codebooks_glb']
                for l in range(num_codebooks):
                    if proj_name == 'o':
                        tmp_dict[proj_name + f"{l}"] = SparsifyFn(self_attn.distrs['h2'], **(sparse_fn_args or {})).to(get_module_device(self_attn))
                    else:
                        tmp_dict[proj_name + f"{l}"] = SparsifyFn(self_attn.distrs['h1'], **(sparse_fn_args or {})).to(get_module_device(self_attn))
            self_attn.sparse_fns = nn.ModuleDict(tmp_dict)

    self_attn.activation_module = ActivationModule(file_path, **(dist_args or {}))

    return self_attn


def _FA2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None, #: Optional[torch.LongTensor] = None,
    position_ids = None, #: Optional[torch.LongTensor] = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] = None,
    past_key_value = None, #: Optional[Cache] = None,
    output_attentions = False, #: bool = False,
    use_cache = False, #: bool = False,
    cache_position = None, #: Optional[torch.LongTensor] = None,
    activation_module = None,
    **kwargs,
): # -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # if isinstance(past_key_value, StaticCache):
    #     raise ValueError(
    #         "`static` cache implementation is not compatible with `attn_implementation==flash_attention_2` "
    #         "make sure to use `sdpa` in the mean time, and open an issue at https://github.com/huggingface/transformers"
    #     )
    
    assert hasattr(self, "vq_ctx")
    vq_ctx = self.vq_ctx

    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    # MONKEYPATCH HERE
    
    if self.grabbing_mode:
        self.activation_module.grab_activations(hidden_states, 'h1')
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    else: 
        if vq_ctx["uniform_sparse_fn"]:
            x_q = self.sparse_fns['q'](hidden_states)
            x_k = self.sparse_fns['k'](hidden_states)
            x_v = self.sparse_fns['v'](hidden_states)
        else:
            num_codebooks = vq_ctx['config_num_codebooks_glb']
            x_q = []
            x_k = []
            x_v = []
            for l in range(num_codebooks):
                x_q.append(self.sparse_fns[f'q{l}'](hidden_states))
                x_k.append(self.sparse_fns[f'k{l}'](hidden_states))
                x_v.append(self.sparse_fns[f'v{l}'](hidden_states))

        query_states = self.q_proj(x_q)
        key_states = self.k_proj(x_k)
        value_states = self.v_proj(x_v)
        
    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    hidden_size = self.config.hidden_size
    num_heads = self.config.num_attention_heads
    head_dim = self.config.hidden_size // self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads

    query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

    # make it compatible with multiple versions.
    if position_embeddings is None:
        assert position_ids is not None
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings
        
    # check device of cos and sin, if they are not on the same device as query_states, key_states, value_states, move them to the correct device
    if cos.device != query_states.device:
        cos = cos.to(query_states.device)
        sin = sin.to(query_states.device)

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)


    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        # logger.warning_once(
        #     f"The input hidden states seems to be silently casted in float32, this might be related to"
        #     f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
        #     f" {target_dtype}."
        # )
        print(f"Casting input hidden states to {target_dtype} (this should not be happening)")

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    # NOTE: sliding window isn't tested for Mistral, please create an issue if something goes wrong
    # However, we don't ever utilize sequence lengths of more than 4096 for the methodology + evals
    attn_output = _flash_attention_forward(
        query_states, key_states, value_states, attention_mask, q_len, position_ids=position_ids, dropout=dropout_rate, sliding_window=getattr(self, "sliding_window", None), is_causal=True
    )

    attn_output = attn_output.reshape(bsz, q_len, hidden_size).contiguous()

    # MONKEYPATCH HERE
    if self.grabbing_mode:
        self.activation_module.grab_activations(attn_output, 'h2')
        attn_output = self.o_proj(attn_output)
    else:
        if vq_ctx["uniform_sparse_fn"]:
            attn_output = self.sparse_fns['o'](attn_output)
            attn_output = self.o_proj(attn_output)
        else:
            num_codebooks = vq_ctx['config_num_codebooks_glb']
            x_output = []
            for l in range(num_codebooks):
                x_output.append(self.sparse_fns[f'o{l}'](attn_output))
            attn_output = self.o_proj(x_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value