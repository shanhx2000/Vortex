import sys
import os
import logging
from vortex.core.model.custom_rvq import CustomRvqLlamaConfig, CustomRvqLlamaForCausalLM, CustomRvqMistralConfig, CustomRvqMistralForCausalLM

import torch
import torch.nn.functional as F
from torch import nn

from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from vortex.teal.teal_rvq.mlp_vq import _monkeypatch_mlp
from vortex.teal.teal_rvq.self_attn_vq import _monkeypatch_self_attn
from vortex.teal.utils.utils_model import get_layer_greedy_sparsities

import types

def _monkeypatch_layer(layer, path, grabbing_mode=False, dist_args=None, sparse_fn_args=None):
    layer.path = path
    layer.grabbing_mode = grabbing_mode
    layer.mlp = _monkeypatch_mlp(layer.mlp, f"{path}/mlp", grabbing_mode=grabbing_mode, dist_args=dist_args, sparse_fn_args=sparse_fn_args)
    layer.self_attn = _monkeypatch_self_attn(layer.self_attn, f"{path}/self_attn", grabbing_mode=grabbing_mode, dist_args=dist_args, sparse_fn_args=sparse_fn_args)
    return layer

class SparseModelMixin:
    """
    RVQ/AQLM counterpart of `vortex.teal.teal.model.SparseModelMixin`: instead
    of one SparsifyFn per q/k/v/o/gate/up/down projection, each projection
    gets one SparsifyFn *per AQLM codebook* (`vq_ctx["config_num_codebooks_glb"]`),
    so input sparsity is searched independently per codebook (README's
    "Codebook-wise Input Sparsity"). Sequential/codebook-serial *output*
    sparsity (the research repo's `SequentialRvqCustomLinear` /
    `bitserial` path) is out of scope here and was dropped during migration.
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):

        print("kwargs at SparseModelMixin", kwargs)

        # Extract custom arguments
        histogram_path = kwargs.pop('histogram_path', None)
        grab_acts = kwargs.pop('grab_acts', False)

        greedy_sparsity_path = kwargs.pop('greedy_sparsity_path', None)
        greedy_sparsity_level = kwargs.pop('greedy_sparsity_level', None)

        uniform_sparsity = kwargs.pop('uniform_sparsity', None)
        mlp_sparsity = kwargs.pop('mlp_sparsity', None)
        self_attn_sparsity = kwargs.pop('self_attn_sparsity', None)
        apply_prefill = kwargs.pop('apply_prefill', True)

        dist_args = kwargs.pop('dist_args', None)
        sparse_fn_args = kwargs.pop('sparse_fn_args', None)

        # Load the config
        config = kwargs.get('config', None)
        if config is None:
            config = cls.config_class.from_pretrained(pretrained_model_name_or_path, **kwargs)
        else:
            kwargs.pop('config', None)

        vq_ctx = kwargs.pop('vq_ctx', None)

        # Create the model
        model = super().from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            **kwargs)

        # Add vq_ctx
        assert vq_ctx is not None
        def attach_vq_ctx(model, vq_ctx):
            model.vq_ctx = vq_ctx
            assert hasattr(model.model, "layers")
            for i, layer in enumerate(model.model.layers):
                assert hasattr(layer, "self_attn")
                layer.self_attn.vq_ctx = vq_ctx
                assert hasattr(layer, "mlp")
                layer.mlp.vq_ctx = vq_ctx
            return model
        model = attach_vq_ctx(model, vq_ctx)

        # Apply sparse layers if histogram_path is provided
        assert histogram_path is not None, "histogram_path must be provided"
        os.makedirs(histogram_path, exist_ok=True)

        model.set_grabbing_mode(grab_acts)
        print(f"Building sparse layers... grab_acts={grab_acts}")
        model.build_sparse_layers(histogram_path,
                                  grab_acts,
                                  dist_args=dist_args,
                                  sparse_fn_args=sparse_fn_args)

        print(model)

        if greedy_sparsity_path is not None:
            assert greedy_sparsity_level is not None, "greedy_sparsity_level must be provided"
            model.load_greedy_sparsities(greedy_sparsity_path, greedy_sparsity_level)
        elif uniform_sparsity is not None:
            model.set_uniform_sparsity(uniform_sparsity)
        elif mlp_sparsity is not None or self_attn_sparsity is not None:
            if mlp_sparsity is not None:
                model.set_mlp_sparsity(mlp_sparsity)
            if self_attn_sparsity is not None:
                model.set_self_attn_sparsity(self_attn_sparsity)
        elif not grab_acts:
            model.reset_sparsities()

        if not grab_acts:
            model.set_apply_prefill(apply_prefill)

        return model

    def set_grabbing_mode(self, mode):
        for layer in self.model.layers:
            layer.mlp.grabbing_mode = mode
            layer.self_attn.grabbing_mode = mode

    def set_apply_prefill(self, apply_prefill):
        assert hasattr(self, "vq_ctx")
        vq_ctx = self.vq_ctx
        num_codebooks = vq_ctx['config_num_codebooks_glb']

        if vq_ctx["uniform_sparse_fn"]:
            for layer in self.model.layers:
                for proj in ['q', 'k', 'v', 'o']:
                    layer.self_attn.sparse_fns[proj].apply_prefill = apply_prefill
                for proj in ['gate', 'up', 'down']:
                    layer.mlp.sparse_fns[proj].apply_prefill = apply_prefill
        else:
            for layer in self.model.layers:
                for proj in ['q', 'k', 'v', 'o']:
                    for l in range(num_codebooks):
                        layer.self_attn.sparse_fns[proj+f'{l}'].apply_prefill = apply_prefill
                for proj in ['gate', 'up', 'down']:
                    for l in range(num_codebooks):
                        layer.mlp.sparse_fns[proj+f'{l}'].apply_prefill = apply_prefill

    def build_sparse_layers(self, histogram_path, grab_acts, dist_args=None, sparse_fn_args=None):
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        layers = []
        os.makedirs(histogram_path, exist_ok=True)

        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, LlamaDecoderLayer) or isinstance(layer, MistralDecoderLayer):
                layers.append(
                    _monkeypatch_layer(
                        layer,
                        path=f"{histogram_path}/layer-{i}",
                        grabbing_mode=grab_acts,
                        dist_args=dist_args,
                        sparse_fn_args=sparse_fn_args
                    )
                )
            else:
                raise ValueError(f"Unknown layer type: {type(layer)}")

        self.model.layers = nn.ModuleList(layers)


    def load_greedy_sparsities(self, greedy_sparsity_path, greedy_sparsity_level, vq_ctx):
        layer_sparsity_levels = [greedy_sparsity_level] * len(self.model.layers)
        sparsities = get_layer_greedy_sparsities(layer_sparsity_levels, greedy_sparsity_path, vq_ctx)
        self.set_sparsities(sparsities)

    def reset_sparsities(self):
        self.set_uniform_sparsity(0)

    def set_mlp_sparsity(self, sparsity):
        assert hasattr(self, "vq_ctx")
        vq_ctx = self.vq_ctx
        num_codebooks = vq_ctx['config_num_codebooks_glb']
        if vq_ctx["uniform_sparse_fn"]:
            for layer in self.model.layers:
                layer.mlp.sparse_fns['gate'].set_threshold(sparsity)
                layer.mlp.sparse_fns['up'].set_threshold(sparsity)
                layer.mlp.sparse_fns['down'].set_threshold(sparsity)
        else:
            for layer in self.model.layers:
                for l in range(num_codebooks):
                    layer.mlp.sparse_fns[f'gate{l}'].set_threshold(sparsity)
                    layer.mlp.sparse_fns[f'up{l}'].set_threshold(sparsity)
                    layer.mlp.sparse_fns[f'down{l}'].set_threshold(sparsity)

    def set_self_attn_sparsity(self, sparsity):
        assert hasattr(self, "vq_ctx")
        vq_ctx = self.vq_ctx
        num_codebooks = vq_ctx['config_num_codebooks_glb']
        if vq_ctx["uniform_sparse_fn"]:
            for layer in self.model.layers:
                layer.self_attn.sparse_fns['q'].set_threshold(sparsity)
                layer.self_attn.sparse_fns['k'].set_threshold(sparsity)
                layer.self_attn.sparse_fns['v'].set_threshold(sparsity)
                layer.self_attn.sparse_fns['o'].set_threshold(sparsity)
        else:
            for layer in self.model.layers:
                for l in range(num_codebooks):
                    layer.self_attn.sparse_fns[f'q{l}'].set_threshold(sparsity)
                    layer.self_attn.sparse_fns[f'k{l}'].set_threshold(sparsity)
                    layer.self_attn.sparse_fns[f'v{l}'].set_threshold(sparsity)
                    layer.self_attn.sparse_fns[f'o{l}'].set_threshold(sparsity)

    def set_uniform_sparsity(self, sparsity):
        self.set_mlp_sparsity(sparsity)
        self.set_self_attn_sparsity(sparsity)

    def set_sparsities(self, sparsities):
        import re

        def parse(proj: str):
            match = re.match(r"([A-Za-z_]+)(\d+)$", proj)
            if not match:
                raise ValueError(f"Invalid proj format: {proj}")
            proj_type = match.group(1)
            proj_cb_id = int(match.group(2))
            return proj_type, proj_cb_id

        assert hasattr(self, "vq_ctx")
        vq_ctx = self.vq_ctx

        for proj, sparses in sparsities.items():
            if vq_ctx["uniform_sparse_fn"]:
                if proj in ['q', 'k', 'v', 'o']:
                    for layer, sparsity in zip(self.model.layers, sparses):
                        layer.self_attn.sparse_fns[proj].set_threshold(sparsity)
                elif proj in ['gate', 'up', 'down']:
                    for layer, sparsity in zip(self.model.layers, sparses):
                        layer.mlp.sparse_fns[proj].set_threshold(sparsity)
                else:
                    assert False, f"Unknown proj_type: {proj}"
            else:
                proj_type, proj_cb_id = parse(proj)
                if proj_type in ['gate', 'up', 'down']:
                    for layer, sparsity in zip(self.model.layers, sparses):
                        layer.mlp.sparse_fns[proj].set_threshold(sparsity)
                elif proj_type in ['q', 'k', 'v', 'o']:
                    for layer, sparsity in zip(self.model.layers, sparses):
                        layer.self_attn.sparse_fns[proj].set_threshold(sparsity)
                else:
                    assert False, f"Unknown proj_type: {proj_type} (parsed from {proj})"


class RvqLlamaSparseConfig(CustomRvqLlamaConfig):
    model_type = "rvq_llama_sparse"

class RvqLlamaSparseForCausalLM(SparseModelMixin, CustomRvqLlamaForCausalLM):
    config_class = RvqLlamaSparseConfig
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.vq_ctx = None
        # Initialize weights and apply final processing
        self.post_init()

class RvqMistralSparseConfig(CustomRvqMistralConfig):
    model_type = "rvq_mistral_sparse"

class RvqMistralSparseForCausalLM(SparseModelMixin, CustomRvqMistralForCausalLM):
    config_class = RvqMistralSparseConfig
    _no_split_modules = ["MistralDecoderLayer"]

    def __init__(self, config):
        super().__init__(config)
        self.vq_ctx = None
        # Initialize weights and apply final processing
        self.post_init()
