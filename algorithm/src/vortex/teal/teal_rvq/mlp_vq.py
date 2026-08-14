import sys
import os
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# sys.path.append(parent_dir)
# sys.path.append(os.path.join(parent_dir, 'utils'))

import types
from torch import nn

# from vortex.teal.utils.utils_vq import ActivationModule, Distribution, SparsifyFn, get_module_device


def _monkeypatch_mlp(mlp, file_path, grabbing_mode=False, dist_args=None, sparse_fn_args=None):
    from vortex.teal.utils.utils_activations import \
        ActivationModule, Distribution, SparsifyFn
    from vortex.teal.utils.utils_model import get_module_device

    assert hasattr(mlp, "vq_ctx")
    vq_ctx = mlp.vq_ctx

    mlp.forward_old = mlp.forward
    mlp.forward = types.MethodType(_mlp_forward, mlp)

    mlp.file_path = file_path
    mlp.grabbing_mode = grabbing_mode

    if not grabbing_mode:
        mlp.distrs = {}
        mlp.distrs['h1'] = Distribution(file_path, hidden_type='h1', **(dist_args or {}))
        mlp.distrs['h2'] = Distribution(file_path, hidden_type='h2', **(dist_args or {}))


        if vq_ctx["uniform_sparse_fn"]:
            mlp.sparse_fns = nn.ModuleDict({
                'gate': SparsifyFn(mlp.distrs['h1'], **(sparse_fn_args or {})).to(get_module_device(mlp)),
                'up': SparsifyFn(mlp.distrs['h1'], **(sparse_fn_args or {})).to(get_module_device(mlp)),
                'down': SparsifyFn(mlp.distrs['h2'], **(sparse_fn_args or {})).to(get_module_device(mlp)),
            })
        else:
            tmp_dict = {}
            for proj_name in ['gate', 'up', 'down']:
                # num_codebooks = vq_ctx['config_num_codebooks'][proj_name]
                num_codebooks = vq_ctx['config_num_codebooks_glb']
                for l in range(num_codebooks):
                    if proj_name == 'down':
                        tmp_dict[proj_name + f"{l}"] = SparsifyFn(mlp.distrs['h2'], **(sparse_fn_args or {})).to(get_module_device(mlp))
                    else:
                        tmp_dict[proj_name + f"{l}"] = SparsifyFn(mlp.distrs['h1'], **(sparse_fn_args or {})).to(get_module_device(mlp))
            mlp.sparse_fns = nn.ModuleDict(tmp_dict)

    mlp.activation_module = ActivationModule(file_path, **(dist_args or {}))

    return mlp

def _mlp_forward(self, x, activation_module=None, vq_ctx=None):
    
    assert hasattr(self, "vq_ctx")
    vq_ctx = self.vq_ctx
    
    if hasattr(self, 'config') and self.config.pretraining_tp > 1:
        # TODO: UNTESTED

        assert 1 == 0, "Pretraining TP > 1 not implemented yet"
    else:
        if self.grabbing_mode:
            self.activation_module.grab_activations(x, 'h1')

            intermediate_states = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
            self.activation_module.grab_activations(intermediate_states, 'h2')
            down_proj = self.down_proj(intermediate_states)
        else:
            if vq_ctx["uniform_sparse_fn"]:
                x_gate = self.sparse_fns['gate'](x)
                x_up = self.sparse_fns['up'](x)

                intermediate_states = self.act_fn(self.gate_proj(x_gate)) * self.up_proj(x_up)
                intermediate_states = self.sparse_fns['down'](intermediate_states)

                down_proj = self.down_proj(intermediate_states)
            else:
                num_codebooks = vq_ctx['config_num_codebooks_glb']
                x_gate = []
                x_up = []
                for l in range(num_codebooks):
                    x_gate.append(self.sparse_fns[f'gate{l}'](x))
                    x_up.append(self.sparse_fns[f'up{l}'](x))
                intermediate_states = self.act_fn(self.gate_proj(x_gate)) * self.up_proj(x_up)
                x_intermediate = []
                for l in range(num_codebooks):
                    x_intermediate.append(self.sparse_fns[f'down{l}'](intermediate_states))
                down_proj = self.down_proj(x_intermediate)

    return down_proj