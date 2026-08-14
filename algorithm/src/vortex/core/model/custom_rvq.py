import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.mistral.configuration_mistral import MistralConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral.modeling_mistral import MistralForCausalLM
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoConfig
from transformers.modeling_utils import no_init_weights
from typing import Iterable
import logging
from accelerate import init_empty_weights, infer_auto_device_map, dispatch_model


class CustomRvqLinear(nn.Linear):
    def __init__(self, in_features, out_features, num_codebooks: int = 4, bias: bool = True,
                 device=None, dtype=None,):
        super().__init__(in_features, out_features, bias=bias)
        self.num_codebooks = num_codebooks
        self.assigned_device = None
        
        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(
            torch.empty((1, 1), **factory_kwargs)
        )
        self.register_buffer("wq", torch.empty(num_codebooks, out_features, in_features))
        self.register_buffer("scales", torch.empty(1, out_features, 1))


    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        
        with torch.no_grad():
            # assert prefix+"weight" in state_dict
            if prefix+"weight" in state_dict:
                assert state_dict[prefix+"weight"].shape == self.weight.shape, \
                    f"Shape mismatch for {prefix}weight: " \
                    f"expected {self.weight.shape}, got {state_dict[prefix+'weight'].shape}"
                self.weight.copy_(state_dict[prefix+"weight"])
                del state_dict[prefix+"weight"]

            wq_key = prefix + "wq"
            # assert wq_key in state_dict, f"{wq_key} not in state_dict."
            if wq_key in state_dict:
                saved_wq = state_dict[wq_key]
                if saved_wq.shape != self.wq.shape:
                    self.wq = saved_wq.clone().to(self.wq.device)
                else:
                    self.wq.copy_(saved_wq)
                del state_dict[wq_key]

            scales_key = prefix + "scales"
            # assert scales_key in state_dict
            if scales_key in state_dict:
                saved_scales = state_dict[scales_key]
                if saved_scales.shape != self.scales.shape:
                    self.scales = saved_scales.clone().to(self.scales.device)
                else:
                    self.scales.copy_(saved_scales)
                del state_dict[scales_key]
            # super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(self, input, steps: int = None):
        
        
        # num_codebooks, out_features, in_features = self.weight.shape
        num_codebooks, out_features, in_features = self.wq.shape
        num_out_group = self.scales.shape[1]
        num_in_group = self.scales.shape[2]
        in_group_size = in_features // num_in_group
        assert in_features % num_in_group == 0, "in_features must divide evenly by num_in_group"

        # Expand scales to match per-feature shape
        out_repeat = out_features // num_out_group
        in_repeat = in_features // num_in_group
        dtype = input[0].dtype
        device = self.wq.device

        if isinstance(input, list):
            assert len(input) == num_codebooks
            assert num_out_group == out_features, "NotImplemented"
            assert len(input) == num_codebooks, f"{len(input)} != {num_codebooks}; wq.shape={self.wq.shape}"

            
            if self.scales.shape[0] == 1 and self.scales.shape[2] == 1:
                compute_dytpe = input[0].dtype
                leading_dims = input[0].shape[:-1]
                output = torch.zeros((*leading_dims, out_features), dtype=compute_dytpe, device=device)
                for i in range(num_codebooks)[::-1]:
                    output += F.linear(input[i].to(compute_dytpe), (self.wq.to(compute_dytpe))[i, :, :])
                output = (output * self.scales[0, :, 0].to(compute_dytpe)).to(input[0].dtype)
            else:
                raise NotImplementedError("only consider output_dim scales now.")
            
            # Finegrained way
            # ...

        else:
            if self.scales.shape[0] == 1 and self.scales.shape[2] == 1:
                weight_sum = (self.wq).sum(dim=0).to(input.dtype)
                output = F.linear(input, weight_sum) * self.scales[0, :, 0].to(input.dtype)
            else:
                raise NotImplementedError("only consider output_dim scales now.")

        # assert self.bias is None
        if self.bias is not None:
            output += self.bias
        

        return output

    def _apply(self, fn):
        super()._apply(fn)        # ensures parameters/buffers moved
        # now your custom tensors/buffers that are not registered:
        if hasattr(self, "wq") and isinstance(self.wq, torch.Tensor):
            self.wq = fn(self.wq)
        if hasattr(self, "scales") and isinstance(self.scales, torch.Tensor):
            self.scales = fn(self.scales)
        return self

    def to(self, *args, **kwargs):
        
        """
        Override nn.Module.to() to ensure buffers (weight, scales) also move properly.
        """
        module = super().to(*args, **kwargs)

        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)

        if len(args) > 0:
            if isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], (str, int)):
                device = torch.device(args[0])
            elif isinstance(args[0], torch.dtype):
                dtype = args[0]

        logging.debug("CustomRvqLinear.to(device=%s, dtype=%s)", device, dtype)

        if hasattr(module, "weight") and isinstance(module.weight, torch.Tensor):
            module.weight = module.weight.to(device=device)
        if hasattr(module, "bias") and isinstance(module.bias, torch.Tensor):
            module.bias = module.bias.to(device=device)

        assert hasattr(module, "wq") and isinstance(module.wq, torch.Tensor)
        module.wq = module.wq.to(device=device, dtype=dtype)
        assert hasattr(module, "scales") and isinstance(module.scales, torch.Tensor)
        module.scales = module.scales.to(device=device, dtype=dtype)

        return module




class RvqConfigMixin:
    model_type: str = None

    def __init__(
        self,
        num_codebooks: int = 2,
        linear_weights_not_to_quantize: List[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_codebooks = num_codebooks
        self.linear_weights_not_to_quantize = (
            linear_weights_not_to_quantize
            or ["model.embed_tokens.weight", "lm_head.weight"]
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update(
            {
                "num_codebooks": self.num_codebooks,
                "linear_weights_not_to_quantize": self.linear_weights_not_to_quantize,
            }
        )
        return d

class CustomRvqLlamaConfig(RvqConfigMixin, LlamaConfig):
    model_type = "custom_rvq_llama"

class CustomRvqMistralConfig(RvqConfigMixin, MistralConfig):
    model_type = "custom_rvq_mistral"



class RvqMixin:
    root_module_name = "model"

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.num_codebooks = getattr(config, "num_codebooks", 8)
        self.linear_weights_not_to_quantize = getattr(config, "linear_weights_not_to_quantize", []) or []
        self.auto_device_map_tmp = None
        root = getattr(self, self.root_module_name, self)
        self._replace_linear_layers(root, prefix=self.root_module_name)

    def _replace_linear_layers(self, module: nn.Module, prefix: str = ""):
        skip = set(self.linear_weights_not_to_quantize)
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if full in skip or "lm_head" in name or "lm_head" in full:
                continue
            if isinstance(child, nn.Linear):
                bias = child.bias is not None
                with no_init_weights():
                    new = CustomRvqLinear(child.in_features, child.out_features, bias=bias, num_codebooks=self.num_codebooks)
                if bias:
                    new.bias.data.copy_(child.bias.data)
                setattr(module, name, new)
                del child
            else:
                self._replace_linear_layers(child, prefix=full)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, no_loading: bool = False, **kwargs):
        config = kwargs.pop("config", None) or AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Force cpu for the base load; our own device map is applied below.
        target_device_map = kwargs.get("device_map", "cpu")
        if "device_map" in kwargs:
            logging.warning("Overriding device_map to 'cpu' for from_pretrained, since we'll apply our own auto device map later.")
            kwargs["device_map"] = "cpu"

        # decide whether to use accelerate
        use_accelerate = False
        if not no_loading and target_device_map == "auto":
            use_accelerate = True

        if use_accelerate:
            # flexible import for different accelerate versions
            try:
                from accelerate import init_empty_weights
                # load_checkpoint_and_dispatch location may vary by version
                try:
                    from accelerate import load_checkpoint_and_dispatch
                except Exception:
                    from accelerate.utils import load_checkpoint_and_dispatch
            except Exception as e:
                raise RuntimeError("accelerate is required for device_map='auto' but could not be imported.") from e

            # 1) build the model skeleton under meta/empty weights
            with init_empty_weights():
                model = None
                constructor_kwargs = {k: v for k, v in (kwargs or {}).items() if k != "state_dict"}

                # 1) ideal: next class in MRO implements from_config
                try:
                    model = super(RvqMixin, cls).from_config(config, **constructor_kwargs)
                except Exception as e_from_config:
                    # 2) common in HF: parents expose _from_config (internal)
                    try:
                        model = super(RvqMixin, cls)._from_config(config, **constructor_kwargs)
                    except Exception as e_from__config:
                        # 3) try class-level _from_config (in case mixin ordering hides parent lookup)
                        try:
                            model = cls._from_config(config, **constructor_kwargs)
                        except Exception as e_cls__from_config:
                            # 4) final fallback: call constructor directly under init_empty_weights()
                            try:
                                # Keep ctor kwargs minimal if needed; some classes dislike unexpected kwargs.
                                # Use only config unless you know cls.__init__ needs more.
                                model = cls(config, **{k: v for k, v in constructor_kwargs.items() if k in ("trust_remote_code",)})
                            except Exception as e_ctor:
                                # Diagnostic: produce an MRO/method table for debugging
                                mro_info = []
                                for b in cls.mro():
                                    mro_info.append(
                                        (b.__name__,
                                        bool(getattr(b, "from_config", None)),
                                        bool(getattr(b, "_from_config", None)),
                                        bool(getattr(b, "from_pretrained", None)))
                                    )
                                raise RuntimeError(
                                    "Failed to construct empty-weight model skeleton via "
                                    "super(...).from_config, super(...)._from_config, cls._from_config, and cls(...). "
                                    f"MRO/method availability: {mro_info}"
                                ) from e_ctor


            # 2) restore the config-derived fields
            model.num_codebooks = getattr(config, "num_codebooks", 8)
            model.linear_weights_not_to_quantize = getattr(config, "linear_weights_not_to_quantize", []) or []

            # 3) swap in CustomRvqLinear on the empty skeleton, keeping module/param names
            root = getattr(model, getattr(cls, "root_module_name", "model"), model)
            model._replace_linear_layers(root, prefix=getattr(cls, "root_module_name", "model"))

            # 4) load and dispatch the weights with accelerate (handles sharded checkpoints)
            #    the usual knobs are read from config/kwargs; adjust as needed
            
            decoder_layer_cls = model.model.layers[0].__class__
            self_attn_cls = model.model.layers[0].self_attn.__class__
            
            # assert False

            no_split_classes = [
                decoder_layer_cls, 
                self_attn_cls,
                decoder_layer_cls.__name__,
                ]
            load_kwargs = dict(
                device_map="auto",
                no_split_module_classes=no_split_classes,
            )
            # optional offload folder / dtype
            if getattr(config, "offload_folder", None):
                load_kwargs["offload_folder"] = config.offload_folder
            if getattr(config, "dtype", None):
                load_kwargs["dtype"] = config.dtype

            try:
                model = load_checkpoint_and_dispatch(
                    model,
                    pretrained_model_name_or_path,
                    **load_kwargs,
                )
            except Exception as e:
                
                # Fail loudly, then fall back to transformers' load_sharded_checkpoint.
                logging.error(f"Accelerate load failed: {e}")
                assert False, f"Accelerate load failed: {e}"
                try:
                    from transformers.modeling_utils import load_sharded_checkpoint
                    # assumes the model can be loaded in its non-dispatched form
                    load_sharded_checkpoint(model, pretrained_model_name_or_path)
                except Exception as e2:
                    raise RuntimeError(f"Failed to load checkpoint with accelerate and fallback failed: {e2}") from e

        else:
            # the original path, without accelerate
            model = super().from_pretrained(pretrained_model_name_or_path, config=config, **kwargs)
            model.num_codebooks = getattr(config, "num_codebooks", 8)
            model.linear_weights_not_to_quantize = getattr(config, "linear_weights_not_to_quantize", []) or []
            root = getattr(model, getattr(cls, "root_module_name", "model"), model)
            model._replace_linear_layers(root, prefix=getattr(cls, "root_module_name", "model"))
            if not no_loading:
                try:
                    from transformers.modeling_utils import load_sharded_checkpoint
                    load_sharded_checkpoint(model, pretrained_model_name_or_path)
                except Exception:
                    # pass
                    assert False
            device_map = target_device_map
            if device_map == "cuda":
                model = model.to("cuda")

        from vortex.teal.utils.utils_model import utils_module_param_buffer_devices
        for i, layer in enumerate(model.model.layers):
            assert len(utils_module_param_buffer_devices(layer)) == 1, f"Decoder Layer {i} has parameters/buffers on multiple devices: {utils_module_param_buffer_devices(layer)}"

        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        self.config.linear_weights_not_to_quantize = list(self.linear_weights_not_to_quantize)
        self.config.num_codebooks = int(self.num_codebooks)
        return super().save_pretrained(save_directory, **kwargs)

class CustomRvqLlamaForCausalLM(RvqMixin, LlamaForCausalLM):
    root_module_name = "rvq_llama_model"
    config_class = CustomRvqLlamaConfig

class CustomRvqMistralForCausalLM(RvqMixin, MistralForCausalLM):
    root_module_name = "rvq_mistral_model"
    config_class = CustomRvqMistralConfig
