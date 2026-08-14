from vortex.core.model.custom_rvq import CustomRvqLinear
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Any

def resolve_device(device_str: str) -> torch.device:
    """
    Normalize a device string into a concrete torch.device.
    - "cpu" stays as cpu
    - "cuda" resolves to the current default GPU (e.g., cuda:0)
    - "cuda:N" stays unchanged
    """
    d = torch.device(device_str)
    if d.type == "cuda" and d.index is None:
        # Resolve to the current CUDA device (usually 0 unless changed)
        return torch.device(f"cuda:{torch.cuda.current_device()}")
    return d

def compare_devices(device1, device2) -> bool:
    """
    Compare two devices (str or torch.device) for equality after normalization.
    - Accepts str (e.g., "cuda", "cuda:0", "cpu") or torch.device
    - "cuda" is normalized to the current default device (e.g., cuda:0)
    """
    def normalize(dev):
        if isinstance(dev, str):
            return resolve_device(dev)
        elif isinstance(dev, torch.device):
            return resolve_device(str(dev))
        else:
            raise TypeError(f"Unsupported device type: {type(dev)}")

    return normalize(device1) == normalize(device2)

def get_gpu_memory(device: torch.device = None) -> Tuple[int, int]:
    """Return (free_memory_bytes, total_memory_bytes) for the given CUDA device."""
    if device is None:
        # provide info for all available CUDA devices
        info = {}
        for i in range(torch.cuda.device_count()):
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
            info["cuda:"+str(i)] = {
                "free_bytes": free_bytes,
                "total_bytes": total_bytes,
            }
        return info
    raise NotImplementedError("get_gpu_memory for a single device is not implemented yet.")

def _move_obj_to_device(obj: Any, device: torch.device):
    """Recursively move tensor-like objects to device (handles tuple/list/dict/Tensor)."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, (list, tuple)):
        moved = [_move_obj_to_device(x, device) for x in obj]
        return type(obj)(moved)
    if isinstance(obj, dict):
        return {k: _move_obj_to_device(v, device) for k, v in obj.items()}
    return obj

def _make_pre_hook(target_device: torch.device):
    """Return a forward_pre_hook that moves inputs to target_device."""
    def pre_hook(module, inputs):
        # inputs is a tuple. Move nested tensors while preserving structure.
        moved_inputs = _move_obj_to_device(inputs, target_device)
        # The hook may return None or the new input tuple
        return moved_inputs
    return pre_hook


def device_map_auto(model, device=None) -> dict:
    """
    Generate a device map for the model, assigning all modules to the current default CUDA device.
    """
    
    # 1. get available cuda devices and their available memory

    dev_count = torch.cuda.device_count()
    devices: List[Dict] = []
    for i in range(dev_count):
        dev = torch.device(f"cuda:{i}")
        # try to get free memory in bytes; fallback to None if not available
        free_bytes = None
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(i)
        except Exception:
            free_bytes = None
        devices.append({"index": i, "device": dev, "free": free_bytes if free_bytes is not None else 0})
    
    # 2. get required memory for each module
    module_mem_reqs: List[Tuple[str, int]] = []  # (module_name, mem_bytes)
    for name, module in model.named_modules():
        if isinstance(module, CustomRvqLinear):
            param_size = sum(p.numel() * p.element_size() \
                    for p in [module.wq, module.scales, module.bias] \
                    if p is not None)
            module_mem_reqs.append((name, param_size))
        if isinstance(module, nn.Linear):
            param_size = sum(p.numel() * p.element_size() \
                    for p in module.parameters() \
                    if p is not None)
            module_mem_reqs.append((name, param_size))
    
    # 3. plan assginments
    device_map: Dict[str, torch.device] = {}
    total_device_free_mem = sum(dev["free"] for dev in devices)
    total_module_mem_req = sum(mem for _, mem in module_mem_reqs)
    assert total_module_mem_req < 0.5 * total_device_free_mem, \
        "Not enough total GPU memory to fit the model modules."
    GB = 1024 ** 3
    planned_chunk_bytes = 4 * GB
    # state for chunking
    last_idx = None            # index in devices of the current reserved chunk
    last_planned = 0           # bytes remaining in the reserved chunk
    for module_name, mem_req in module_mem_reqs:
        if last_idx is not None and last_planned >= mem_req:
            # assign to the last reserved chunk
            device_map[module_name] = devices[last_idx]["device"]
            last_planned -= mem_req
            devices[last_idx]["free"] -= mem_req
        else:
            # need to find a new chunk
            assigned = False
            for dev in devices:
                if dev["free"] >= max(mem_req, planned_chunk_bytes):
                    device_map[module_name] = dev["device"]
                    dev["free"] -= mem_req
                    # reserve a new chunk
                    last_idx = dev["index"]
                    last_planned = planned_chunk_bytes - mem_req
                    assigned = True
                    break
            if not assigned:
                raise RuntimeError(f"Cannot assign module {module_name} due to insufficient GPU memory.")


    # 4. assign modules evenly to devices
        # If Linear/CustomRvqLinear layer, 
        # create hooks that change the device of inputs,
        # move layer to the assigned device.
    for name, module in model.named_modules():
        if name not in device_map:
            continue
        assigned_device_str = device_map[name]
        # canonicalize to torch.device
        assigned_device = torch.device(assigned_device_str)

        # Try to move the module to the device
        try:
            module.to(assigned_device)
        except Exception:
            # Some modules (ScriptModule, custom wrappers) may not support .to()
            # but we still want to register hooks if appropriate
            pass



        # For plain nn.Linear, register a pre-hook to move inputs
        if isinstance(module, nn.Linear):
            if not getattr(module, "_device_map_hook_installed", False):
                hook = module.register_forward_pre_hook(_make_pre_hook(assigned_device))
                module._device_map_hook_installed = True

    return model


def check_module_device(module, device):
    for name, param in module.named_parameters():
        if param.device != device:
            print(f"Parameter {name} is on {param.device}, expected {device}")
            return False

    for name, buf in module.named_buffers():
        if buf.device != device:
            print(f"Buffer {name} is on {buf.device}, expected {device}")
            return False

    print("✓ All parameters and buffers are on", device)
    return True