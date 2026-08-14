"""Per-model layer descriptions: shapes, head counts, and op sequences.

Each model function returns an ordered mapping of op name to dimensions, from
which `networks.py` builds the op list. Three models carry the paper's results:
llama-2-7b, llama_2_13b and mistral_7b.
"""
from collections import OrderedDict


def llama_7b(sequence_length=1):
    return {
        "model_type": "dense",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "expert_intermediate_size": 11008,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 2048,
        "layer": OrderedDict([
            ("fc_q_proj", [4096, 4096, sequence_length]),
            ("fc_k_proj", [4096, 4096, sequence_length]),
            ("fc_v_proj", [4096, 4096, sequence_length]),
            ("attention", [4096, sequence_length, 8]),
            ("fc_o_proj", [4096, 4096, sequence_length]),
            ("fc_gate_proj", [11008, 4096, sequence_length]),
            ("fc_up_proj", [11008, 4096, sequence_length]),
            ("fc_down_proj", [4096, 11008, sequence_length]),
        ]),
    }


def llama_13b(sequence_length=1):
    return {
        "model_type": "dense",
        "hidden_size": 5120,
        "num_layers": 40,
        "num_attention_heads": 40,
        "num_kv_heads": 40,
        "head_dim": 128,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "expert_intermediate_size": 13824,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 2048,
        "layer": OrderedDict([
            ("fc_q_proj", [5120, 5120, sequence_length]),
            ("fc_k_proj", [5120, 5120, sequence_length]),
            ("fc_v_proj", [5120, 5120, sequence_length]),
            ("attention", [5120, sequence_length, 8]),
            ("fc_o_proj", [5120, 5120, sequence_length]),
            ("fc_gate_proj", [13824, 5120, sequence_length]),
            ("fc_up_proj", [13824, 5120, sequence_length]),
            ("fc_down_proj", [5120, 13824, sequence_length]),
        ]),
    }


def llama_30b(sequence_length=1):
    return {
        "model_type": "dense",
        "hidden_size": 6656,
        "num_layers": 60,
        "num_attention_heads": 52,
        "num_kv_heads": 52,
        "head_dim": 128,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "expert_intermediate_size": 17920,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 2048,
        "layer": OrderedDict([
            ("fc_q_proj", [6656, 6656, sequence_length]),
            ("fc_k_proj", [6656, 6656, sequence_length]),
            ("fc_v_proj", [6656, 6656, sequence_length]),
            ("attention", [6656, sequence_length, 8]),
            ("fc_o_proj", [6656, 6656, sequence_length]),
            ("fc_gate_proj", [17920, 6656, sequence_length]),
            ("fc_up_proj", [17920, 6656, sequence_length]),
            ("fc_down_proj", [6656, 17920, sequence_length]),
        ]),
    }


def llama_65b(sequence_length=1):
    return {
        "model_type": "dense",
        "hidden_size": 8192,
        "num_layers": 80,
        "num_attention_heads": 64,
        "num_kv_heads": 64,
        "head_dim": 128,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "expert_intermediate_size": 22016,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 2048,
        "layer": OrderedDict([
            ("fc_q_proj", [8192, 8192, sequence_length]),
            ("fc_k_proj", [8192, 8192, sequence_length]),
            ("fc_v_proj", [8192, 8192, sequence_length]),
            ("attention", [8192, sequence_length, 8]),
            ("fc_o_proj", [8192, 8192, sequence_length]),
            ("fc_gate_proj", [22016, 8192, sequence_length]),
            ("fc_up_proj", [22016, 8192, sequence_length]),
            ("fc_down_proj", [8192, 22016, sequence_length]),
        ]),
    }


def llama_2_7b(sequence_length=1):
    return {
        "model_type": "dense",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 32,
        "head_dim": 128,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "expert_intermediate_size": 11008,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 4096,
        "layer": OrderedDict([
            ("fc_q_proj", [4096, 4096, sequence_length]),
            ("fc_k_proj", [4096, 4096, sequence_length]),
            ("fc_v_proj", [4096, 4096, sequence_length]),
            ("attention", [4096, sequence_length, 8]),
            ("fc_o_proj", [4096, 4096, sequence_length]),
            ("fc_gate_proj", [11008, 4096, sequence_length]),
            ("fc_up_proj", [11008, 4096, sequence_length]),
            ("fc_down_proj", [4096, 11008, sequence_length]),
        ]),
    }


def llama_2_13b(sequence_length=1):
    return {
        "model_type": "dense",
        "hidden_size": 5120,
        "num_layers": 40,
        "num_attention_heads": 40,
        "num_kv_heads": 40,
        "head_dim": 128,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "expert_intermediate_size": 13824,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 4096,
        "layer": OrderedDict([
            ("fc_q_proj", [5120, 5120, sequence_length]),
            ("fc_k_proj", [5120, 5120, sequence_length]),
            ("fc_v_proj", [5120, 5120, sequence_length]),
            ("attention", [5120, sequence_length, 8]),
            ("fc_o_proj", [5120, 5120, sequence_length]),
            ("fc_gate_proj", [13824, 5120, sequence_length]),
            ("fc_up_proj", [13824, 5120, sequence_length]),
            ("fc_down_proj", [5120, 13824, sequence_length]),
        ]),
    }


def llama3_8b(sequence_length=1):
    return {
        "model_type": "dense",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "expert_intermediate_size": 14336,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 8192,
        "layer": OrderedDict([
            ("fc_q_proj", [4096, 4096, sequence_length]),
            ("fc_k_proj", [1024, 4096, sequence_length]),
            ("fc_v_proj", [1024, 4096, sequence_length]),
            ("attention", [4096, sequence_length, 8]),
            ("fc_o_proj", [4096, 4096, sequence_length]),
            ("fc_gate_proj", [14336, 4096, sequence_length]),
            ("fc_up_proj", [14336, 4096, sequence_length]),
            ("fc_down_proj", [4096, 14336, sequence_length]),
        ]),
    }


def llama_3_1_8b(sequence_length=1):
    return {
        "model_type": "dense",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "expert_intermediate_size": 14336,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 4096,
        "layer": OrderedDict([
            ("fc_q_proj", [4096, 4096, sequence_length]),
            ("fc_k_proj", [1024, 4096, sequence_length]),
            ("fc_v_proj", [1024, 4096, sequence_length]),
            ("attention", [4096, sequence_length, 8]),
            ("fc_o_proj", [4096, 4096, sequence_length]),
            ("fc_gate_proj", [14336, 4096, sequence_length]),
            ("fc_up_proj", [14336, 4096, sequence_length]),
            ("fc_down_proj", [4096, 14336, sequence_length]),
        ]),
    }


def mistral_7b(sequence_length=1):
    return {
        "model_type": "dense",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_experts": 1,
        "num_experts_per_tok": 1,
        "expert_intermediate_size": 14336,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 32768,
        "layer": OrderedDict([
            ("fc_q_proj", [4096, 4096, sequence_length]),
            ("fc_k_proj", [1024, 4096, sequence_length]),
            ("fc_v_proj", [1024, 4096, sequence_length]),
            ("attention", [4096, sequence_length, 8]),
            ("fc_o_proj", [4096, 4096, sequence_length]),
            ("fc_gate_proj", [14336, 4096, sequence_length]),
            ("fc_up_proj", [14336, 4096, sequence_length]),
            ("fc_down_proj", [4096, 14336, sequence_length]),
        ]),
    }


def mixtral_8x7b(sequence_length=1):
    return {
        "model_type": "moe",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_attention_heads": 32,
        "num_kv_heads": 8,
        "head_dim": 128,
        "num_experts": 8,
        "num_experts_per_tok": 2,
        "expert_intermediate_size": 14336,
        "shared_expert": False,
        "shared_expert_intermediate_size": 0,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 32768,
        "layer": OrderedDict([
            ("fc_q_proj", [4096, 4096, sequence_length]),
            ("fc_k_proj", [1024, 4096, sequence_length]),
            ("fc_v_proj", [1024, 4096, sequence_length]),
            ("attention", [4096, sequence_length, 8]),
            ("fc_o_proj", [4096, 4096, sequence_length]),
            ("sfu_rmsnorm", [sequence_length, 4096]),
            ("fc_gate_proj", [14336, 4096, sequence_length]),
            ("fc_up_proj", [14336, 4096, sequence_length]),
            ("sfu_swiglu", [sequence_length, 14336]),
            ("fc_down_proj", [4096, 14336, sequence_length]),
            ("sfu_rmsnorm_2", [sequence_length, 4096]),
        ]),
    }


def qwen3_30b_a3b(sequence_length=1):
    return {
        "model_type": "moe",
        "hidden_size": 2048,
        "num_layers": 48,
        "num_attention_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 128,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "expert_intermediate_size": 768,
        "shared_expert": True,
        "shared_expert_intermediate_size": 6144,
        "activation": "silu",
        "sequence_length": sequence_length,
        "max_input_length": 40960,
        "layer": OrderedDict([
            ("fc_q_proj", [4096, 2048, sequence_length]),
            ("fc_k_proj", [512, 2048, sequence_length]),
            ("fc_v_proj", [512, 2048, sequence_length]),
            ("attention", [4096, sequence_length, 4]),
            ("fc_o_proj", [2048, 4096, sequence_length]),
            ("sfu_rmsnorm", [sequence_length, 2048]),
            ("fc_gate_proj", [768, 2048, sequence_length]),
            ("fc_up_proj", [768, 2048, sequence_length]),
            ("sfu_swiglu", [sequence_length, 768]),
            ("fc_down_proj", [2048, 768, sequence_length]),
            ("sfu_swiglu_shared", [sequence_length, 6144]),
            ("sfu_rmsnorm_2", [sequence_length, 2048]),
        ]),
    }


MODEL_REGISTRY = {
    "llama_7b": llama_7b,
    "llama_13b": llama_13b,
    "llama_30b": llama_30b,
    "llama_65b": llama_65b,
    "llama_2_7b": llama_2_7b,
    "llama_2_13b": llama_2_13b,
    "llama3_8b": llama3_8b,
    "llama_3_1_8b": llama_3_1_8b,
    "mistral_7b": mistral_7b,
    "mixtral_8x7b": mixtral_8x7b,
    "qwen3_30b_a3b": qwen3_30b_a3b,
}

MODEL_NAME_ALIASES = {
    "llama-2-7b": "llama_2_7b",
    "llama-2-13b": "llama_2_13b",
    "llama-3.1-8b": "llama_3_1_8b",
    "llama-3-8b": "llama3_8b",
    "mistral-7b": "mistral_7b",
    "mixtral-8x7b": "mixtral_8x7b",
    "qwen3-30b-a3b": "qwen3_30b_a3b",
}


def normalize_model_name(model_name: str) -> str:
    return MODEL_NAME_ALIASES.get(model_name, model_name)


def get_model_definition(model_name: str, sequence_length: int = 1) -> dict:
    normalized = normalize_model_name(model_name)
    if normalized not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")
    return MODEL_REGISTRY[normalized](sequence_length)


# Not used by the simulator; only by this module's own __main__ demo below.
def get_model_config(model_name: str) -> OrderedDict:
    normalized = normalize_model_name(model_name)

    if normalized == "llama_7b":
        return OrderedDict([
            ("layer_num", 32),
            ("is_moe", False),
            ("num_experts", 1),
            ("num_experts_per_token", 1),
            ("fc_q", [4096, 4096]),
            ("fc_k", [4096, 4096]),
            ("fc_v", [4096, 4096]),
            ("attention", [32, 128, 1]),
            ("fc_o", [4096, 4096]),
            ("fc_gate", [11008, 4096]),
            ("fc_up", [11008, 4096]),
            ("fc_down", [4096, 11008]),
        ])
    elif normalized == "llama_13b":
        return OrderedDict([
            ("layer_num", 40),
            ("is_moe", False),
            ("num_experts", 1),
            ("num_experts_per_token", 1),
            ("fc_q", [5120, 5120]),
            ("fc_k", [5120, 5120]),
            ("fc_v", [5120, 5120]),
            ("attention", [40, 128, 1]),
            ("fc_o", [5120, 5120]),
            ("fc_gate", [13824, 5120]),
            ("fc_up", [13824, 5120]),
            ("fc_down", [5120, 13824]),
        ])
    elif normalized == "llama_30b":
        return OrderedDict([
            ("layer_num", 60),
            ("is_moe", False),
            ("num_experts", 1),
            ("num_experts_per_token", 1),
            ("fc_q", [6656, 6656]),
            ("fc_k", [6656, 6656]),
            ("fc_v", [6656, 6656]),
            ("attention", [52, 128, 1]),
            ("fc_o", [6656, 6656]),
            ("fc_gate", [17920, 6656]),
            ("fc_up", [17920, 6656]),
            ("fc_down", [6656, 17920]),
        ])
    elif normalized == "llama_65b":
        return OrderedDict([
            ("layer_num", 80),
            ("is_moe", False),
            ("num_experts", 1),
            ("num_experts_per_token", 1),
            ("fc_q", [8192, 8192]),
            ("fc_k", [8192, 8192]),
            ("fc_v", [8192, 8192]),
            ("attention", [64, 128, 1]),
            ("fc_o", [8192, 8192]),
            ("fc_gate", [22016, 8192]),
            ("fc_up", [22016, 8192]),
            ("fc_down", [8192, 22016]),
        ])
    elif normalized == "llama_2_7b":
        return OrderedDict([
            ("layer_num", 32),
            ("is_moe", False),
            ("num_experts", 1),
            ("num_experts_per_token", 1),
            ("fc_q", [4096, 4096]),
            ("fc_k", [4096, 4096]),
            ("fc_v", [4096, 4096]),
            ("attention", [32, 128, 1]),
            ("fc_o", [4096, 4096]),
            ("fc_gate", [11008, 4096]),
            ("fc_up", [11008, 4096]),
            ("fc_down", [4096, 11008]),
        ])
    elif normalized == "llama_2_13b":
        return OrderedDict([
            ("layer_num", 40),
            ("is_moe", False),
            ("num_experts", 1),
            ("num_experts_per_token", 1),
            ("fc_q", [5120, 5120]),
            ("fc_k", [5120, 5120]),
            ("fc_v", [5120, 5120]),
            ("attention", [40, 128, 1]),
            ("fc_o", [5120, 5120]),
            ("fc_gate", [13824, 5120]),
            ("fc_up", [13824, 5120]),
            ("fc_down", [5120, 13824]),
        ])
    elif normalized == "llama3_8b":
        return OrderedDict([
            ("layer_num", 32),
            ("is_moe", False),
            ("num_experts", 1),
            ("num_experts_per_token", 1),
            ("fc_q", [4096, 4096]),
            ("fc_k", [1024, 4096]),
            ("fc_v", [1024, 4096]),
            ("attention", [32, 128, 4]),
            ("fc_o", [4096, 4096]),
            ("fc_gate", [14336, 4096]),
            ("fc_up", [14336, 4096]),
            ("fc_down", [4096, 14336]),
        ])
    elif normalized == "llama_3_1_8b":
        return OrderedDict([
            ("layer_num", 32),
            ("is_moe", False),
            ("num_experts", 1),
            ("num_experts_per_token", 1),
            ("fc_q", [4096, 4096]),
            ("fc_k", [1024, 4096]),
            ("fc_v", [1024, 4096]),
            ("fc_o", [4096, 4096]),
            ("attention", [32, 128, 4]),
            ("fc_gate", [14336, 4096]),
            ("fc_up", [14336, 4096]),
            ("fc_down", [4096, 14336]),
        ])
    elif normalized == "mistral_7b":
        return OrderedDict([
            ("layer_num", 32),
            ("is_moe", False),
            ("num_experts", 1),
            ("num_experts_per_token", 1),
            ("fc_q", [4096, 4096]),
            ("fc_k", [1024, 4096]),
            ("fc_v", [1024, 4096]),
            ("attention", [32, 128, 4]),
            ("fc_o", [4096, 4096]),
            ("fc_gate", [14336, 4096]),
            ("fc_up", [14336, 4096]),
            ("fc_down", [4096, 14336]),
        ])
    elif normalized == "mixtral_8x7b":
        return OrderedDict([
            ("layer_num", 32),
            ("is_moe", True),
            ("num_experts", 8),
            ("num_experts_per_token", 2),
            ("fc_q", [4096, 4096]),
            ("fc_k", [1024, 4096]),
            ("fc_v", [1024, 4096]),
            ("fc_o", [4096, 4096]),
            ("attention", [32, 128, 4]),
            ("fc_gate", [14336, 4096]),
            ("fc_up", [14336, 4096]),
            ("fc_down", [4096, 14336]),
        ])
    elif normalized == "qwen3_30b_a3b":
        return OrderedDict([
            ("layer_num", 48),
            ("is_moe", True),
            ("num_experts", 128),
            ("num_experts_per_token", 8),
            ("fc_q", [4096, 2048]),
            ("fc_k", [512, 2048]),
            ("fc_v", [512, 2048]),
            ("fc_o", [2048, 4096]),
            ("attention", [32, 128, 8]),
            ("fc_gate", [768, 2048]),
            ("fc_up", [768, 2048]),
            ("fc_down", [2048, 768]),
        ])
    raise ValueError(f"Unsupported model: {model_name}")


def get_model_max_input_length(model_name: str) -> int:
    normalized = normalize_model_name(model_name)
    if normalized == "llama_7b":
        return 2048
    elif normalized == "llama_13b":
        return 2048
    elif normalized == "llama_30b":
        return 2048
    elif normalized == "llama_65b":
        return 2048
    elif normalized == "llama_2_7b":
        return 4096
    elif normalized == "llama_2_13b":
        return 4096
    elif normalized == "llama3_8b":
        return 8192
    elif normalized == "llama_3_1_8b":
        return 4096
    elif normalized == "mistral_7b":
        return 32768
    elif normalized == "mixtral_8x7b":
        return 32768
    elif normalized == "qwen3_30b_a3b":
        return 40960
    raise ValueError(f"Unsupported model: {model_name}")


if __name__ == "__main__":
    config = get_model_config("qwen3-30b-a3b")
    print("Model Config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("is_moe:", config.get("is_moe", False))