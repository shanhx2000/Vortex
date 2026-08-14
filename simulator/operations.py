"""Operation types and tensor descriptors.

Each op carries shapes plus `A_info` / `B_info` / `out_info` dicts describing
dtype, memory location, and (once quantisation is applied) a quantization_config.
Op *names* are load-bearing: the dataflow selector keys on "sv" to force LUF,
and the sparsity loader parses "layer<N>" and "<x>_proj" out of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional


DEFAULT_DTYPE = "fp16"
DEFAULT_LOC = "dram"


@dataclass
class OP:
    """Base operation class used by network and hardware simulators."""

    name: str
    parent_name: Optional[str] = None
    op_type: str = "op"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.parent_name is None:
            self.parent_name = self.name

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parent_name": self.parent_name,
            "op_type": self.op_type,
            "metadata": self.metadata,
        }


@dataclass
class Idle(OP):
    cycles: int = 1

    def __init__(self, name: str = "idle", parent_name: Optional[str] = None, cycles: int = 1):
        super().__init__(name=name, parent_name=parent_name, op_type="idle")
        self.cycles = cycles


@dataclass
class GEMM(OP):
    """Matrix multiply: A(MxK) x B(KxN) -> O(MxN)."""

    M: int = 1
    K: int = 1
    N: int = 1
    A_info: Dict[str, Any] = field(default_factory=dict)
    B_info: Dict[str, Any] = field(default_factory=dict)
    out_info: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        M: int,
        K: int,
        N: int,
        A_info: Optional[Dict[str, Any]] = None,
        B_info: Optional[Dict[str, Any]] = None,
        out_info: Optional[Dict[str, Any]] = None,
        parent_name: Optional[str] = None,
    ):
        super().__init__(name=name, parent_name=parent_name, op_type="gemm")
        self.M = M
        self.K = K
        self.N = N
        self.A_info = _normalize_tensor_info(A_info)
        self.B_info = _normalize_tensor_info(B_info)
        self.out_info = _normalize_tensor_info(out_info)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.M, self.K, self.N


@dataclass
class GEMV(GEMM):
    """Matrix-vector multiply represented as GEMM with M=1."""

    def __init__(
        self,
        name: str,
        K: int,
        N: int,
        A_info: Optional[Dict[str, Any]] = None,
        B_info: Optional[Dict[str, Any]] = None,
        out_info: Optional[Dict[str, Any]] = None,
        parent_name: Optional[str] = None,
    ):
        super().__init__(
            name=name,
            M=1,
            K=K,
            N=N,
            A_info=A_info,
            B_info=B_info,
            out_info=out_info,
            parent_name=parent_name,
        )
        self.op_type = "gemv"


@dataclass
class VMul(OP):
    N: int = 1
    A_info: Dict[str, Any] = field(default_factory=dict)
    B_info: Dict[str, Any] = field(default_factory=dict)
    out_info: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        N: int,
        A_info: Optional[Dict[str, Any]] = None,
        B_info: Optional[Dict[str, Any]] = None,
        out_info: Optional[Dict[str, Any]] = None,
        parent_name: Optional[str] = None,
    ):
        super().__init__(name=name, parent_name=parent_name, op_type="vmul")
        self.N = N
        self.A_info = _normalize_tensor_info(A_info)
        self.B_info = _normalize_tensor_info(B_info)
        self.out_info = _normalize_tensor_info(out_info)


@dataclass
class VAdd(OP):
    N: int = 1
    A_info: Dict[str, Any] = field(default_factory=dict)
    B_info: Dict[str, Any] = field(default_factory=dict)
    out_info: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        N: int,
        A_info: Optional[Dict[str, Any]] = None,
        B_info: Optional[Dict[str, Any]] = None,
        out_info: Optional[Dict[str, Any]] = None,
        parent_name: Optional[str] = None,
    ):
        super().__init__(name=name, parent_name=parent_name, op_type="vadd")
        self.N = N
        self.A_info = _normalize_tensor_info(A_info)
        self.B_info = _normalize_tensor_info(B_info)
        self.out_info = _normalize_tensor_info(out_info)


@dataclass
class Sum(OP):
    N: int = 1
    I_info: Dict[str, Any] = field(default_factory=dict)
    out_info: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        N: int,
        I_info: Optional[Dict[str, Any]] = None,
        out_info: Optional[Dict[str, Any]] = None,
        parent_name: Optional[str] = None,
    ):
        super().__init__(name=name, parent_name=parent_name, op_type="sum")
        self.N = N
        self.I_info = _normalize_tensor_info(I_info)
        self.out_info = _normalize_tensor_info(out_info)


@dataclass
class Softmax(OP):
    N: int = 1
    I_info: Dict[str, Any] = field(default_factory=dict)
    out_info: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        N: int,
        I_info: Optional[Dict[str, Any]] = None,
        out_info: Optional[Dict[str, Any]] = None,
        parent_name: Optional[str] = None,
    ):
        super().__init__(name=name, parent_name=parent_name, op_type="softmax")
        self.N = N
        self.I_info = _normalize_tensor_info(I_info)
        self.out_info = _normalize_tensor_info(out_info)


@dataclass
class Quantize(OP):
    N: int = 1
    I_info: Dict[str, Any] = field(default_factory=dict)
    out_info: Dict[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        N: int,
        I_info: Optional[Dict[str, Any]] = None,
        out_info: Optional[Dict[str, Any]] = None,
        parent_name: Optional[str] = None,
    ):
        super().__init__(name=name, parent_name=parent_name, op_type="quantize")
        self.N = N
        self.I_info = _normalize_tensor_info(I_info)
        self.out_info = _normalize_tensor_info(out_info)


def default_tensor_info(dtype: str = DEFAULT_DTYPE, loc: str = DEFAULT_LOC) -> Dict[str, Any]:
    return {
        "dtype": dtype,
        "loc": loc,
        "quantization_config": None,
        "sparsity_info": {
            "input_sparsity": {
                "s": 0.0,
                "batched_sparsity_fn": None,
            }
        },
    }


def _normalize_tensor_info(info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    base = default_tensor_info()
    if info is None:
        return base

    merged = dict(base)
    merged.update(info)

    if merged.get("quantization_config") is None:
        merged["quantization_config"] = None

    sparsity_info = merged.get("sparsity_info") or {}
    input_sparsity = sparsity_info.get("input_sparsity") or sparsity_info.get("input sparsity") or {}
    sparsity_value = _normalize_sparsity_value(input_sparsity.get("s", 0.0))
    sparsity_info["input_sparsity"] = {
        "s": sparsity_value,
        "batched_sparsity_fn": input_sparsity.get("batched_sparsity_fn"),
        "mode": input_sparsity.get("mode", "scalar" if isinstance(sparsity_value, float) else "profile"),
    }
    merged["sparsity_info"] = sparsity_info
    return merged


def _normalize_sparsity_value(value: Any) -> float | list[float]:
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return float(value)


Operations = OP
TensorInfo = Dict[str, Any]
BatchedSparsityFn = Optional[Callable[[int], float]]
