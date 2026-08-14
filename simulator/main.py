from __future__ import annotations

import argparse
from typing import Any, Dict, List
import time

from model_runner import (
    CATEGORY_ORDER,
    DEFAULT_ATTENTION_CONFIG,
    available_methods,
    build_model_workload,
    run_method_workload,
    validate_workload,
)

# python main.py --model-name llama_2_7b --methods ant --input-length 0 --output-length 1
# python main.py --model-name llama_2_7b --methods vortex --quant-scheme "AQLM|CQ" --input-length 0 --output-length 1
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one Vortex/baseline simulation.")
    parser.add_argument("--model-name", default="llama-2-7b", help="Model name from llm_configs.")
    parser.add_argument("--methods", nargs="+", default=["vortex"], help="One or more methods, or 'all'.")
    parser.add_argument("--input-length", type=int, default=8, help="Prefill input length.")
    parser.add_argument("--output-length", type=int, default=1, help="Decode output length.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--default-dtype", default="fp16", help="Default tensor dtype.")
    parser.add_argument("--compute-mode", choices=["lut_based", "dequant"], default="lut_based", help="Vortex compute mode.")
    parser.add_argument("--processed-sparsity", type=str, default=None, help="Override processed sparsity.")
    parser.add_argument("--quant-scheme", type=str, default="", help="Quantization scheme to apply, e.g. 'AQLM|CQ'.")
    # force-flow: str ["MUF", "LUF", "auto"], default="auto", help="Force a specific dataflow. Only for testing."
    parser.add_argument("--force-dataflow", choices=["MUF", "LUF", None], default=None, help="Force a specific dataflow. Only for testing.")
    return parser.parse_args()


def resolve_methods(raw_methods: List[str]) -> List[str]:
    methods: List[str] = []
    for item in raw_methods:
        methods.extend(part.strip() for part in item.split(",") if part.strip())
    if "all" in {method.lower() for method in methods}:
        return available_methods()
    return methods


def build_attention_config(args: argparse.Namespace) -> Dict[str, Any]:
    attention_config = {
        # "quantize_kv": bool(args.quantize_kv),
        "quantize_kv": bool(getattr(args, "quantize_kv", False)),  # default to False if not present
        "kv_cache_loc": DEFAULT_ATTENTION_CONFIG["kv_cache_loc"],
        # "cq_config": dict(DEFAULT_ATTENTION_CONFIG["cq_config"]),
    }
    return attention_config


def build_runtime_args(args: argparse.Namespace) -> Dict[str, Any]:
    runtime_args: Dict[str, Any] = {"compute_mode": args.compute_mode}
    if args.force_dataflow is not None:
        runtime_args["force_dataflow"] = args.force_dataflow
    return runtime_args


def print_workload_summary(workload_check: Dict[str, Any], methods: List[str], args: argparse.Namespace) -> None:
    print(f"model={args.model_name} methods={','.join(methods)}")
    print(
        "workload",
        f"input_length={args.input_length}",
        f"output_length={args.output_length}",
        f"prefill_ops={workload_check['prefill_ops']}",
        f"decode_ops={workload_check['decode_ops']}",
        f"sanity={'passed' if workload_check['passed'] else 'failed'}",
    )
    for check in workload_check["checks"]:
        print(f"  [{ 'ok' if check['passed'] else '!!' }] {check['name']}: {check['detail']}")


def print_method_summary(result: Dict[str, Any]) -> None:
    print(f"\nmethod={result['method']} sanity={'passed' if result['sanity']['passed'] else 'failed'}")
    for phase_name in ["prefill", "decode", "total"]:
        phase = result[phase_name]
        print(
            f"  {phase_name:7s}",
            f"impl={phase['phase_method']}",
            f"cycles={phase['total_cycles']}",
            f"latency(ms)={phase['total_cycles']/500e6 * 1e3}",
            f"energy={phase['total_energy']:.6e}J",
            f"power(total/core/sram/dram)="
            f"{phase['total_power']:.6e}/"
            f"{phase['core_power']:.6e}/"
            f"{phase['sram_power']:.6e}/"
            f"{phase['dram_power']:.6e}W",
            f"area={phase['area_mm2']:.6f}mm^2",
        )
        for category in CATEGORY_ORDER:
            category_stats = phase["categories"][category]
            print(
                f"    {category:13s}",
                f"cycles={category_stats['total_cycles']}",
                f"energy={category_stats['total_energy']:.6e}J",
                f"power(total/core/sram/dram)="
                f"{category_stats['total_power']:.6e}/"
                f"{category_stats['core_power']:.6e}/"
                f"{category_stats['sram_power']:.6e}/"
                f"{category_stats['dram_power']:.6e}W",
            )
    for check in result["sanity"]["checks"]:
        print(f"  [{ 'ok' if check['passed'] else '!!' }] {check['name']}: {check['detail']}")

def run_simulation(
    model_name="llama-2-7b",
    methods=("vortex",),
    input_length=8,
    output_length=1,
    batch_size=1,
    default_dtype="fp16",
    compute_mode="lut_based",
    processed_sparsity=None, # e.g. "RVQ_0.3"
    quant_scheme="", # e.g. "AQLM|CQ"
    force_dataflow=None,
):
    # Pack the arguments into an object so the CLI path and the programmatic
    # path below share one code path.
    class Args:
        pass

    args = Args()
    args.model_name = model_name
    args.methods = list(methods)
    args.input_length = input_length
    args.output_length = output_length
    args.batch_size = batch_size
    args.default_dtype = default_dtype
    args.compute_mode = compute_mode
    args.processed_sparsity = processed_sparsity
    args.force_dataflow = force_dataflow
    args.quant_scheme = quant_scheme

    methods = resolve_methods(args.methods)
    assert len(methods) == 1, "Only one method can be tested at a time in this example."

    unknown_methods = [m for m in methods if m not in available_methods()]
    if unknown_methods:
        raise ValueError(f"Unknown methods: {unknown_methods}. Available: {available_methods()}")

    attention_config = build_attention_config(args)
    runtime_args = build_runtime_args(args)

    sparsity_config = {}
    if args.processed_sparsity is not None:
        sparsity_method = args.processed_sparsity.split("_")[0]
        sparsity_value = float(args.processed_sparsity.split("_")[1])
        assert sparsity_method in ["RVQ"], "Only RVQ sparsity method is supported in this example."
        if sparsity_method == "RVQ":
            sparsity_config={
                "method": "RVQ",
                "model_name": model_name,
                "contextual_sparsity": sparsity_value,
            }

    st_time = time.time()
    workload = build_model_workload(
        model_name=args.model_name,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
        default_dtype=args.default_dtype,
        attention_config=attention_config,
        sparsity_config=sparsity_config,
    )
    print(f"{time.time() - st_time} seconds for build_model_workload")

    if quant_scheme:
        # split "AQLM|CQ" into list of schemes
        schemes = [scheme for scheme in quant_scheme.split("|")]
        from quantization import apply_quant_methods
        for scheme in schemes:
            if scheme not in ["CQ"]:
                apply_quant_methods(workload.prefill_ops, [scheme])
            apply_quant_methods(workload.decode_ops, [scheme])

    workload_check = validate_workload(workload)

    results = {}
    for method in methods:
        result = run_method_workload(workload, method, args=runtime_args)
        results[method] = result

    return results, args

def main():
    args = parse_args()
    return run_simulation(
        model_name=args.model_name,
        methods=args.methods,
        input_length=args.input_length,
        output_length=args.output_length,
        batch_size=args.batch_size,
        default_dtype=args.default_dtype,
        compute_mode=args.compute_mode,
        processed_sparsity=args.processed_sparsity,
        quant_scheme=args.quant_scheme,
        force_dataflow=args.force_dataflow,
    )


if __name__ == "__main__":
    results, args = main()
    print(results)
    for result in results.values():
        print_method_summary(result)
