"""Helpers shared by the hardware (B*) plot scripts."""
import numpy as np
import pandas as pd

# The columns that identify one simulated configuration. Used for
# de-duplication; must match helper.log_results_to_csv's schema.
CONFIG_KEY = [
    "model_name", "input_length", "output_length",
    "batch_size", "method", "quant_scheme",
    "processed_sparsity", "force_dataflow",
]

MODEL_NAME_MAP = {
    "llama-2-7b": "Llama2-7B",
    "mistral_7b": "Mistral-7B",
    "llama_2_13b": "Llama2-13B",
}


def load_latest(csv_path):
    """Read a results CSV, keeping only the newest row per configuration.

    The Vortex sweeps historically appended without de-duplicating, so a CSV
    could hold several generations of the same configuration. run_simulation.sh
    no longer produces duplicates, but hand-run data and the reference CSVs still
    can, and silently averaging stale rows with fresh ones is the exact failure
    this guards against.
    """
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    key = [c for c in CONFIG_KEY if c in df.columns]
    return df.sort_values("timestamp").drop_duplicates(subset=key, keep="last")


def geo_mean(values):
    vals = [v for v in values if v is not None and v > 0 and not np.isnan(v)]
    return float(np.exp(np.mean(np.log(vals)))) if vals else np.nan


def model_label(name, mapping=None):
    mapping = MODEL_NAME_MAP if mapping is None else mapping
    return mapping.get(name, name)


def require_rows(df, what):
    """Fail loudly on an empty selection.

    Worth the four lines: the naming migration (ccarray -> vortex, CtL/LtC ->
    MUF/LUF) turns a stale filter into an *empty* dataframe, not an error, and
    matplotlib will happily render a blank chart.
    """
    if len(df) == 0:
        raise SystemExit(
            f"No rows matched for {what}.\n"
            f"  Most likely the input CSV predates the vortex/MUF/LUF rename,\n"
            f"  or the simulation group that produces it has not been run.")
    return df
