"""Write one raw 12-lead recording as a CSV file for the code-execution sandbox."""
import numpy as np

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def record_csv(signal, sampling_rate, decimals=3):
    """CSV text: t_ms and the 12 leads in mV, one row per sample. NaNs (rare in PTB-XL) become 0."""
    arr = np.nan_to_num(np.asarray(signal, dtype=float))
    if arr.ndim != 2 or arr.shape[1] != len(LEADS):
        raise ValueError(f"expected an (n, 12) signal, got {arr.shape}")
    t_ms = np.arange(arr.shape[0]) * 1000.0 / sampling_rate
    fmt = f"%.{decimals}f"
    rows = [",".join(("t_ms",) + LEADS)]
    for t, values in zip(t_ms, arr):
        rows.append(f"{t:g}," + ",".join(fmt % v for v in values))
    return "\n".join(rows) + "\n"
