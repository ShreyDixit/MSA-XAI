#!/usr/bin/env python3
"""Run quantify.py with an equivalent float64 CUDA spatial-shuffle backend.

All arguments are forwarded to the main pipeline. Its CPU implementation remains
the reference. Only the permutation gather/reduction is replaced, with the same
NumPy-generated masks and permutation banks. A separate backend manifest records
this entry point and the measured difference from the CPU implementation.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import quantify
from quantification.metrics import shuffled_capture as cpu_shuffled_capture
from quantification.storage import sha256


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=quantify.BASE / "quantification/results")
    args, _ = parser.parse_known_args()
    if not args.device.startswith("cuda"):
        raise ValueError("Use quantify.py directly for a CPU-only run.")
    banks = {}

    def gpu_shuffled_capture(change, mask, permutations):
        total = np.sum(change, dtype=np.float64)
        if total == 0:
            return np.full(len(permutations), np.nan)
        key = id(permutations)
        if key not in banks:
            # Keep a strong reference to prevent object-id reuse.
            banks[key] = (permutations, torch.as_tensor(permutations, dtype=torch.long, device=args.device))
        bank = banks[key][1]
        selected = torch.as_tensor(np.flatnonzero(mask), dtype=torch.long, device=args.device)
        values = torch.as_tensor(change.ravel(), dtype=torch.float64, device=args.device)
        with torch.inference_mode():
            return (values[bank[:, selected]].sum(dim=1) / total).cpu().numpy()

    # Validate real-sized inputs and zero-change handling before replacing the backend.
    rng = np.random.default_rng(2810)
    change = rng.random((128, 128))
    mask = quantify.top_mask(rng.random((128, 128)), .2, rng)
    permutations = np.array([rng.permutation(16384) for _ in range(100)], dtype=np.int32)
    cpu = cpu_shuffled_capture(change, mask, permutations)
    gpu = gpu_shuffled_capture(change, mask, permutations)
    error = float(np.max(np.abs(cpu - gpu)))
    np.testing.assert_allclose(gpu, cpu, rtol=1e-12, atol=1e-12)
    assert np.isnan(gpu_shuffled_capture(np.zeros_like(change), mask, permutations)).all()
    banks.clear()
    quantify.log(f"CUDA shuffle backend matches CPU: max error {error:.3g}")
    quantify.shuffled_capture = gpu_shuffled_capture
    quantify.main()
    provenance = {"backend": "float64 CUDA gather and sum; same NumPy permutation banks as CPU reference",
                  "script_sha256": sha256(__file__), "reference_metrics_sha256": sha256(quantify.BASE / "quantification/metrics.py"),
                  "device": args.device, "device_name": torch.cuda.get_device_name(args.device),
                  "real_size_cpu_comparison_max_absolute_error": error,
                  "validation_tolerance": 1e-12}
    quantify.write_json(args.output / "spatial_control_backend.json", provenance)


if __name__ == "__main__":
    main()
