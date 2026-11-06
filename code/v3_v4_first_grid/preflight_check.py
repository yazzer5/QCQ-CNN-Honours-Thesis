#!/usr/bin/env python
"""
preflight_check.py — Run this LOCALLY before uploading to Gadi.

Builds all 4 models (M1-M4), verifies the hard-coded parameter-count
asserts pass, and runs one dummy forward pass per model on random data
to catch shape/API errors before burning GPU-hours on Gadi.

Usage: python preflight_check.py
"""
import torch
from train_gadi import build_model, EXPECTED_PARAMS

def main():
    x = torch.rand(1, 1, 28, 28)   # batch=1, matches mandatory quantum batch size
    for model_name in ["M1", "M2", "M3", "M4"]:
        print(f"\n--- Building {model_name} ---")
        model = build_model(model_name, seed=42)
        total, (fe, bb, be) = model.total_params()
        assert total == EXPECTED_PARAMS[model_name]
        print(f"Running dummy forward pass for {model_name} (this is slow for "
              f"quantum front-ends since they run gate-by-gate per patch)...")
        with torch.no_grad():
            out = model(x)
        print(f"{model_name} output shape: {out.shape} (expect [1, 2])")
        assert out.shape == (1, 2), f"{model_name} bad output shape {out.shape}"
        print(f"{model_name} OK — front={fe} backbone={bb} back_end={be} total={total}")

    print("\nALL 4 MODELS PASSED PRE-FLIGHT CHECK. Safe to upload to Gadi.")

if __name__ == "__main__":
    main()
