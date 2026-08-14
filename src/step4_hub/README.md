# Step 4 Hub — Piper (vi) Quantization & Deployment to Qualcomm AI Hub

This directory contains scripts to quantize and deploy Piper (Vietnamese) text-to-speech model to Qualcomm AI Hub.

**Target device:** IQ-9075 EVK (Hexagon v73+, supports w8a16 quantization)

**Strategy:** 
1. Export Piper ONNX with fixed shape (required by HTP compiler)
2. Prepare calibration data (≥200 Vietnamese sentences)
3. Quantize to w8a16 using `submit_quantize_job`
4. Compile for target device with `--quantize_io` flag
5. Run inference on hardware and verify against fp32 reference
6. Generate comprehensive report

## Scripts

- `export_piper_onnx.py` — Export Piper to fixed-shape ONNX, verify against original
- `make_calibration_data.py` — Prepare calibration dataset from corpus
- `quantize_piper_w8a16.py` — Submit quantization job to Qualcomm AI Hub
- `compile_and_profile.py` — Compile and profile on target device
- `verify_piper_w8a16.py` — Verify quantized model against fp32 reference
- `generate_report.py` — Generate final deployment report

## Output structure

```
outputs/piper_vi/
├── piper_vi_fp32_fixed_shape.onnx     # Fixed-shape ONNX (batchsize=1)
├── piper_vi_int8_w8a16.onnx           # Quantized model
├── piper_vi_int8_w8a16.bin            # QNN context binary (compiled)
├── calibration_inputs.npz             # Calibration data
├── job_log.json                       # All job IDs (quantize/compile/profile/inference)
├── hw_outputs.npz                     # Inference outputs from hardware
├── verify_results.json                # Per-output cosine similarity scores
├── compile_error.log                  # (if needed) Detailed compilation errors
└── REPORT.md                          # Final deployment report
```

## Usage

```bash
# 1. Export Piper
python export_piper_onnx.py --voice vi_VN-vais1000-medium --output_dir outputs/piper_vi

# 2. Prepare calibration data
python make_calibration_data.py --input_manifest data/mt/manifest.json \
  --output_dir outputs/piper_vi --num_samples 200

# 3. Quantize (requires qai-hub configured with API token)
python quantize_piper_w8a16.py --model outputs/piper_vi/piper_vi_fp32_fixed_shape.onnx \
  --calib_data outputs/piper_vi/calibration_inputs.npz \
  --output_dir outputs/piper_vi

# 4. Compile and profile
python compile_and_profile.py --model outputs/piper_vi/piper_vi_int8_w8a16.onnx \
  --device "IQ-9075 EVK" --output_dir outputs/piper_vi

# 5. Verify
python verify_piper_w8a16.py --fp32_model outputs/piper_vi/piper_vi_fp32_fixed_shape.onnx \
  --hw_outputs outputs/piper_vi/hw_outputs.npz \
  --output_dir outputs/piper_vi

# 6. Generate report
python generate_report.py --job_log outputs/piper_vi/job_log.json \
  --verify_results outputs/piper_vi/verify_results.json \
  --output_file outputs/piper_vi/REPORT.md
```

## Key points from runbook

- **w8a16 = weights int8, activations int16** (NOT w8a8)
- **Use `submit_quantize_job` with `activations_dtype` explicit** (don't rely on compile job flags)
- **Include `--quantize_io` in compile options** (required by HTP)
- **Verify cosine similarity ≥ 0.95** against fp32 reference
- **If verify fails, use per-layer bisect** (not whole model)
- **Log every job ID** for audit trail
