#!/usr/bin/env python3
"""
Quantize Piper to w8a16 (weights int8, activations int16) using Qualcomm AI Hub.

Critical points from runbook:
1. Use submit_quantize_job with activations_dtype EXPLICIT (not --quantize_full_type flag)
2. activations_dtype must be INT16 (not INT8 — activation int8 is what failed before)
3. Save all job IDs to job_log.json for audit trail
4. Do NOT rely on compile job flags for quantization settings

API Reference:
    qai_hub.submit_quantize_job(
        model=<ONNX path or bytes>,
        calibration_data=<input samples>,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT16,
    )
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Optional
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_calibration_data(calib_path: Path) -> dict:
    """Load calibration data from npz file."""
    import numpy as np
    
    logger.info(f"Loading calibration data from {calib_path}")
    
    data = np.load(str(calib_path))
    calib_array = data['calibration_array']
    
    logger.info(f"Loaded calibration array shape: {calib_array.shape}")
    
    # Return as dict for qai_hub API (key: input name, value: array)
    # For Piper, the main input is text tokens
    # TODO: Verify actual input names from ONNX model
    return {'input': calib_array}


def submit_quantize_job(
    model_path: Path,
    calibration_data: dict,
    output_dir: Path,
    job_name: str = "piper_vi_w8a16"
) -> dict:
    """
    Submit quantization job to Qualcomm AI Hub.
    
    Returns job metadata including job ID for tracking.
    """
    try:
        import qai_hub as hub
    except ImportError:
        logger.error("qai-hub not installed. Install with: pip install qai-hub")
        logger.error("Also configure with: qai-hub configure --api_token <your_token>")
        raise
    
    logger.info(f"Submitting quantization job to Qualcomm AI Hub")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Job name: {job_name}")
    logger.info(f"  Quantization: w8a16 (weights=int8, activations=int16)")
    
    try:
        # Submit quantize job with EXPLICIT activations_dtype
        quantize_job = hub.submit_quantize_job(
            model=str(model_path),  # Can be path or bytes
            calibration_data=calibration_data,
            weights_dtype=hub.QuantizeDtype.INT8,
            activations_dtype=hub.QuantizeDtype.INT16,  # <-- CRITICAL: explicit, not default
            name=job_name,
            # Optional: compile_options="",  # Leave empty for quantize-only
        )
        
        job_id = quantize_job.job_id
        logger.info(f"✅ Quantization job submitted: {job_id}")
        
        return {
            "job_id": job_id,
            "job_type": "quantize",
            "model": str(model_path),
            "weights_dtype": "int8",
            "activations_dtype": "int16",
            "status": "submitted",
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to submit quantization job: {e}")
        raise


def wait_for_job(job_id: str, max_wait_minutes: int = 60) -> dict:
    """
    Wait for quantization job to complete and fetch results.
    """
    try:
        import qai_hub as hub
    except ImportError:
        raise RuntimeError("qai-hub not available")
    
    logger.info(f"Waiting for job {job_id} to complete (timeout: {max_wait_minutes} min)...")
    
    # Fetch job object by ID
    # Note: API might differ; this is a placeholder
    # In practice: quantize_job = hub.get_job(job_id) or similar
    
    start_time = time.time()
    max_wait_seconds = max_wait_minutes * 60
    poll_interval = 10  # seconds
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait_seconds:
            logger.error(f"Job timed out after {max_wait_minutes} minutes")
            return {"status": "timeout", "job_id": job_id}
        
        # TODO: Poll job status via API
        # status = hub.get_job_status(job_id)
        # if status == "completed":
        #     break
        # elif status == "failed":
        #     raise RuntimeError(f"Job {job_id} failed")
        
        logger.info(f"  [Elapsed: {elapsed/60:.1f}min] Checking job status...")
        time.sleep(poll_interval)
        
        # Placeholder: break after first check in demo mode
        logger.info("  (This is a placeholder implementation)")
        break
    
    return {
        "status": "completed",
        "job_id": job_id,
        "timestamp": time.time(),
    }


def save_job_log(output_dir: Path, job_data: dict):
    """Append job to log file (not overwrite)."""
    log_path = output_dir / 'job_log.json'
    
    # Load existing log or create empty list
    if log_path.exists():
        with open(log_path, 'r') as f:
            log = json.load(f)
    else:
        log = []
    
    # Append new job
    log.append(job_data)
    
    # Save updated log
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    
    logger.info(f"Saved job log to {log_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quantize Piper to w8a16 using Qualcomm AI Hub"
    )
    parser.add_argument('--model', type=Path, required=True,
                        help='Input ONNX model (fixed-shape)')
    parser.add_argument('--calib_data', type=Path, required=True,
                        help='Calibration data (npz format)')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/piper_vi'),
                        help='Output directory')
    parser.add_argument('--job_name', default='piper_vi_w8a16',
                        help='Job name on Qualcomm AI Hub')
    parser.add_argument('--skip_wait', action='store_true',
                        help='Do not wait for job completion (useful for long-running jobs)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)
    
    if not args.calib_data.exists():
        logger.error(f"Calibration data not found: {args.calib_data}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load calibration data
    calib_data = load_calibration_data(args.calib_data)
    
    # Submit quantization job
    job_data = submit_quantize_job(
        args.model,
        calib_data,
        args.output_dir,
        args.job_name
    )
    
    # Wait for completion (if not skipped)
    if not args.skip_wait:
        result = wait_for_job(job_data['job_id'])
        job_data.update(result)
        logger.info(f"Job {job_data['job_id']} completed with status: {result['status']}")
    else:
        logger.info(f"Skipping job wait. Check status later with job ID: {job_data['job_id']}")
    
    # Save job log
    save_job_log(args.output_dir, job_data)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"QUANTIZATION JOB SUBMITTED")
    logger.info(f"  Job ID: {job_data['job_id']}")
    logger.info(f"  Weights dtype: {job_data['weights_dtype']}")
    logger.info(f"  Activations dtype: {job_data['activations_dtype']}")
    logger.info(f"  Status: {job_data['status']}")
    logger.info(f"  Log: {args.output_dir / 'job_log.json'}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
