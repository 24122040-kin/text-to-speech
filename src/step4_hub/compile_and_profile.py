#!/usr/bin/env python3
"""
Compile quantized Piper model to QNN binary for Qualcomm AI Hub device.

Critical settings:
1. Device must be IQ-9075 EVK or later (Hexagon v73+)
2. Must include --quantize_io flag (required by HTP for int8/int16 models)
3. Must use qnn_context_binary runtime
4. Compile job is separate from quantize job (not same step)

After compilation:
- Profile latency/power on target device
- Run inference to get actual hardware outputs
"""

import json
import logging
import argparse
from pathlib import Path
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_available_devices() -> list:
    """List available Qualcomm AI Hub devices."""
    try:
        import qai_hub as hub
    except ImportError:
        logger.error("qai-hub not installed")
        return []
    
    logger.info("Fetching available devices from Qualcomm AI Hub...")
    
    try:
        devices = hub.get_devices()
        device_names = [d.name for d in devices]
        logger.info(f"Available devices: {device_names}")
        return device_names
    except Exception as e:
        logger.error(f"Failed to fetch devices: {e}")
        return []


def submit_compile_job(
    model_path: Path,
    device_name: str,
    output_dir: Path,
    job_name: str = "piper_vi_w8a16_compile"
) -> dict:
    """
    Submit compilation job for quantized model.
    
    Key flags:
    - --target_runtime qnn_context_binary
    - --quantize_io (REQUIRED for int8/int16 HTP models)
    """
    try:
        import qai_hub as hub
    except ImportError:
        logger.error("qai-hub not installed")
        raise
    
    logger.info(f"Submitting compilation job")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Device: {device_name}")
    logger.info(f"  Options: --quantize_io (required for int8/int16)")
    
    # Build compile options
    compile_options = "--target_runtime qnn_context_binary --quantize_io"
    
    logger.info(f"  Full options: {compile_options}")
    
    try:
        # Get device object
        device = hub.Device(device_name)
        
        # Submit compile job
        compile_job = hub.submit_compile_job(
            model=str(model_path),
            device=device,
            options=compile_options,
            name=job_name,
        )
        
        job_id = compile_job.job_id
        logger.info(f"✅ Compilation job submitted: {job_id}")
        
        return {
            "job_id": job_id,
            "job_type": "compile",
            "model": str(model_path),
            "device": device_name,
            "target_runtime": "qnn_context_binary",
            "quantize_io": True,
            "status": "submitted",
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to submit compilation job: {e}")
        logger.error(f"Ensure qai-hub is configured: qai-hub configure --api_token <token>")
        raise


def submit_profile_job(
    model_path: Path,
    device_name: str,
    output_dir: Path,
    job_name: str = "piper_vi_profile"
) -> dict:
    """
    Submit profiling job to measure latency and power on device.
    """
    try:
        import qai_hub as hub
    except ImportError:
        logger.error("qai-hub not installed")
        raise
    
    logger.info(f"Submitting profiling job for {model_path}")
    
    try:
        device = hub.Device(device_name)
        
        profile_job = hub.submit_profile_job(
            model=str(model_path),
            device=device,
            name=job_name,
        )
        
        job_id = profile_job.job_id
        logger.info(f"✅ Profile job submitted: {job_id}")
        
        return {
            "job_id": job_id,
            "job_type": "profile",
            "model": str(model_path),
            "device": device_name,
            "status": "submitted",
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to submit profile job: {e}")
        raise


def submit_inference_job(
    model_path: Path,
    device_name: str,
    test_inputs: dict,
    output_dir: Path,
    job_name: str = "piper_vi_inference"
) -> dict:
    """
    Submit inference job to get actual hardware outputs.
    
    test_inputs should be different from calibration data to avoid overly optimistic results.
    """
    try:
        import qai_hub as hub
    except ImportError:
        logger.error("qai-hub not installed")
        raise
    
    logger.info(f"Submitting inference job")
    logger.info(f"  Model: {model_path}")
    logger.info(f"  Device: {device_name}")
    
    try:
        device = hub.Device(device_name)
        
        inference_job = hub.submit_inference_job(
            model=str(model_path),
            device=device,
            inputs=test_inputs,
            name=job_name,
        )
        
        job_id = inference_job.job_id
        logger.info(f"✅ Inference job submitted: {job_id}")
        
        return {
            "job_id": job_id,
            "job_type": "inference",
            "model": str(model_path),
            "device": device_name,
            "num_test_samples": len(test_inputs),
            "status": "submitted",
            "timestamp": time.time(),
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to submit inference job: {e}")
        raise


def save_job_log(output_dir: Path, job_data: dict):
    """Append job to log file (append mode, not overwrite)."""
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
    
    logger.info(f"Saved job to log: {log_path}")


def load_test_inputs(input_dir: Path) -> dict:
    """Load test inputs from calibration data (with different samples if possible)."""
    import numpy as np
    
    # For now, use calibration data; in real scenario, use separate test set
    calib_path = input_dir / 'calibration_inputs.npz'
    
    if not calib_path.exists():
        logger.warning(f"Calibration data not found: {calib_path}")
        logger.warning("Using empty test inputs")
        return {}
    
    data = np.load(str(calib_path))
    test_array = data['calibration_array']
    
    # Take first few samples for inference
    num_test = min(5, test_array.shape[0])
    test_subset = test_array[:num_test]
    
    logger.info(f"Using {num_test} test samples from calibration data")
    
    # Return as dict for qai_hub API
    return {'input': test_subset}


def main():
    parser = argparse.ArgumentParser(
        description="Compile and profile Piper quantized model on Qualcomm AI Hub"
    )
    parser.add_argument('--model', type=Path, required=True,
                        help='Quantized ONNX model')
    parser.add_argument('--device', default='IQ-9075 EVK',
                        help='Target device (must support Hexagon v73+ for w8a16)')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/piper_vi'),
                        help='Output directory')
    parser.add_argument('--list_devices', action='store_true',
                        help='List available devices and exit')
    parser.add_argument('--skip_profile', action='store_true',
                        help='Skip profiling (faster for testing)')
    parser.add_argument('--skip_inference', action='store_true',
                        help='Skip inference (faster for testing)')
    
    args = parser.parse_args()
    
    # List devices if requested
    if args.list_devices:
        devices = get_available_devices()
        if devices:
            logger.info("\nAvailable devices:")
            for i, dev in enumerate(devices, 1):
                logger.info(f"  {i}. {dev}")
        return
    
    # Validate model
    if not args.model.exists():
        logger.error(f"Model not found: {args.model}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    jobs = []
    
    # 1. Compile
    try:
        compile_job_data = submit_compile_job(
            args.model,
            args.device,
            args.output_dir
        )
        save_job_log(args.output_dir, compile_job_data)
        jobs.append(compile_job_data)
    except Exception as e:
        logger.error(f"Compilation failed: {e}")
        sys.exit(1)
    
    # 2. Profile (optional)
    if not args.skip_profile:
        try:
            profile_job_data = submit_profile_job(
                args.model,
                args.device,
                args.output_dir
            )
            save_job_log(args.output_dir, profile_job_data)
            jobs.append(profile_job_data)
        except Exception as e:
            logger.error(f"Profiling failed: {e}")
    
    # 3. Inference (optional)
    if not args.skip_inference:
        try:
            test_inputs = load_test_inputs(args.output_dir)
            if test_inputs:
                inference_job_data = submit_inference_job(
                    args.model,
                    args.device,
                    test_inputs,
                    args.output_dir
                )
                save_job_log(args.output_dir, inference_job_data)
                jobs.append(inference_job_data)
            else:
                logger.warning("No test inputs available, skipping inference")
        except Exception as e:
            logger.error(f"Inference submission failed: {e}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info(f"COMPILATION/PROFILING JOBS SUBMITTED")
    for job in jobs:
        logger.info(f"  {job['job_type'].upper()}: {job['job_id']}")
    logger.info(f"  Log: {args.output_dir / 'job_log.json'}")
    logger.info(f"{'='*60}\n")


if __name__ == '__main__':
    main()
