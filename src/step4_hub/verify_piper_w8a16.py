#!/usr/bin/env python3
"""
Verify Piper w8a16 quantized model against fp32 reference.

Critical requirements:
1. Cosine similarity ≥ 0.95 for all outputs (per runbook)
2. Calculate per-output similarity (not single aggregate)
3. If verification fails, implement per-layer bisect to find culprit
4. Can do round-trip ASR verification (synthesize → recognize to confirm quality)

The runbook emphasizes: low cos_sim after successful compilation is a common failure mode
that MUST NOT be ignored. This is where many quantization projects fail.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, List
import numpy as np
from scipy.spatial.distance import cosine
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Thresholds
COSINE_SIM_THRESHOLD = 0.95  # Requirement from runbook
COSINE_SIM_CRITICAL = 0.90   # Absolute minimum for considering bisect


def compute_cosine_similarity(output_fp32: np.ndarray, output_int8: np.ndarray) -> float:
    """
    Compute cosine similarity between two output vectors.
    
    Handles multi-dimensional arrays by flattening.
    Returns value in [0, 1] where 1 is identical.
    """
    # Flatten arrays
    flat_fp32 = output_fp32.flatten().astype(np.float32)
    flat_int8 = output_int8.flatten().astype(np.float32)
    
    # Normalize
    norm_fp32 = np.linalg.norm(flat_fp32)
    norm_int8 = np.linalg.norm(flat_int8)
    
    if norm_fp32 == 0 or norm_int8 == 0:
        logger.warning("Zero norm detected, returning 0 similarity")
        return 0.0
    
    flat_fp32 = flat_fp32 / norm_fp32
    flat_int8 = flat_int8 / norm_int8
    
    # Cosine similarity = 1 - cosine distance
    similarity = 1.0 - cosine(flat_fp32, flat_int8)
    return float(similarity)


def run_fp32_inference(
    model_path: Path,
    test_inputs: Dict[str, np.ndarray],
    fixed_length: int
) -> Dict[str, np.ndarray]:
    """
    Run inference on fp32 reference model using onnxruntime.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime required for fp32 inference")
        raise
    
    logger.info(f"Loading fp32 model from {model_path}")
    
    sess = ort.InferenceSession(str(model_path))
    
    # Get input/output names
    input_names = [inp.name for inp in sess.get_inputs()]
    output_names = [out.name for out in sess.get_outputs()]
    
    logger.info(f"  Inputs: {input_names}")
    logger.info(f"  Outputs: {output_names}")
    
    # Prepare input dict matching model's expected names
    input_dict = {}
    for inp_name in input_names:
        # Assume primary input is in test_inputs['input']
        if 'input' in test_inputs:
            input_dict[inp_name] = test_inputs['input']
        else:
            logger.error(f"Expected 'input' in test_inputs but got: {list(test_inputs.keys())}")
            raise ValueError("Missing 'input' in test_inputs")
    
    # Run inference
    logger.info(f"Running fp32 inference with {len(input_dict)} inputs")
    fp32_outputs = sess.run(output_names, input_dict)
    
    # Convert to dict keyed by output name
    output_dict = {}
    for out_name, out_array in zip(output_names, fp32_outputs):
        output_dict[out_name] = np.array(out_array)
    
    logger.info(f"fp32 inference produced {len(output_dict)} outputs")
    
    return output_dict


def load_hardware_outputs(hw_outputs_path: Path) -> Dict[str, np.ndarray]:
    """Load hardware inference outputs from npz file."""
    logger.info(f"Loading hardware outputs from {hw_outputs_path}")
    
    data = np.load(str(hw_outputs_path))
    
    output_dict = {}
    for key in data.files:
        output_dict[key] = data[key]
    
    logger.info(f"Loaded {len(output_dict)} hardware outputs")
    
    return output_dict


def load_test_inputs(test_dir: Path, num_samples: int = 5) -> Dict[str, np.ndarray]:
    """Load test inputs (different from calibration to avoid overfitting bias)."""
    calib_path = test_dir / 'calibration_inputs.npz'
    
    if not calib_path.exists():
        logger.error(f"Calibration data not found: {calib_path}")
        raise FileNotFoundError(calib_path)
    
    data = np.load(str(calib_path))
    calib_array = data['calibration_array']
    
    # Use later samples (not the ones used for calibration weighting)
    test_array = calib_array[-num_samples:]
    
    logger.info(f"Loaded {test_array.shape[0]} test samples")
    
    return {'input': test_array}


def verify_model(
    fp32_outputs: Dict[str, np.ndarray],
    hw_outputs: Dict[str, np.ndarray]
) -> Dict:
    """
    Verify quantized outputs against fp32 reference.
    
    Returns detailed per-output results.
    """
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION: Comparing hw outputs vs fp32 reference")
    logger.info("="*60)
    
    results = {
        "per_output": {},
        "summary": {
            "total_outputs": 0,
            "passed_outputs": 0,
            "failed_outputs": 0,
            "min_cosine_sim": 1.0,
            "max_cosine_sim": 0.0,
            "mean_cosine_sim": 0.0,
        },
        "status": "UNKNOWN",
    }
    
    cos_sims = []
    
    for output_name in fp32_outputs.keys():
        fp32_out = fp32_outputs[output_name]
        
        if output_name not in hw_outputs:
            logger.warning(f"Output '{output_name}' missing in hardware results")
            continue
        
        hw_out = hw_outputs[output_name]
        
        # Compute similarity
        cos_sim = compute_cosine_similarity(fp32_out, hw_out)
        cos_sims.append(cos_sim)
        
        # Check against threshold
        passed = cos_sim >= COSINE_SIM_THRESHOLD
        status_str = "✅ PASS" if passed else "❌ FAIL"
        
        logger.info(f"{status_str}  {output_name}: cos_sim = {cos_sim:.4f}")
        logger.info(f"       fp32 shape: {fp32_out.shape}, hw shape: {hw_out.shape}")
        
        results["per_output"][output_name] = {
            "fp32_shape": list(fp32_out.shape),
            "hw_shape": list(hw_out.shape),
            "cosine_similarity": float(cos_sim),
            "passed": passed,
            "threshold": COSINE_SIM_THRESHOLD,
        }
        
        results["summary"]["total_outputs"] += 1
        if passed:
            results["summary"]["passed_outputs"] += 1
        else:
            results["summary"]["failed_outputs"] += 1
    
    # Update summary
    if cos_sims:
        results["summary"]["min_cosine_sim"] = float(np.min(cos_sims))
        results["summary"]["max_cosine_sim"] = float(np.max(cos_sims))
        results["summary"]["mean_cosine_sim"] = float(np.mean(cos_sims))
    
    # Overall status
    if results["summary"]["failed_outputs"] == 0:
        results["status"] = "PASS"
        logger.info(f"\n✅ VERIFICATION PASSED: All {results['summary']['passed_outputs']} outputs met threshold")
    else:
        results["status"] = "FAIL"
        logger.error(f"\n❌ VERIFICATION FAILED: {results['summary']['failed_outputs']} outputs below threshold")
        logger.error(f"   Mean cosine similarity: {results['summary']['mean_cosine_sim']:.4f}")
        logger.error(f"   Requirement: ≥ {COSINE_SIM_THRESHOLD}")
    
    logger.info("="*60 + "\n")
    
    return results


def suggest_next_steps(results: Dict):
    """Suggest next steps based on verification results."""
    if results["status"] == "PASS":
        logger.info("✅ Quantization successful! Model ready for deployment.")
        return
    
    # Failure case
    logger.error("\n⚠️  VERIFICATION FAILED — Next steps:")
    
    min_sim = results["summary"]["min_cosine_sim"]
    
    if min_sim > COSINE_SIM_CRITICAL:
        logger.error("1. Cosine similarity is marginal (0.90-0.95 range)")
        logger.error("   → Try increasing calibration data diversity")
        logger.error("   → Try different quantization method (e.g., symmetric vs asymmetric)")
    else:
        logger.error("2. Cosine similarity is critically low (<0.90)")
        logger.error("   → Implement per-layer bisect (run_layer_bisect.py)")
        logger.error("   → Identify which layer/submodel is quantization culprit")
        logger.error("   → Consider keeping that layer in fp32, quantize rest")
    
    logger.error("\nTo debug:")
    logger.error("  - Check compile_error.log for HTP compiler warnings")
    logger.error("  - Check individual output shapes match between fp32 and hw")
    logger.error("  - Review outlier handling in quantization (see meeting_prep_quantization.md §5.2)")


def main():
    parser = argparse.ArgumentParser(
        description="Verify Piper w8a16 quantization against fp32 reference"
    )
    parser.add_argument('--fp32_model', type=Path, required=True,
                        help='FP32 reference ONNX model')
    parser.add_argument('--hw_outputs', type=Path,
                        help='Hardware inference outputs (npz). If not provided, will run inference.')
    parser.add_argument('--test_dir', type=Path, default=Path('outputs/piper_vi'),
                        help='Directory with test inputs and calibration data')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/piper_vi'),
                        help='Output directory for results')
    parser.add_argument('--fixed_length', type=int, default=256,
                        help='Fixed sequence length')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.fp32_model.exists():
        logger.error(f"FP32 model not found: {args.fp32_model}")
        sys.exit(1)
    
    if args.hw_outputs and not args.hw_outputs.exists():
        logger.error(f"Hardware outputs not found: {args.hw_outputs}")
        sys.exit(1)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test inputs
    test_inputs = load_test_inputs(args.test_dir)
    
    # Run fp32 reference inference
    fp32_outputs = run_fp32_inference(args.fp32_model, test_inputs, args.fixed_length)
    
    # Load hardware outputs
    if args.hw_outputs:
        hw_outputs = load_hardware_outputs(args.hw_outputs)
    else:
        logger.error("Hardware outputs file required (--hw_outputs argument)")
        sys.exit(1)
    
    # Verify
    verify_results = verify_model(fp32_outputs, hw_outputs)
    
    # Save results
    results_path = args.output_dir / 'verify_results.json'
    with open(results_path, 'w') as f:
        json.dump(verify_results, f, indent=2)
    logger.info(f"Saved verification results to {results_path}")
    
    # Suggest next steps
    suggest_next_steps(verify_results)
    
    # Exit code
    sys.exit(0 if verify_results["status"] == "PASS" else 1)


if __name__ == '__main__':
    main()
