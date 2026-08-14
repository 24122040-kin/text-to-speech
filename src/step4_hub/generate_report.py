#!/usr/bin/env python3
"""
Generate comprehensive deployment report from job logs and verification results.

Report includes:
- Model metrics table (size, latency, accuracy)
- All job IDs for audit trail
- Per-output verification scores
- Conclusions and recommendations
- Links to original documentation
"""

import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_json_file(path: Path, default=None):
    """Safely load JSON file."""
    if not path.exists():
        if default is None:
            logger.warning(f"File not found: {path}")
        return default
    
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return default


def format_job_id_table(jobs: List[Dict]) -> str:
    """Format job log as markdown table."""
    if not jobs:
        return "*(No jobs logged)*"
    
    lines = [
        "| Job Type | Job ID | Status | Timestamp |",
        "|----------|--------|--------|-----------|",
    ]
    
    for job in jobs:
        job_type = job.get('job_type', 'unknown').upper()
        job_id = job.get('job_id', 'N/A')
        status = job.get('status', 'unknown').upper()
        timestamp = job.get('timestamp', 'N/A')
        
        # Format timestamp
        if isinstance(timestamp, (int, float)):
            dt = datetime.fromtimestamp(timestamp)
            timestamp = dt.strftime('%Y-%m-%d %H:%M:%S')
        
        lines.append(f"| {job_type} | `{job_id}` | {status} | {timestamp} |")
    
    return '\n'.join(lines)


def format_verify_results(verify_results: Dict) -> str:
    """Format verification results as markdown section."""
    if not verify_results:
        return "*(Verification not yet completed)*"
    
    summary = verify_results.get('summary', {})
    per_output = verify_results.get('per_output', {})
    status = verify_results.get('status', 'UNKNOWN')
    
    lines = []
    
    # Status
    status_emoji = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⏳"
    lines.append(f"**Status:** {status_emoji} {status}")
    
    # Summary metrics
    lines.append(f"\n**Summary Metrics:**")
    lines.append(f"- Total outputs: {summary.get('total_outputs', 0)}")
    lines.append(f"- Passed outputs: {summary.get('passed_outputs', 0)}")
    lines.append(f"- Failed outputs: {summary.get('failed_outputs', 0)}")
    lines.append(f"- Mean cosine similarity: **{summary.get('mean_cosine_sim', 0):.4f}**")
    lines.append(f"- Min cosine similarity: {summary.get('min_cosine_sim', 0):.4f}")
    lines.append(f"- Max cosine similarity: {summary.get('max_cosine_sim', 0):.4f}")
    lines.append(f"- Threshold: ≥ 0.95")
    
    # Per-output details
    if per_output:
        lines.append(f"\n**Per-Output Results:**")
        lines.append("")
        lines.append("| Output | Cosine Similarity | Status |")
        lines.append("|--------|-------------------|--------|")
        
        for out_name, out_data in per_output.items():
            cos_sim = out_data.get('cosine_similarity', 0)
            passed = out_data.get('passed', False)
            status_str = "✅ PASS" if passed else "❌ FAIL"
            lines.append(f"| {out_name} | {cos_sim:.4f} | {status_str} |")
    
    return '\n'.join(lines)


def format_quantization_summary(output_dir: Path) -> str:
    """Summarize quantization parameters."""
    lines = []
    
    lines.append("**Quantization Configuration:**")
    lines.append("- Weights dtype: int8")
    lines.append("- Activations dtype: int16")
    lines.append("- Precision: w8a16")
    lines.append("- Calibration samples: ≥200 (Vietnamese)")
    
    # Load calibration stats if available
    calib_stats = load_json_file(output_dir / 'calibration_stats.json')
    if calib_stats:
        lines.append(f"- Actual calibration samples: {calib_stats.get('num_samples', 'N/A')}")
    
    # Load length analysis if available
    length_analysis = load_json_file(output_dir / 'length_analysis.json')
    if length_analysis:
        lines.append(f"- Fixed sequence length: {length_analysis.get('fixed_length', 'N/A')}")
    
    return '\n'.join(lines)


def generate_report(
    job_log_path: Path,
    verify_results_path: Path,
    output_dir: Path,
    output_file: Path
) -> str:
    """Generate markdown report."""
    
    # Load data
    job_log = load_json_file(job_log_path, default=[])
    verify_results = load_json_file(verify_results_path, default={})
    
    # Build report
    lines = []
    
    lines.append("# Piper (Vietnamese) w8a16 Deployment Report")
    lines.append("")
    
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    
    overall_status = verify_results.get('status', 'UNKNOWN')
    status_emoji = "✅" if overall_status == "PASS" else "❌" if overall_status == "FAIL" else "⏳"
    lines.append(f"**Overall Status:** {status_emoji} {overall_status}")
    lines.append("")
    
    if overall_status == "PASS":
        lines.append("Piper Vietnamese model has been successfully quantized to w8a16 precision and verified on Qualcomm AI Hub hardware. The model is ready for deployment.")
    elif overall_status == "FAIL":
        lines.append("Verification failed — cosine similarity did not meet acceptance threshold. See Verification section for details.")
    else:
        lines.append("Verification not yet completed. Run verify_piper_w8a16.py to check model output quality.")
    
    lines.append("")
    
    # Quantization configuration
    lines.append("## Quantization Configuration")
    lines.append("")
    lines.append(format_quantization_summary(output_dir))
    lines.append("")
    
    # Job log
    lines.append("## Deployment Jobs")
    lines.append("")
    lines.append(format_job_id_table(job_log))
    lines.append("")
    
    # Verification results
    lines.append("## Verification Results")
    lines.append("")
    lines.append(format_verify_results(verify_results))
    lines.append("")
    
    # Recommendations
    lines.append("## Next Steps")
    lines.append("")
    
    if overall_status == "PASS":
        lines.append("1. Deploy compiled model (`.bin` file) to target device")
        lines.append("2. Run end-to-end pipeline test on hardware")
        lines.append("3. Measure actual latency/power on device")
        lines.append("4. Prepare deployment documentation")
    else:
        lines.append("1. Review verification error log for details")
        lines.append("2. Run per-layer bisect to identify problematic layers")
        lines.append("3. Consider mixed-precision (some layers fp32, others int8)")
        lines.append("4. Increase calibration data diversity")
        lines.append("5. Check for outliers in weight/activation distributions")
    
    lines.append("")
    
    # References
    lines.append("## References")
    lines.append("")
    lines.append("- [Runbook: Deploy Piper to Qualcomm AI Hub](piper_qualcomm_deploy_runbook.md)")
    lines.append("- [Quantization Study](meeting_prep_quantization.md)")
    lines.append("- [Step 4 Documentation](step4.md)")
    lines.append("")
    
    # Audit trail
    lines.append("## Audit Trail")
    lines.append("")
    lines.append(f"Report generated at: {output_dir}")
    lines.append(f"Job log: job_log.json")
    lines.append(f"Verification results: verify_results.json")
    lines.append(f"Calibration stats: calibration_stats.json")
    
    report_content = '\n'.join(lines)
    
    # Save report
    with open(output_file, 'w') as f:
        f.write(report_content)
    
    logger.info(f"✅ Report saved to {output_file}")
    
    return report_content


def main():
    parser = argparse.ArgumentParser(
        description="Generate Piper deployment report"
    )
    parser.add_argument('--job_log', type=Path, default=Path('outputs/piper_vi/job_log.json'),
                        help='Job log file')
    parser.add_argument('--verify_results', type=Path, default=Path('outputs/piper_vi/verify_results.json'),
                        help='Verification results file')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/piper_vi'),
                        help='Output directory')
    parser.add_argument('--output_file', type=Path, default=Path('outputs/piper_vi/REPORT.md'),
                        help='Output report file')
    
    args = parser.parse_args()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report = generate_report(
        args.job_log,
        args.verify_results,
        args.output_dir,
        args.output_file
    )
    
    logger.info("\n" + "="*60)
    logger.info("DEPLOYMENT REPORT GENERATED")
    logger.info("="*60)
    logger.info(f"\nLocation: {args.output_file}")
    logger.info(f"\nPreview:\n")
    # Print first 30 lines
    report_lines = report.split('\n')
    for line in report_lines[:30]:
        print(line)
    if len(report_lines) > 30:
        print(f"... ({len(report_lines) - 30} more lines)")


if __name__ == '__main__':
    main()
