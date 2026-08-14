#!/usr/bin/env python3
"""
Master orchestration script for Piper quantization workflow.

Runs all steps in sequence:
1. Export Piper ONNX with fixed shape
2. Prepare calibration data
3. Quantize to w8a16
4. Compile and profile
5. Run inference and verify
6. Generate final report

Usage:
    python run_quantization_pipeline.py --all
    python run_quantization_pipeline.py --step export
    python run_quantization_pipeline.py --step quantize --skip_wait
"""

import json
import logging
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, List
import time

from common import setup_logging, Config, validate_output_dir, summarize_job_log

logger = setup_logging('pipeline')


class PipelineRunner:
    """Orchestrate Piper quantization pipeline."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = validate_output_dir(output_dir)
        self.config = Config(self.output_dir)
        self.steps_completed = []
        self.steps_failed = []
    
    def run_step(
        self,
        step_name: str,
        script_path: Path,
        args: List[str] = None,
        skip_if_done: bool = True
    ) -> bool:
        """
        Run a single pipeline step.
        
        Returns True if successful, False otherwise.
        """
        if skip_if_done and self.config.get(f'{step_name}_done'):
            logger.info(f"⏭️  Skipping {step_name} (already completed)")
            return True
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STEP: {step_name.upper()}")
        logger.info(f"{'='*60}")
        
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            return False
        
        # Build command
        cmd = ['python', str(script_path)]
        if args:
            cmd.extend(args)
        
        # Run
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True)
            logger.info(f"✅ {step_name} completed successfully")
            self.config.set(f'{step_name}_done', True)
            self.config.set(f'{step_name}_timestamp', time.time())
            self.steps_completed.append(step_name)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ {step_name} failed with exit code {e.returncode}")
            self.steps_failed.append(step_name)
            return False
        except Exception as e:
            logger.error(f"❌ {step_name} error: {e}")
            self.steps_failed.append(step_name)
            return False
    
    def run_all(self, skip_wait: bool = False) -> bool:
        """Run complete pipeline."""
        step_dir = Path(__file__).parent
        
        steps = [
            ('export', step_dir / 'export_piper_onnx.py', [
                '--output_dir', str(self.output_dir)
            ]),
            ('calibration', step_dir / 'make_calibration_data.py', [
                '--output_dir', str(self.output_dir),
                '--num_samples', '200'
            ]),
            ('quantize', step_dir / 'quantize_piper_w8a16.py', [
                '--model', str(self.output_dir / 'piper_vi_fp32_fixed_shape.onnx'),
                '--calib_data', str(self.output_dir / 'calibration_inputs.npz'),
                '--output_dir', str(self.output_dir),
                *([] if not skip_wait else ['--skip_wait'])
            ]),
            ('compile', step_dir / 'compile_and_profile.py', [
                '--model', str(self.output_dir / 'piper_vi_int8_w8a16.onnx'),
                '--output_dir', str(self.output_dir)
            ]),
            ('verify', step_dir / 'verify_piper_w8a16.py', [
                '--fp32_model', str(self.output_dir / 'piper_vi_fp32_fixed_shape.onnx'),
                '--hw_outputs', str(self.output_dir / 'hw_outputs.npz'),
                '--output_dir', str(self.output_dir)
            ]),
            ('report', step_dir / 'generate_report.py', [
                '--job_log', str(self.output_dir / 'job_log.json'),
                '--verify_results', str(self.output_dir / 'verify_results.json'),
                '--output_dir', str(self.output_dir),
                '--output_file', str(self.output_dir / 'REPORT.md')
            ]),
        ]
        
        success = True
        for step_name, script_path, args in steps:
            if not self.run_step(step_name, script_path, args):
                logger.error(f"Pipeline stopped at {step_name}")
                success = False
                break
        
        self.print_summary()
        return success
    
    def print_summary(self):
        """Print pipeline execution summary."""
        logger.info(f"\n{'='*60}")
        logger.info("PIPELINE SUMMARY")
        logger.info(f"{'='*60}")
        
        if self.steps_completed:
            logger.info(f"\n✅ Completed ({len(self.steps_completed)}):")
            for step in self.steps_completed:
                logger.info(f"  - {step}")
        
        if self.steps_failed:
            logger.error(f"\n❌ Failed ({len(self.steps_failed)}):")
            for step in self.steps_failed:
                logger.error(f"  - {step}")
        
        # Show job log
        job_log_path = self.output_dir / 'job_log.json'
        if job_log_path.exists():
            logger.info(f"\n{summarize_job_log(job_log_path)}")
        
        # Show report
        report_path = self.output_dir / 'REPORT.md'
        if report_path.exists():
            logger.info(f"\n📄 Report: {report_path}")
        
        logger.info(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run Piper quantization pipeline"
    )
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/piper_vi'),
                        help='Output directory')
    parser.add_argument('--all', action='store_true',
                        help='Run all steps in sequence')
    parser.add_argument('--step', choices=[
        'export', 'calibration', 'quantize', 'compile', 'verify', 'report'
    ], help='Run single step')
    parser.add_argument('--skip_wait', action='store_true',
                        help='Skip waiting for long-running jobs (quantize, compile)')
    parser.add_argument('--reset', action='store_true',
                        help='Reset pipeline (clear cached step completions)')
    
    args = parser.parse_args()
    
    runner = PipelineRunner(args.output_dir)
    
    if args.reset:
        logger.info("Resetting pipeline cache...")
        runner.config.data = {}
        runner.config.save()
    
    # Run
    if args.all:
        success = runner.run_all(skip_wait=args.skip_wait)
        sys.exit(0 if success else 1)
    
    elif args.step:
        step_dir = Path(__file__).parent
        step_scripts = {
            'export': step_dir / 'export_piper_onnx.py',
            'calibration': step_dir / 'make_calibration_data.py',
            'quantize': step_dir / 'quantize_piper_w8a16.py',
            'compile': step_dir / 'compile_and_profile.py',
            'verify': step_dir / 'verify_piper_w8a16.py',
            'report': step_dir / 'generate_report.py',
        }
        
        script = step_scripts[args.step]
        success = runner.run_step(args.step, script, skip_if_done=False)
        sys.exit(0 if success else 1)
    
    else:
        parser.print_help()
        logger.error("\nError: specify --all or --step <step_name>")
        sys.exit(1)


if __name__ == '__main__':
    main()
