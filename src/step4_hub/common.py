#!/usr/bin/env python3
"""
Common utilities for Piper quantization workflows.

Provides shared functions for:
- Job logging and tracking
- File operations
- Configuration management
- Logging setup
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import time


def setup_logging(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Configure logging with both console and file output."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / f'{name}.log')
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def load_job_log(job_log_path: Path) -> list:
    """Load job log (append mode)."""
    if not job_log_path.exists():
        return []
    
    try:
        with open(job_log_path, 'r') as f:
            return json.load(f)
    except:
        return []


def append_job_log(job_log_path: Path, job_data: Dict[str, Any]):
    """Append job to log file (does not overwrite)."""
    log = load_job_log(job_log_path)
    log.append(job_data)
    
    with open(job_log_path, 'w') as f:
        json.dump(log, f, indent=2)


def save_config(config_path: Path, config: Dict[str, Any]):
    """Save configuration to JSON file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    if not config_path.exists():
        return {}
    
    with open(config_path, 'r') as f:
        return json.load(f)


def create_job_record(
    job_type: str,
    model_path: Optional[str] = None,
    device: Optional[str] = None,
    status: str = "submitted",
    **kwargs
) -> Dict[str, Any]:
    """Create a standardized job record for logging."""
    record = {
        "job_type": job_type,
        "status": status,
        "timestamp": time.time(),
    }
    
    if model_path:
        record["model"] = str(model_path)
    
    if device:
        record["device"] = device
    
    # Add any additional fields
    record.update(kwargs)
    
    return record


def validate_model_file(model_path: Path) -> bool:
    """Validate that model file exists and is readable."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    if not model_path.is_file():
        raise ValueError(f"Not a file: {model_path}")
    
    if model_path.suffix.lower() != '.onnx':
        raise ValueError(f"Expected .onnx file, got: {model_path.suffix}")
    
    return True


def validate_output_dir(output_dir: Path) -> Path:
    """Ensure output directory exists and is writable."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test write permissions
    test_file = output_dir / '.write_test'
    try:
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise PermissionError(f"Cannot write to {output_dir}: {e}")
    
    return output_dir


def summarize_job_log(job_log_path: Path, max_show: int = 10) -> str:
    """Create text summary of job log."""
    log = load_job_log(job_log_path)
    
    if not log:
        return "No jobs logged yet"
    
    lines = [f"Job Log Summary ({len(log)} total jobs):"]
    
    for i, job in enumerate(log[-max_show:], 1):
        job_type = job.get('job_type', 'unknown').upper()
        job_id = job.get('job_id', 'N/A')[:16]  # Truncate long IDs
        status = job.get('status', 'unknown')
        lines.append(f"  {i}. [{job_type}] {job_id}... → {status}")
    
    if len(log) > max_show:
        lines.append(f"  ... and {len(log) - max_show} more")
    
    return '\n'.join(lines)


class Config:
    """Configuration manager for quantization pipeline."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.config_file = self.output_dir / 'config.json'
        self.data = self.load()
    
    def load(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def save(self):
        """Save configuration to file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get configuration value."""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.data[key] = value
        self.save()
    
    def __repr__(self):
        return f"Config({self.config_file}): {self.data}"
