#!/usr/bin/env python3
"""
Prepare calibration data for Piper w8a16 quantization.

Calibration data must:
1. Be representative of real Vietnamese text distribution
2. Cover diverse lengths (short/medium/long sentences)
3. Be at least 200 samples (per runbook requirements)
4. Be pre-tokenized with proper padding/truncation to fixed_length

Uses data/mt/manifest.json as source of real Vietnamese sentences.
"""

import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_vietnamese_sentences(manifest_path: Path, max_samples: int = 500) -> List[str]:
    """
    Extract Vietnamese text samples from manifest file.
    
    Manifest format (expected): one JSON per line with 'vi' or 'text' field
    """
    logger.info(f"Reading Vietnamese sentences from {manifest_path}")
    
    sentences = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(sentences) >= max_samples:
                break
            
            try:
                data = json.loads(line.strip())
            except:
                continue
            
            # Try to find Vietnamese text
            text = data.get('vi') or data.get('vi_VN') or data.get('text')
            if text and isinstance(text, str) and len(text.strip()) > 2:
                sentences.append(text.strip())
    
    logger.info(f"Extracted {len(sentences)} Vietnamese sentences from manifest")
    return sentences


def tokenize_piper_text(text: str, fixed_length: int) -> np.ndarray:
    """
    Tokenize Vietnamese text using Piper's tokenizer.
    
    Returns numpy array of token IDs, padded/truncated to fixed_length.
    """
    try:
        from piper.phonemizer import Phonemizer
        from piper.voice import PiperVoice
    except ImportError:
        logger.error("piper-tts required for tokenization")
        raise
    
    # Placeholder: Piper uses character-level encoding with phoneme processing
    # For now, we'll use simple character-based encoding as a stand-in
    # TODO: Use actual Piper tokenizer when available
    
    # Simple encoding: convert text to character indices
    chars = sorted(set(text))
    char_to_id = {c: i + 1 for i, c in enumerate(chars)}  # Reserve 0 for padding
    
    tokens = [char_to_id.get(c, 0) for c in text]
    
    # Pad or truncate to fixed_length
    if len(tokens) < fixed_length:
        tokens = tokens + [0] * (fixed_length - len(tokens))
    else:
        tokens = tokens[:fixed_length]
    
    return np.array(tokens, dtype=np.int32)


def prepare_calibration_dataset(
    sentences: List[str],
    fixed_length: int,
    num_samples: int = 200,
    output_dir: Path = None
) -> Tuple[np.ndarray, dict]:
    """
    Prepare calibration dataset with proper distribution.
    
    Strategy:
    1. Sort sentences by length to ensure coverage of short/medium/long
    2. Sample uniformly across length distribution
    3. Create a diverse calibration set
    
    Returns:
        (calibration_array, stats_dict)
    """
    logger.info(f"Preparing calibration dataset with {num_samples} samples")
    
    if len(sentences) < num_samples:
        logger.warning(f"Only {len(sentences)} sentences available, using all")
        selected_sentences = sentences
    else:
        # Sort by length and sample uniformly across distribution
        sentences_with_len = [(s, len(s)) for s in sentences]
        sentences_with_len.sort(key=lambda x: x[1])
        
        # Sample uniformly from length-sorted list
        step = len(sentences_with_len) // num_samples
        selected_sentences = [s for s, _ in sentences_with_len[::step]][:num_samples]
    
    # Tokenize all sentences
    calibration_inputs = []
    lengths = []
    
    for text in selected_sentences:
        tokens = tokenize_piper_text(text, fixed_length)
        calibration_inputs.append(tokens)
        lengths.append(np.count_nonzero(tokens))  # Count non-padding tokens
    
    # Stack into single array
    calibration_array = np.stack(calibration_inputs, axis=0)  # Shape: (num_samples, fixed_length)
    
    # Compute statistics
    stats = {
        "num_samples": len(calibration_inputs),
        "fixed_length": fixed_length,
        "min_length": int(np.min(lengths)),
        "max_length": int(np.max(lengths)),
        "mean_length": float(np.mean(lengths)),
        "median_length": float(np.median(lengths)),
        "array_shape": list(calibration_array.shape),
        "dtype": str(calibration_array.dtype),
    }
    
    logger.info(f"Calibration dataset prepared:")
    logger.info(f"  Shape: {calibration_array.shape}")
    logger.info(f"  Active token lengths: min={stats['min_length']}, "
                f"max={stats['max_length']}, mean={stats['mean_length']:.1f}")
    
    return calibration_array, stats


def main():
    parser = argparse.ArgumentParser(
        description="Prepare calibration data for Piper quantization"
    )
    parser.add_argument('--input_manifest', type=Path, default=Path('data/mt/manifest.json'),
                        help='Input manifest file with Vietnamese texts')
    parser.add_argument('--output_dir', type=Path, default=Path('outputs/piper_vi'),
                        help='Output directory')
    parser.add_argument('--fixed_length', type=int, default=256,
                        help='Fixed sequence length for model input')
    parser.add_argument('--num_samples', type=int, default=200,
                        help='Number of calibration samples (minimum 200 per runbook)')
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load manifest and extract Vietnamese sentences
    if not args.input_manifest.exists():
        logger.error(f"Manifest not found: {args.input_manifest}")
        raise FileNotFoundError(f"{args.input_manifest}")
    
    sentences = extract_vietnamese_sentences(args.input_manifest)
    
    if not sentences:
        logger.error("No Vietnamese sentences found in manifest")
        raise ValueError("Empty sentence list")
    
    # Prepare calibration dataset
    calibration_array, stats = prepare_calibration_dataset(
        sentences,
        args.fixed_length,
        args.num_samples,
        args.output_dir
    )
    
    # Save calibration data
    calib_path = args.output_dir / 'calibration_inputs.npz'
    np.savez(calib_path, calibration_array=calibration_array)
    logger.info(f"✅ Saved calibration data to {calib_path}")
    
    # Save statistics
    stats_path = args.output_dir / 'calibration_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"✅ Saved statistics to {stats_path}")


if __name__ == '__main__':
    main()
