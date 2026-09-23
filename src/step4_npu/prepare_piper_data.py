#!/usr/bin/env python3
"""Step 4 NPU -- Prepare Piper (vi) calibration + test data (fresh pipeline).

Uses the REAL Piper 1.6.1 tokenizer (espeak-ng phonemizer + phoneme_id_map
from the voice config) to tokenize real Vietnamese sentences from the project
manifests. Emits fixed-length padded arrays for both calibration and held-out
test sets, plus the tokenized text for the evaluation phase.

Inputs (Piper VITS text->audio end-to-end, incl. vocoder):
    input         : int64 [1, L]     phoneme ids, padded to fixed L
    input_lengths : int64 [1]        real phoneme length (unpadded)
    scales        : float32 [3]      [noise_scale, length_scale, noise_w_scale]

Output:
    output        : float32 [1, T, 1, 1]   audio waveform (dynamic T)

Usage:
    python prepare_piper_data.py --output_dir outputs/piper_vi_npu
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_SCALES = [0.667, 1.0, 0.8]  # noise_scale, length_scale, noise_w_scale (from config)


def load_real_vietnamese_sentences(root: Path) -> list:
    """Load real Vietnamese sentences from MT + ASR manifests (dedup, keep order)."""
    sentences = []

    mt_path = root / "data" / "mt" / "manifest.json"
    if mt_path.exists():
        with open(mt_path, "r", encoding="utf-8") as f:
            mt = json.load(f)
        for item in mt:
            if isinstance(item, dict) and item.get("vi"):
                sentences.append(item["vi"].strip())

    asr_path = root / "data" / "asr" / "manifest.json"
    if asr_path.exists():
        with open(asr_path, "r", encoding="utf-8") as f:
            asr = json.load(f)
        for item in asr:
            if isinstance(item, dict) and item.get("lang") == "vi" and item.get("transcript"):
                sentences.append(item["transcript"].strip())

    seen = set()
    uniq = []
    for s in sentences:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def tokenize_piper(text: str, phonemizer, id_map: dict) -> list:
    """Real Piper tokenizer: espeak phonemize -> phoneme ids (incl. BOS/EOS/PAD)."""
    from piper.voice import phonemes_to_ids

    phonemes = phonemizer.phonemize("vi", text)[0]
    return phonemes_to_ids(phonemes, id_map)


def build_dataset(texts, fixed_length, phonemizer, id_map):
    """Tokenize texts, pad to fixed_length, return arrays + real lengths."""
    input_arr = np.zeros((len(texts), fixed_length), dtype=np.int64)
    lengths = np.zeros((len(texts),), dtype=np.int64)
    failed = 0
    for i, text in enumerate(texts):
        try:
            ids = tokenize_piper(text, phonemizer, id_map)
            n = len(ids)
            if n > fixed_length:
                ids = ids[:fixed_length]
                n = fixed_length
            input_arr[i, :n] = ids
            lengths[i] = n
        except Exception as e:  # noqa: BLE001
            failed += 1
            logger.warning("tokenize failed for %r: %s", text[:40], e)
    logger.info("tokenized %d texts (failed=%d), lengths min=%d max=%d",
                len(texts), failed, lengths.min(), lengths.max())
    return input_arr, lengths


def main():
    parser = argparse.ArgumentParser(description="Prepare Piper calibration + test data")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/piper_vi_npu"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--num_calib", type=int, default=220, help="Calibration samples")
    parser.add_argument("--num_test", type=int, default=8, help="Held-out test samples")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_real_vietnamese_sentences(args.root)
    logger.info("Loaded %d real Vietnamese sentences", len(sentences))
    if len(sentences) < 10:
        raise RuntimeError("Too few real Vietnamese sentences found")

    # Tokenizer + config
    from piper.phonemize_espeak import EspeakPhonemizer

    phonemizer = EspeakPhonemizer()
    cfg_path = args.output_dir / "vi_VN-vais1000-medium.onnx.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Voice config not found: {cfg_path} (run download_piper_model.py first)")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    id_map = cfg["phoneme_id_map"]
    logger.info("phoneme vocab size: %d (num_symbols=%d)", len(id_map), cfg.get("num_symbols"))

    rng = random.Random(args.seed)

    # Split: hold out test sentences first (real, never spliced)
    rng.shuffle(sentences)
    test_texts = sentences[: args.num_test]
    calib_source = sentences[args.num_test:]

    # Measure real phoneme lengths to choose fixed_length automatically
    probe_lens = []
    for s in calib_source[:200]:
        try:
            probe_lens.append(len(tokenize_piper(s, phonemizer, id_map)))
        except Exception:  # noqa: BLE001
            pass
    if probe_lens:
        p95 = float(np.percentile(probe_lens, 95))
        fixed_length = int(np.ceil(p95 / 32) * 32)
        fixed_length = max(64, min(672, fixed_length))
        logger.info("phoneme len stats: n=%d min=%d max=%d p50=%.0f p95=%.0f -> fixed_length=%d",
                    len(probe_lens), min(probe_lens), max(probe_lens),
                    np.percentile(probe_lens, 50), p95, fixed_length)
    else:
        fixed_length = 256
        logger.warning("no probe lengths; using fixed_length=%d", fixed_length)

    # Build calibration set: real + splice variants up to num_calib
    from prepare_piper_data_splice import make_splice_variants

    calib_texts = make_splice_variants(calib_source, args.num_calib, rng)
    logger.info("Calibration set: %d samples", len(calib_texts))

    # Calibration arrays
    calib_input, calib_lengths = build_dataset(calib_texts, fixed_length, phonemizer, id_map)
    calib_scales = np.tile(np.array(DEFAULT_SCALES, dtype=np.float32), (len(calib_input), 1))

    # Test arrays
    test_input, test_lengths = build_dataset(test_texts, fixed_length, phonemizer, id_map)
    test_scales = np.tile(np.array(DEFAULT_SCALES, dtype=np.float32), (len(test_input), 1))

    # Save everything (calibration per-sample dicts for qai-hub, arrays for local use)
    calib_dict = {
        "input": [calib_input[i:i + 1] for i in range(len(calib_input))],
        "input_lengths": [calib_lengths[i:i + 1] for i in range(len(calib_lengths))],
        "scales": [calib_scales[i] for i in range(len(calib_scales))],
    }
    test_dict = {
        "input": [test_input[i:i + 1] for i in range(len(test_input))],
        "input_lengths": [test_lengths[i:i + 1] for i in range(len(test_lengths))],
        "scales": [test_scales[i] for i in range(len(test_scales))],
    }
    np.savez(
        args.output_dir / "piper_vi_npu_data.npz",
        input=calib_input, input_lengths=calib_lengths, scales=calib_scales,
        test_input=test_input, test_lengths=test_lengths, test_scales=test_scales,
        calib_texts=np.array(calib_texts, dtype=object),
        test_texts=np.array(test_texts, dtype=object),
    )
    with open(args.output_dir / "calibration_dict.json", "w", encoding="utf-8") as f:
        json.dump({"num_calib": len(calib_input), "num_test": len(test_input),
                   "fixed_length": fixed_length}, f, indent=2)

    stats = {
        "fixed_length": fixed_length,
        "num_calib": len(calib_input),
        "num_test": len(test_input),
        "calib_length_min": int(calib_lengths.min()),
        "calib_length_max": int(calib_lengths.max()),
        "calib_length_mean": float(calib_lengths.mean()),
        "calib_length_p50": float(np.percentile(calib_lengths, 50)),
        "calib_length_p95": float(np.percentile(calib_lengths, 95)),
        "test_texts": test_texts,
    }
    with open(args.output_dir / "data_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info("Saved data to %s", args.output_dir / "piper_vi_npu_data.npz")
    logger.info("Calib: %d samples, fixed_length=%d | Test: %d samples",
                len(calib_input), fixed_length, len(test_input))
    for i, t in enumerate(test_texts):
        logger.info("  test[%d] len=%d: %s", i, int(test_lengths[i]), t[:70])


if __name__ == "__main__":
    main()
