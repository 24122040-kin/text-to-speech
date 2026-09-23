#!/usr/bin/env python3
"""Step 4 NPU -- Evaluate Piper (vi) hardware audio quality (round-trip ASR).

Synthesized NPU audio -> Vietnamese ASR (PhoWhisper-small) -> WER/CER vs the
source text. Uses only stdlib + numpy for wav I/O and edit-distance metrics so
no extra pip installs are required.

Usage:
    python evaluate_hw_pipeline.py --output_dir outputs/piper_vi_npu [--skip_asr]
"""

import argparse
import json
import logging
import os
import re
import sys
import wave
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SR = 16000


def normalize_text(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_wav(path: Path) -> np.ndarray:
    """Read a 22050 Hz wav (written by stdlib wave) -> 16 kHz mono float32."""
    with wave.open(str(path), "rb") as wf:
        n = wf.getnframes()
        raw = wf.readframes(n)
        sr = wf.getframerate()
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if sr != SR:
        # simple linear resample
        t_out = np.arange(int(len(x) * SR / sr)) / SR
        t_in = np.arange(len(x)) / sr
        x = np.interp(t_out, t_in, x).astype(np.float32)
    return x


def edit_distance(a, b):
    """Levenshtein distance over token lists."""
    dp = np.zeros((len(a) + 1, len(b) + 1), dtype=np.int32)
    dp[:, 0] = np.arange(len(a) + 1)
    dp[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    return int(dp[len(a), len(b)])


def wer_cer(hyp: str, ref: str):
    rw, hw = ref.split(), hyp.split()
    wer = edit_distance(rw, hw) / max(len(rw), 1)
    cer = edit_distance(list(ref), list(hyp)) / max(len(ref), 1)
    return wer, cer


def transcribe(wav: np.ndarray, model_dir: str | None = None) -> str:
    """PhoWhisper-small (vinai) transcription from a numpy waveform."""
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    torch.set_grad_enabled(False)
    model_id = model_dir or "vinai/PhoWhisper-small"
    processor = AutoProcessor.from_pretrained(model_id, local_files_only=model_dir is not None)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, local_files_only=model_dir is not None)
    model.eval()
    features = processor(wav, sampling_rate=SR, return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**features)
    return processor.batch_decode(gen, skip_special_tokens=True)[0]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Piper HW audio")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/piper_vi_npu"))
    parser.add_argument("--skip_asr", action="store_true")
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Local PhoWhisper model dir (avoids HF symlink cache)")
    args = parser.parse_args()

    out = args.output_dir
    data = np.load(out / "piper_vi_npu_data.npz", allow_pickle=True)
    test_texts = [str(t) for t in data["test_texts"]]
    sim = json.load(open(out / "hw" / "similarity_results.json", encoding="utf-8"))
    wav_dir = out / "hw" / "audio"

    results = []
    for r in sim["per_sample"]:
        idx = r["idx"]
        wav_path = wav_dir / f"hw_{idx}.wav"
        if not wav_path.exists():
            logger.warning("missing %s", wav_path)
            continue
        wav = load_wav(wav_path)
        entry = {"idx": idx, "text": test_texts[idx],
                 "duration_sec": round(len(wav) / SR, 3),
                 "peak": round(float(np.abs(wav).max()), 4),
                 "rms": round(float(np.sqrt(np.mean(wav ** 2))), 4),
                 "cos_sim": r["audio_cos"]}
        if not args.skip_asr:
            try:
                hyp = transcribe(wav, model_dir=args.model_dir)
                wer, cer = wer_cer(normalize_text(hyp), normalize_text(test_texts[idx]))
                entry.update({"hypothesis": hyp, "wer": round(wer, 4), "cer": round(cer, 4)})
                logger.info("test[%d] WER=%.3f CER=%.3f hyp=%s", idx, wer, cer, hyp[:70])
            except Exception as e:
                logger.warning("ASR failed for test[%d]: %s", idx, e)
                entry["asr_error"] = str(e)[:300]
        results.append(entry)

    wers = [r.get("wer") for r in results if "wer" in r]
    cers = [r.get("cer") for r in results if "cer" in r]
    summary = {"mean_wer": round(float(np.mean(wers)), 4) if wers else None,
               "mean_cer": round(float(np.mean(cers)), 4) if cers else None,
               "n_asr": len(wers)}
    with open(out / "hw" / "evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_sample": results}, f, indent=2, ensure_ascii=False)
    logger.info("evaluation -> %s", out / "hw" / "evaluation_results.json")
    logger.info("SUMMARY: %s", summary)


if __name__ == "__main__":
    main()
