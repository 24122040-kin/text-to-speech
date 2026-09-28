#!/usr/bin/env python3
"""Check similarity between input text and generated audio from the 100% Zero-CPU NPU Pipeline.

Uses PhoWhisper ASR to transcribe the output audio and compares the recognized text
with the original input text using WER, CER, and Sequence Matching (Levenshtein similarity).
"""

import difflib
import json
import logging
import re
import sys
import unicodedata
import wave
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize Vietnamese text for fair similarity evaluation."""
    text = unicodedata.normalize("NFC", text.lower().strip())
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Collapse multiple whitespaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_cer(ref: str, hyp: str) -> float:
    """Compute Character Error Rate (CER) using Levenshtein distance."""
    r = list(ref)
    h = list(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]) + 1
    return float(d[len(r), len(h)]) / max(len(r), 1)


def compute_wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate (WER) using Levenshtein distance on words."""
    r = ref.split()
    h = hyp.split()
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j], d[i, j - 1], d[i - 1, j - 1]) + 1
    return float(d[len(r), len(h)]) / max(len(r), 1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check similarity using PhoWhisper ASR")
    parser.add_argument("--in_dir", type=Path, default=Path("outputs/piper_vi_npu/zero_cpu_results"))
    parser.add_argument("--summary_file", type=Path, default=Path("outputs/piper_vi_npu/e2e_results/evaluation_summary.json"))
    args = parser.parse_args()

    e2e_dir = args.in_dir
    summary_file = args.summary_file
    
    if not summary_file.exists():
        logger.error("Summary file not found: %s", summary_file)
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    logger.info("Loaded %d samples from %s", len(samples), summary_file)

    # Initialize PhoWhisper ASR pipeline
    model_id = "outputs/phowhisper_small"
    if not Path(model_id).exists():
        model_id = "vinai/phowhisper-small"

    logger.info("Loading PhoWhisper model from: %s", model_id)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=False,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    def read_wav_16k(wav_path: Path) -> np.ndarray:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            n_frames = wf.getnframes()
            data = wf.readframes(n_frames)
            if sampwidth == 2:
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 4:
                audio = np.frombuffer(data, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(data, dtype=np.uint8).astype(np.float32) / 128.0 - 1.0
            if n_channels > 1:
                audio = audio.reshape(-1, n_channels).mean(axis=1)
            return audio

    results = []
    print("\n" + "=" * 90)
    print(" BẢNG KIỂM TRA ĐỘ TƯƠNG ĐỒNG (SIMILARITY) GIỮA INPUT TEXT VÀ AUDIO OUTPUT PIPELINE")
    print("=" * 90)

    for item in samples:
        idx = item["sample_idx"]
        input_text = item["text"]
        wav_path = e2e_dir / item["wav_16k"]

        if not wav_path.exists():
            logger.warning("WAV file not found: %s", wav_path)
            continue

        # Transcribe audio using PhoWhisper
        audio_np = read_wav_16k(wav_path)
        input_features = processor(audio_np, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(device, dtype=torch_dtype)

        forced_decoder_ids = processor.get_decoder_prompt_ids(language="vi", task="transcribe")
        with torch.no_grad():
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        # Normalize texts for comparison
        norm_ref = normalize_text(input_text)
        norm_hyp = normalize_text(transcription)

        # Compute metrics
        cer = compute_cer(norm_ref, norm_hyp)
        wer = compute_wer(norm_ref, norm_hyp)
        seq_sim = difflib.SequenceMatcher(None, norm_ref, norm_hyp).ratio()

        res_entry = {
            "sample_idx": idx,
            "input_text": input_text,
            "asr_transcription": transcription,
            "norm_input": norm_ref,
            "norm_asr": norm_hyp,
            "similarity_score": round(seq_sim * 100.0, 2),
            "cer_percent": round(cer * 100.0, 2),
            "wer_percent": round(wer * 100.0, 2),
            "audio_duration_sec": item["duration_sec"],
        }
        results.append(res_entry)

        print(f"\n[Sample #{idx:02d}] (Thời lượng: {item['duration_sec']}s | Độ tương đồng: {seq_sim*100.0:.1f}%)")
        print(f"  📝 INPUT:         {input_text}")
        print(f"  🎙️ PHOWHISPER ASR: {transcription}")
        print(f"  📊 Similarity:    {seq_sim*100.0:.2f}% | WER: {wer*100.0:.2f}% | CER: {cer*100.0:.2f}%")

    # Overall Metrics
    mean_sim = np.mean([r["similarity_score"] for r in results])
    mean_wer = np.mean([r["wer_percent"] for r in results])
    mean_cer = np.mean([r["cer_percent"] for r in results])

    print("\n" + "=" * 90)
    print(" TỔNG KẾT TOÀN DIỆN VỀ ĐỘ TƯƠNG ĐỒNG (INPUT vs OUTPUT)")
    print("=" * 90)
    print(f"  • Độ tương đồng trung bình (Sequence Similarity):  {mean_sim:.2f}%")
    print(f"  • Tỉ lệ lỗi từ trung bình (Word Error Rate - WER): {mean_wer:.2f}%")
    print(f"  • Tỉ lệ lỗi ký tự trung bình (Character Error Rate - CER): {mean_cer:.2f}%")
    print("=" * 90 + "\n")

    # Save to JSON
    out_file = e2e_dir / "similarity_report.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "overall": {
                    "mean_similarity_score": round(float(mean_sim), 2),
                    "mean_wer_percent": round(float(mean_wer), 2),
                    "mean_cer_percent": round(float(mean_cer), 2),
                },
                "samples": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info("Similarity report saved to %s", out_file)


if __name__ == "__main__":
    main()
