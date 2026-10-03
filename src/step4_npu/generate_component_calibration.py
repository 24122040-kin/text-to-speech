#!/usr/bin/env python3
"""Step 4 NPU -- Generate calibration + test activations for all 8 components.

Runs the NPU tensor pipeline over Vietnamese calibration/test sentences and
captures each component's inputs, matching the official Qualcomm calibration recipe.
Generates .npz calibration and test files for all 8 hardware-targeted components:
  1. byte_text_encoder
  2. encoder
  3. sdp
  4. monotonic_aligner
  5. flow
  6. decoder
  7. overlap_add
  8. audio_resampler

Usage:
    python generate_component_calibration.py --output_dir outputs/piper_vi_npu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from piper_components_pipeline import (  # noqa: E402
    ComponentRunner,
    phonemize_vi,
    prepare_input,
)
from byte_text_pipeline import text_to_byte_tensor  # noqa: E402
from alignment_pipeline import MonotonicAligner  # noqa: E402
from export_piper_components import (  # noqa: E402
    DEC_SEQ_OVERLAP,
    DEC_SEQ_LEN,
    DEFAULT_NOISE_SCALE,
    DEFAULT_NOISE_SCALE_W,
    MAX_DEC_SEQ_LEN,
    MAX_SEQ_LEN,
    UPSAMPLED_MAX_SEQ_LEN,
    ENCODER_HIDDEN_DIM,
)

CHUNK_AUDIO_LEN = 16384
OVERLAP_AUDIO_LEN = 3072
STRIDE_AUDIO_LEN = 10240


def collect(comp: ComponentRunner, texts: list[str], id_map: dict) -> dict:
    """Run the pipeline for each text, collect per-component inputs for all 8 NPU graphs."""
    aligner = MonotonicAligner()
    aligner.eval()

    byte_bi, byte_bl = [], []
    enc_x, enc_xl = [], []
    sdp_xe, sdp_xm, sdp_ls, sdp_ns = [], [], [], []
    align_wc, align_xm, align_yl = [], [], []
    flow_mp, flow_lp, flow_ym, flow_at, flow_ns = [], [], [], [], []
    dec_z = []
    ola_cc, ola_pt, ola_if = [], [], []
    resamp_in = []

    for text in texts:
        # 1. Byte Text Encoder inputs
        b_idx, b_len = text_to_byte_tensor(text, max_seq_len=MAX_SEQ_LEN)
        byte_bi.append(b_idx)
        byte_bl.append(b_len)

        # 2. Phoneme Encoder inputs & inference
        ids = phonemize_vi(text, id_map)
        x, xl = prepare_input(ids)
        x_encoded, m_p, logs_p, x_mask = comp.encoder(x, xl)

        # 3. SDP inputs & inference
        ls_arr = np.array([1.0], np.float32)
        nsw_arr = np.array([DEFAULT_NOISE_SCALE_W], np.float32)
        y_lengths, w_ceil = comp.sdp(x_encoded, x_mask, ls_arr, nsw_arr)
        yl_int = int(y_lengths[0])
        yl_arr = np.array([yl_int], dtype=np.int32)

        # 4. Monotonic Aligner inputs & tensor alignment (NPU vector operations)
        with torch.no_grad():
            w_t = torch.from_numpy(w_ceil.reshape(1, 1, MAX_SEQ_LEN).astype(np.float32))
            xm_t = torch.from_numpy(x_mask.reshape(1, 1, MAX_SEQ_LEN).astype(np.float32))
            yl_t = torch.tensor([yl_int], dtype=torch.int32)
            attn_t, y_mask_t = aligner(w_t, xm_t, yl_t)
            attn_sq = attn_t.numpy()
            y_mask = y_mask_t.numpy()

        align_wc.append(w_ceil.reshape(1, 1, MAX_SEQ_LEN).astype(np.float32))
        align_xm.append(x_mask.reshape(1, 1, MAX_SEQ_LEN).astype(np.float32))
        align_yl.append(yl_arr)

        # 5. Flow inputs & inference
        ns_arr = np.array([DEFAULT_NOISE_SCALE], np.float32)
        z = comp.flow(m_p, logs_p, y_mask, attn_sq, ns_arr)

        # 6. Decoder sliding windows & Vocoder inference
        first_len = min(MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP, yl_int)
        zb = np.zeros((1, ENCODER_HIDDEN_DIM, DEC_SEQ_LEN), np.float32)
        zb[:, :, :first_len] = z[:, :, :first_len]
        dec_z.append(zb)

        prev_t = np.zeros((1, 1, OVERLAP_AUDIO_LEN), dtype=np.float32)
        chunk_out = comp.decoder(zb).reshape(1, 1, CHUNK_AUDIO_LEN).astype(np.float32)

        # 7. Overlap-Add inputs (first chunk)
        ola_cc.append(chunk_out)
        ola_pt.append(prev_t)
        ola_if.append(np.array([1.0], dtype=np.float32))

        # 8. Resampler inputs (22.05 kHz -> 16 kHz)
        resamp_in.append(chunk_out[:, :, :STRIDE_AUDIO_LEN].astype(np.float32))

        # Save tail from first chunk
        prev_t = chunk_out[:, :, STRIDE_AUDIO_LEN : STRIDE_AUDIO_LEN + OVERLAP_AUDIO_LEN].copy()

        total = MAX_DEC_SEQ_LEN
        chunk_idx = 1
        while total < yl_int:
            start_f = total - DEC_SEQ_OVERLAP
            end_f = total + MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP
            actual_end = min(end_f, yl_int, z.shape[2])
            zb = np.zeros((1, ENCODER_HIDDEN_DIM, DEC_SEQ_LEN), np.float32)
            if start_f < z.shape[2] and actual_end > start_f:
                valid_span = actual_end - start_f
                zb[:, :, :valid_span] = z[:, :, start_f:actual_end]
            dec_z.append(zb)
            chunk_out = comp.decoder(zb).reshape(1, 1, CHUNK_AUDIO_LEN).astype(np.float32)

            ola_cc.append(chunk_out)
            ola_pt.append(prev_t)
            ola_if.append(np.array([0.0], dtype=np.float32))
            resamp_in.append(chunk_out[:, :, :STRIDE_AUDIO_LEN].astype(np.float32))

            # Update tail from subsequent chunk
            prev_t = chunk_out[:, :, STRIDE_AUDIO_LEN + OVERLAP_AUDIO_LEN : STRIDE_AUDIO_LEN + 2 * OVERLAP_AUDIO_LEN].copy()

            total += MAX_DEC_SEQ_LEN
            chunk_idx += 1

        enc_x.append(x)
        enc_xl.append(xl)
        sdp_xe.append(x_encoded)
        sdp_xm.append(x_mask)
        sdp_ls.append(ls_arr)
        sdp_ns.append(nsw_arr)
        flow_mp.append(m_p)
        flow_lp.append(logs_p)
        flow_ym.append(y_mask)
        flow_at.append(attn_sq)
        flow_ns.append(ns_arr)

    return {
        "byte_text_encoder": {"byte_indices": byte_bi, "byte_lengths": byte_bl},
        "encoder": {"x": enc_x, "x_lengths": enc_xl},
        "sdp": {
            "x_encoded": sdp_xe,
            "x_mask": sdp_xm,
            "length_scale": sdp_ls,
            "noise_scale_w": sdp_ns,
        },
        "monotonic_aligner": {
            "w_ceil": align_wc,
            "x_mask": align_xm,
            "y_lengths": align_yl,
        },
        "flow": {
            "m_p": flow_mp,
            "logs_p": flow_lp,
            "y_mask": flow_ym,
            "attn_squeezed": flow_at,
            "noise_scale": flow_ns,
        },
        "decoder": {"z": dec_z},
        "overlap_add": {
            "curr_chunk": ola_cc,
            "prev_tail": ola_pt,
            "is_first": ola_if,
        },
        "audio_resampler": {"audio_22050hz": resamp_in},
    }


def save_npz_dict(path: Path, per_sample: dict):
    """Save {name: [arrays]} as one npz with per-sample stacked arrays."""
    save = {}
    for k, v in per_sample.items():
        save[k] = np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save)
    logger.info("saved %s (%d items, keys=%s)", path, len(list(per_sample.values())[0]), list(save.keys()))


def main():
    parser = argparse.ArgumentParser(description="Generate component calibration/test data for all 8 components")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/piper_vi_npu"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--num_calib", type=int, default=64)
    parser.add_argument("--num_test", type=int, default=8)
    args = parser.parse_args()

    # texts: reuse data_stats test texts + calibration sentences
    data = np.load(args.output_dir / "piper_vi_npu_data.npz", allow_pickle=True)
    test_texts = [str(t) for t in data["test_texts"]][: args.num_test]
    calib_texts = [str(t) for t in data["calib_texts"]][: args.num_calib]

    cfg = json.load(open(args.output_dir / "vi_VN-vais1000-medium.onnx.json", encoding="utf-8"))
    id_map = cfg["phoneme_id_map"]

    comp = ComponentRunner(args.output_dir / "components", "ort")

    calib = collect(comp, calib_texts, id_map)
    for name, per_sample in calib.items():
        save_npz_dict(args.output_dir / "calib" / f"calib_{name}.npz", per_sample)

    test = collect(comp, test_texts, id_map)
    for name, per_sample in test.items():
        save_npz_dict(args.output_dir / "calib" / f"test_{name}.npz", per_sample)

    with open(args.output_dir / "calib" / "component_meta.json", "w", encoding="utf-8") as f:
        json.dump({"calib_texts": calib_texts, "test_texts": test_texts}, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
