#!/usr/bin/env python3
"""Step 4 NPU -- Generate calibration + test activations for the 4 components.

Runs the fp32 (ORT) pipeline over Vietnamese calibration/test sentences and
captures each component's inputs, exactly like the official Qualcomm
calibration recipe. These npz files feed submit_quantize_job (w8a16) and
submit_inference_job (hardware test inputs).

Usage:
    python generate_component_calibration.py --output_dir outputs/piper_vi_npu
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from piper_components_pipeline import (  # noqa: E402
    ComponentRunner,
    generate_path_np,
    phonemize_vi,
    prepare_input,
)
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


def collect(comp: ComponentRunner, texts: list[str], id_map: dict) -> dict:
    """Run the pipeline for each text, collect per-component inputs."""
    enc_x, enc_xl = [], []
    sdp_xe, sdp_xm, sdp_ls, sdp_ns = [], [], [], []
    flow_mp, flow_lp, flow_ym, flow_at, flow_ns = [], [], [], [], []
    dec_z = []

    for text in texts:
        ids = phonemize_vi(text, id_map)
        x, xl = prepare_input(ids)
        x_encoded, m_p, logs_p, x_mask = comp.encoder(x, xl)

        y_lengths, w_ceil = comp.sdp(
            x_encoded, x_mask,
            np.array([1.0], np.float32), np.array([DEFAULT_NOISE_SCALE_W], np.float32))
        yl = int(y_lengths[0])
        y_mask = (np.arange(UPSAMPLED_MAX_SEQ_LEN) < yl)[None, None, :].astype(np.float32)
        attn_mask = x_mask[:, :, None, :] * y_mask[:, :, :, None]
        attn = generate_path_np(w_ceil, attn_mask)
        attn_sq = attn[:, 0, :, :].astype(np.float32)

        z = comp.flow(m_p, logs_p, y_mask, attn_sq,
                      np.array([DEFAULT_NOISE_SCALE], np.float32))

        # decoder windows (same loop as decode_chunks)
        zb = np.zeros((1, ENCODER_HIDDEN_DIM, DEC_SEQ_LEN), np.float32)
        zb[:, :, :(MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)] = z[:, :, :(MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)]
        dec_z.append(zb)
        total = MAX_DEC_SEQ_LEN
        while total < min(yl, z.shape[2] - MAX_DEC_SEQ_LEN - DEC_SEQ_OVERLAP):
            zb = z[:, :, total - DEC_SEQ_OVERLAP: total + MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP]
            dec_z.append(zb)
            total += MAX_DEC_SEQ_LEN

        enc_x.append(x); enc_xl.append(xl)
        sdp_xe.append(x_encoded); sdp_xm.append(x_mask)
        sdp_ls.append(np.array([1.0], np.float32)); sdp_ns.append(np.array([DEFAULT_NOISE_SCALE_W], np.float32))
        flow_mp.append(m_p); flow_lp.append(logs_p); flow_ym.append(y_mask)
        flow_at.append(attn_sq); flow_ns.append(np.array([DEFAULT_NOISE_SCALE], np.float32))

    return {
        "encoder": {"x": enc_x, "x_lengths": enc_xl},
        "sdp": {"x_encoded": sdp_xe, "x_mask": sdp_xm,
                "length_scale": sdp_ls, "noise_scale_w": sdp_ns},
        "flow": {"m_p": flow_mp, "logs_p": flow_lp, "y_mask": flow_ym,
                 "attn_squeezed": flow_at, "noise_scale": flow_ns},
        "decoder": {"z": dec_z},
    }


def save_npz_dict(path: Path, per_sample: dict):
    """Save {name: [arrays]} as one npz with per-sample stacked arrays."""
    save = {}
    for k, v in per_sample.items():
        save[k] = np.stack(v) if isinstance(v[0], np.ndarray) else np.array(v)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **save)
    logger.info("saved %s (%d samples, keys=%s)", path, len(v[0]), list(save.keys()))


def main():
    parser = argparse.ArgumentParser(description="Generate component calibration/test data")
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
        json.dump({"calib_texts": calib_texts, "test_texts": test_texts}, f,
                  indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
