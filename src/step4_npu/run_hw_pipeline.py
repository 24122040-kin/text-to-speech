#!/usr/bin/env python3
"""Step 4 NPU -- Run the Piper (vi) 4-component pipeline ENTIRELY on the NPU.

Chains hardware inference jobs on Dragonwing IQ-9075 EVK so that every
intermediate activation comes from the NPU (encoder -> sdp -> flow ->
decoder), with only the non-neural alignment (generate_path) computed on the
host -- exactly the official Qualcomm PiperTTS orchestration.

Stages (each resumable by job id in outputs/piper_vi_npu/job_log_hw.json):
  enc : encoder(x, x_lengths)              -> x_encoded, m_p, logs_p, x_mask
  sdp : sdp(hw x_encoded, hw x_mask, ...)  -> y_lengths, w_ceil
  host: generate_path(hw w_ceil, ...)      -> attn_squeezed, y_mask
  flow: flow(hw m_p, logs_p, y_mask, attn) -> z
  dec : decoder(z windows)                 -> audio chunks

Usage:
    python run_hw_pipeline.py --stage enc | sdp | align | flow | dec | all
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

# --- Windows sandbox tempfile patches (see deploy_piper_components.py) ---
_orig_td = tempfile.TemporaryDirectory


class _SafeTemporaryDirectory(_orig_td):
    def cleanup(self):
        try:
            super().cleanup()
        except Exception:  # noqa: BLE001
            pass


tempfile.TemporaryDirectory = _SafeTemporaryDirectory

import uuid as _uuid


def _safe_mkdtemp(*args, **kwargs):
    import tempfile as _tf

    base = kwargs.pop("dir", None) or _tf.gettempdir()
    prefix = kwargs.pop("prefix", "tmp")
    suffix = kwargs.pop("suffix", "")
    d = os.path.join(base, f"{prefix}{_uuid.uuid4().hex[:10]}{suffix}")
    os.makedirs(d, exist_ok=False)
    return d


tempfile.mkdtemp = _safe_mkdtemp

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEVICE_NAME = "Dragonwing IQ-9075 EVK"
MAX_SEQ_LEN = 512
ENCODER_HIDDEN_DIM = 192
UPSAMPLED_MAX_SEQ_LEN = 1536
DEC_SEQ_OVERLAP = 12
MAX_DEC_SEQ_LEN = 40
DEC_SEQ_LEN = 64
UPSAMPLE_FACTOR = 256
SAMPLE_RATE = 22050
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_SCALE_W = 0.8

# QNN reorders the encoder's ONNX outputs to [m_p, logs_p, x_encoded, x_mask]
# (same order as the official qualcomm/PiperTTS-EN metadata.json).
ENC_OUT = {"m_p": 0, "logs_p": 1, "x_encoded": 2, "x_mask": 3}

sys.path.insert(0, str(Path(__file__).resolve().parent))
from piper_components_pipeline import generate_path_np  # noqa: E402


def _log(output_dir: Path, record: dict):
    p = output_dir / "job_log_hw.json"
    log = []
    if p.exists():
        try:
            log = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            log = []
    log.append(record)
    p.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")


def _is_success(status) -> bool:
    code = getattr(status, "code", None) or str(status)
    return str(code).lower() in ("completed", "success", "successful")


def _wait_job(job, output_dir: Path, tag: str):
    logger.info("waiting %s job %s ...", tag, job.job_id)
    job.wait()
    status = job.get_status()
    code = getattr(status, "code", None) or str(status)
    logger.info("%s status: %s", tag, code)
    if not _is_success(status):
        logger.error("%s job failed: %s", tag, job.url)
        try:
            job.download_job_logs(str(output_dir / f"hw_logs_{tag}"))
        except Exception:
            pass
        sys.exit(1)
    return job


def stage_enc(output_dir: Path):
    import qai_hub as hub

    model = output_dir / "components" / "piper_vi_encoder.iq9075.bin"
    data = np.load(output_dir / "calib" / "test_encoder.npz")
    inputs = {"x": [data["x"][i] for i in range(len(data["x"]))],
              "x_lengths": [data["x_lengths"][i] for i in range(len(data["x_lengths"]))]}
    job = hub.submit_inference_job(model=str(model), device=hub.Device(DEVICE_NAME),
                                   inputs=inputs, name="piper_vi_enc_hw")
    _log(output_dir, {"stage": "enc", "job_id": job.job_id, "url": job.url})
    job = _wait_job(job, output_dir, "enc")
    out_dir = output_dir / "hw" / "enc"; out_dir.mkdir(parents=True, exist_ok=True)
    job.download_output_data(str(out_dir))
    logger.info("encoder hw outputs -> %s", out_dir)


def stage_sdp(output_dir: Path):
    import qai_hub as hub

    model = output_dir / "components" / "piper_vi_sdp.iq9075.bin"
    enc = load_hw_outputs(output_dir / "hw" / "enc")
    # enc outputs are [m_p, logs_p, x_encoded, x_mask] after QNN reorder
    n = len(enc[0])
    inputs = {
        "x_encoded": enc[ENC_OUT["x_encoded"]],
        "x_mask": enc[ENC_OUT["x_mask"]],
        "length_scale": [np.array([DEFAULT_LENGTH_SCALE], np.float32)] * n,
        "noise_scale_w": [np.array([DEFAULT_NOISE_SCALE_W], np.float32)] * n,
    }
    job = hub.submit_inference_job(model=str(model), device=hub.Device(DEVICE_NAME),
                                   inputs=inputs, name="piper_vi_sdp_hw")
    _log(output_dir, {"stage": "sdp", "job_id": job.job_id, "url": job.url})
    job = _wait_job(job, output_dir, "sdp")
    out_dir = output_dir / "hw" / "sdp"; out_dir.mkdir(parents=True, exist_ok=True)
    job.download_output_data(str(out_dir))
    logger.info("sdp hw outputs -> %s", out_dir)


def stage_align(output_dir: Path):
    """Host-side alignment from HW w_ceil -> attn_squeezed + y_mask (npz)."""
    sdp = load_hw_outputs(output_dir / "hw" / "sdp")  # [y_lengths, w_ceil]
    enc = load_hw_outputs(output_dir / "hw" / "enc")
    y_lengths_list, w_ceil_list = sdp
    x_mask_list = enc[ENC_OUT["x_mask"]]
    attns, ymasks, yls = [], [], []
    for i in range(len(y_lengths_list)):
        yl = int(np.round(float(np.asarray(y_lengths_list[i]).flatten()[0])))
        yl = max(1, yl)
        wc = np.asarray(w_ceil_list[i]).reshape(1, 1, MAX_SEQ_LEN).astype(np.float32)
        xm = np.asarray(x_mask_list[i]).reshape(1, 1, MAX_SEQ_LEN).astype(np.float32)
        y_mask = (np.arange(UPSAMPLED_MAX_SEQ_LEN) < yl)[None, None, :].astype(np.float32)
        attn_mask = xm[:, :, None, :] * y_mask[:, :, :, None]
        attn = generate_path_np(wc, attn_mask)
        attns.append(attn[:, 0, :, :].astype(np.float32))
        ymasks.append(y_mask)
        yls.append(yl)
    out = output_dir / "hw" / "align"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "align.npz",
             attn_squeezed=np.stack(attns), y_mask=np.stack(ymasks),
             y_lengths=np.array(yls, dtype=np.int32))
    logger.info("alignment done: %d samples, y_lengths=%s", len(yls), yls)


def stage_flow(output_dir: Path):
    import qai_hub as hub

    model = output_dir / "components" / "piper_vi_flow.iq9075.bin"
    enc = load_hw_outputs(output_dir / "hw" / "enc")
    align = np.load(output_dir / "hw" / "align" / "align.npz")
    n = len(enc[0])
    inputs = {
        "m_p": enc[ENC_OUT["m_p"]],
        "logs_p": enc[ENC_OUT["logs_p"]],
        "y_mask": [align["y_mask"][i] for i in range(n)],
        "attn_squeezed": [align["attn_squeezed"][i] for i in range(n)],
        "noise_scale": [np.array([DEFAULT_NOISE_SCALE], np.float32)] * n,
    }
    job = hub.submit_inference_job(model=str(model), device=hub.Device(DEVICE_NAME),
                                   inputs=inputs, name="piper_vi_flow_hw")
    _log(output_dir, {"stage": "flow", "job_id": job.job_id, "url": job.url})
    job = _wait_job(job, output_dir, "flow")
    out_dir = output_dir / "hw" / "flow"; out_dir.mkdir(parents=True, exist_ok=True)
    job.download_output_data(str(out_dir))
    logger.info("flow hw outputs -> %s", out_dir)


def stage_dec(output_dir: Path):
    import qai_hub as hub

    model = output_dir / "components" / "piper_vi_decoder.iq9075.bin"
    flow = load_hw_outputs(output_dir / "hw" / "flow")  # [z] per sample
    align = np.load(output_dir / "hw" / "align" / "align.npz")
    yls = align["y_lengths"]
    z_windows, sample_ids = [], []
    for i, z_i in enumerate(flow[0]):
        z = np.asarray(z_i).reshape(1, ENCODER_HIDDEN_DIM, UPSAMPLED_MAX_SEQ_LEN).astype(np.float32)
        yl = int(yls[i])
        zb = np.zeros((1, ENCODER_HIDDEN_DIM, DEC_SEQ_LEN), np.float32)
        zb[:, :, :(MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)] = z[:, :, :(MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)]
        z_windows.append(zb); sample_ids.append(i)
        total = MAX_DEC_SEQ_LEN
        while total < min(yl, z.shape[2] - MAX_DEC_SEQ_LEN - DEC_SEQ_OVERLAP):
            zb = z[:, :, total - DEC_SEQ_OVERLAP: total + MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP]
            z_windows.append(zb); sample_ids.append(i)
            total += MAX_DEC_SEQ_LEN
    inputs = {"z": z_windows}
    job = hub.submit_inference_job(model=str(model), device=hub.Device(DEVICE_NAME),
                                   inputs=inputs, name="piper_vi_dec_hw")
    _log(output_dir, {"stage": "dec", "job_id": job.job_id, "url": job.url,
                      "num_windows": len(z_windows)})
    job = _wait_job(job, output_dir, "dec")
    out_dir = output_dir / "hw" / "dec"; out_dir.mkdir(parents=True, exist_ok=True)
    job.download_output_data(str(out_dir))
    np.save(out_dir / "sample_ids.npy", np.array(sample_ids))
    np.savez(out_dir / "z_windows.npz", z=np.stack(z_windows))
    logger.info("decoder hw outputs -> %s (%d windows)", out_dir, len(z_windows))


def load_hw_outputs(dir_path: Path):
    """Load downloaded inference outputs grouped by output index.

    qai-hub 0.54 downloads an .h5 dataset (data/<output_idx>/batch_<sample>)
    or, for older versions, individual .npy/.npz files. Returns a list of
    lists: output_index -> [sample_0, sample_1, ...].
    """
    h5_files = sorted(dir_path.glob("*.h5"), key=lambda p: p.stat().st_mtime)
    if h5_files:
        import h5py

        with h5py.File(h5_files[-1], "r") as h:  # newest (stale runs can leave old ones)
            groups = []
            for oidx in sorted(h["data"].keys(), key=int):
                g = h[f"data/{oidx}"]
                keys = sorted(g.keys(), key=lambda k: int(k.split("_")[1]))
                batches = [g[k][()] for k in keys]
                groups.append(batches)
        return groups

    files = sorted(dir_path.iterdir())
    groups = {}
    for f in files:
        if f.suffix == ".npy":
            stem = f.stem
            parts = stem.split("_")
            try:
                oidx = int(parts[0]) if len(parts) == 1 else int(parts[1])
            except ValueError:
                oidx = 0
            groups.setdefault(oidx, []).append((int(parts[-1]) if parts[-1].isdigit() else len(groups.get(oidx, [])), np.load(f)))
        elif f.suffix == ".npz":
            d = np.load(f)
            for k in d.files:
                groups.setdefault(0, []).append((0, d[k]))
    out = []
    for oidx in sorted(groups):
        items = sorted(groups[oidx], key=lambda x: x[0])
        out.append([it[1] for it in items])
    return out


def stage_all(output_dir: Path):
    stage_enc(output_dir)
    stage_sdp(output_dir)
    stage_align(output_dir)
    stage_flow(output_dir)
    stage_dec(output_dir)


def main():
    parser = argparse.ArgumentParser(description="Run Piper components on hardware")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/piper_vi_npu"))
    parser.add_argument("--stage", required=True,
                        choices=["enc", "sdp", "align", "flow", "dec", "all"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == "enc":
        stage_enc(args.output_dir)
    elif args.stage == "sdp":
        stage_sdp(args.output_dir)
    elif args.stage == "align":
        stage_align(args.output_dir)
    elif args.stage == "flow":
        stage_flow(args.output_dir)
    elif args.stage == "dec":
        stage_dec(args.output_dir)
    elif args.stage == "all":
        stage_all(args.output_dir)


if __name__ == "__main__":
    main()
