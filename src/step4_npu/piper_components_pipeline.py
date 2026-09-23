#!/usr/bin/env python3
"""Step 4 NPU -- Piper (vi) split-model synthesis (host glue + fp32 reference).

Implements the official Qualcomm PiperTTS orchestration for the 4 NPU
components, running components either via PyTorch (fp32 reference) or via
ONNX Runtime (fp32 reference / local sanity) -- later the SAME inputs feed the
compiled QNN binaries on IQ-9075.

All neural computation lives in the 4 components (encoder/sdp/flow/decoder);
this module only does non-neural glue: espeak phonemization, padding, the
argmax alignment (generate_path) and the sliding-window vocoder assembly.

Usage:
    python piper_components_pipeline.py --onnx_dir outputs/piper_vi_npu/components \
        --config outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx.json \
        --text 'Xin chào' --backend torch|ort
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

MAX_SEQ_LEN = 512
ENCODER_HIDDEN_DIM = 192
UPSAMPLED_MAX_SEQ_LEN = 1536
DEC_SEQ_OVERLAP = 12
MAX_DEC_SEQ_LEN = 40
DEC_SEQ_LEN = MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP
UPSAMPLE_FACTOR = 256
SAMPLE_RATE = 22050
DEFAULT_NOISE_SCALE = 0.667
DEFAULT_LENGTH_SCALE = 1.0
DEFAULT_NOISE_SCALE_W = 0.8


def phonemize_vi(text: str, id_map: dict) -> list[int]:
    """espeak-ng phonemize + phoneme->id (piper 1.6.1)."""
    from piper.phonemize_espeak import EspeakPhonemizer
    from piper.voice import phonemes_to_ids

    phonemizer = EspeakPhonemizer()
    phonemes = phonemizer.phonemize("vi", text)[0]
    return phonemes_to_ids(phonemes, id_map)


def prepare_input(phoneme_ids: list[int]):
    """Pad/truncate to MAX_SEQ_LEN, return (x int32, x_lengths int32)."""
    actual = min(len(phoneme_ids), MAX_SEQ_LEN)
    if len(phoneme_ids) > MAX_SEQ_LEN:
        phoneme_ids = phoneme_ids[:MAX_SEQ_LEN]
    else:
        phoneme_ids = phoneme_ids + [0] * (MAX_SEQ_LEN - len(phoneme_ids))
    return np.array([phoneme_ids], dtype=np.int32), np.array([actual], dtype=np.int32)


def generate_path_np(duration, mask):
    """generate_path (monotonic alignment) in numpy.

    Mirrors the torch reference exactly:
      cum = cumsum(duration) -> [b,1,t_x]; x = arange(t_y)
      path[b*t_x, t_y] = (x < cum)
      pad t_x axis by 1 at front, diff along t_x -> per-phoneme boundaries
      -> [b,1,t_y,t_x] * mask
    """
    b, _, t_y, t_x = mask.shape
    cum = np.cumsum(duration, axis=-1).reshape(b * t_x)
    x = np.arange(t_y, dtype=cum.dtype)
    path = (x[None, :] < cum[:, None]).astype(mask.dtype)  # [b*t_x, t_y]
    path = path.reshape(b, t_x, t_y)
    path_padded = np.pad(path, ((0, 0), (1, 0), (0, 0)))[:, :-1, :]
    path = path - path_padded
    return path[:, None, :, :].transpose(0, 1, 3, 2) * mask


class ComponentRunner:
    """Run the 4 components via torch or onnxruntime."""

    def __init__(self, onnx_dir: Path, backend: str = "ort"):
        self.onnx_dir = Path(onnx_dir)
        self.backend = backend
        if backend == "ort":
            import onnxruntime as ort
            self.sess = {
                "encoder": ort.InferenceSession(str(self.onnx_dir / "piper_vi_encoder.onnx"),
                                                providers=["CPUExecutionProvider"]),
                "sdp": ort.InferenceSession(str(self.onnx_dir / "piper_vi_sdp.onnx"),
                                            providers=["CPUExecutionProvider"]),
                "flow": ort.InferenceSession(str(self.onnx_dir / "piper_vi_flow.onnx"),
                                             providers=["CPUExecutionProvider"]),
                "decoder": ort.InferenceSession(str(self.onnx_dir / "piper_vi_decoder.onnx"),
                                                providers=["CPUExecutionProvider"]),
            }
        else:
            import torch
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from export_piper_components import (
                Decoder, Encoder, Flow, SDP, build_model_from_onnx,
            )
            self.torch = torch
            gen = build_model_from_onnx(
                str(self.onnx_dir.parent / "vi_VN-vais1000-medium.onnx"),
                str(self.onnx_dir.parent / "vi_VN-vais1000-medium.onnx.json"))
            self.enc = Encoder(gen)
            self.sdp = SDP(gen)
            self.flow = Flow(gen)
            self.dec = Decoder(gen)
            self.enc.eval(); self.sdp.eval(); self.flow.eval(); self.dec.eval()

    def _run(self, comp, feeds):
        if self.backend == "ort":
            return self.sess[comp].run(None, feeds)
        import torch
        tfeeds = {k: torch.from_numpy(v) for k, v in feeds.items()}
        with torch.no_grad():
            model = getattr(self, {"encoder": "enc", "sdp": "sdp", "flow": "flow",
                                   "decoder": "dec"}[comp])
            out = model(*tfeeds.values())
            if not isinstance(out, tuple):
                out = (out,)
            return [o.numpy() for o in out]

    def encoder(self, x, x_lengths):
        outs = self._run("encoder", {"x": x, "x_lengths": x_lengths})
        return outs[0], outs[1], outs[2], outs[3]

    def sdp(self, x_encoded, x_mask, length_scale, noise_scale_w):
        outs = self._run("sdp", {"x_encoded": x_encoded, "x_mask": x_mask,
                                 "length_scale": length_scale,
                                 "noise_scale_w": noise_scale_w})
        return outs[0], outs[1]

    def flow(self, m_p, logs_p, y_mask, attn_squeezed, noise_scale):
        outs = self._run("flow", {"m_p": m_p, "logs_p": logs_p, "y_mask": y_mask,
                                  "attn_squeezed": attn_squeezed,
                                  "noise_scale": noise_scale})
        return outs[0]

    def decoder(self, z_buf):
        outs = self._run("decoder", {"z": z_buf})
        return outs[0]


def synthesize(comp: ComponentRunner, phoneme_ids: list[int],
               noise_scale: float = DEFAULT_NOISE_SCALE,
               length_scale: float = DEFAULT_LENGTH_SCALE,
               noise_scale_w: float = DEFAULT_NOISE_SCALE_W) -> tuple:
    """Full pipeline; returns (audio_np, y_lengths, z, attn_squeezed, w_ceil)."""
    x, x_lengths = prepare_input(phoneme_ids)

    x_encoded, m_p, logs_p, x_mask = comp.encoder(x, x_lengths)

    y_lengths, w_ceil = comp.sdp(
        x_encoded, x_mask,
        np.array([length_scale], dtype=np.float32),
        np.array([noise_scale_w], dtype=np.float32),
    )

    y_lengths = int(y_lengths[0])
    y_mask = (np.arange(UPSAMPLED_MAX_SEQ_LEN) < y_lengths)[None, None, :].astype(np.float32)
    attn_mask = x_mask[:, :, None, :] * y_mask[:, :, :, None]
    attn = generate_path_np(w_ceil, attn_mask)
    attn_squeezed = attn[:, 0, :, :].astype(np.float32)  # [1, t_y, t_x]

    z = comp.flow(m_p, logs_p, y_mask, attn_squeezed,
                  np.array([noise_scale], dtype=np.float32))

    # sliding-window decode
    z_buf = np.zeros((1, ENCODER_HIDDEN_DIM, MAX_DEC_SEQ_LEN + 2 * DEC_SEQ_OVERLAP),
                     dtype=np.float32)
    z_buf[:, :, :(MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)] = z[:, :, :(MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP)]
    audio_chunk = comp.decoder(z_buf)
    audio = audio_chunk.squeeze()[:MAX_DEC_SEQ_LEN * UPSAMPLE_FACTOR]
    total = MAX_DEC_SEQ_LEN
    while total < min(y_lengths, z.shape[2] - MAX_DEC_SEQ_LEN - DEC_SEQ_OVERLAP):
        z_buf = z[:, :, total - DEC_SEQ_OVERLAP: total + MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP]
        audio_chunk = comp.decoder(z_buf)
        audio_chunk = audio_chunk.squeeze()[DEC_SEQ_OVERLAP * UPSAMPLE_FACTOR:
                                            (MAX_DEC_SEQ_LEN + DEC_SEQ_OVERLAP) * UPSAMPLE_FACTOR]
        audio = np.concatenate([audio, audio_chunk])
        total += MAX_DEC_SEQ_LEN

    audio = audio[:y_lengths * UPSAMPLE_FACTOR]
    return audio, y_lengths, z, attn_squeezed, w_ceil


def main():
    parser = argparse.ArgumentParser(description="Piper (vi) split-model synthesis")
    parser.add_argument("--onnx_dir", type=Path,
                        default=Path("outputs/piper_vi_npu/components"))
    parser.add_argument("--config", type=Path,
                        default=Path("outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx.json"))
    parser.add_argument("--text", default="Xin chào thế giới")
    parser.add_argument("--backend", default="ort", choices=["ort", "torch"])
    parser.add_argument("--out_wav", type=Path, default=Path("outputs/piper_vi_npu/test_synth.wav"))
    args = parser.parse_args()

    cfg = json.load(open(args.config, encoding="utf-8"))
    id_map = cfg["phoneme_id_map"]

    comp = ComponentRunner(args.onnx_dir, args.backend)
    phoneme_ids = phonemize_vi(args.text, id_map)
    logger.info("phonemes: %d", len(phoneme_ids))

    audio, y_lengths, *_ = synthesize(comp, phoneme_ids)
    logger.info("audio: %d samples (%.2fs), y_lengths=%d",
                len(audio), len(audio) / SAMPLE_RATE, y_lengths)

    import soundfile as sf
    args.out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.out_wav), audio.astype(np.float32), SAMPLE_RATE)
    logger.info("saved %s", args.out_wav)


if __name__ == "__main__":
    main()
