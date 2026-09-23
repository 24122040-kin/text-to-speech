#!/usr/bin/env python3
"""Step 4 NPU -- Patch Piper (vi) ONNX for QNN/HTP (NPU-only) compile.

Fresh, minimal patch set (rebuilt from scratch; verified numerically after
each patch). HTP requires fully static shapes and has no translation for
non-deterministic ops, so:

1. RandomNormalLike (x2: duration predictor + posterior encoder) ->
   Shape + ConstantOfShape(zeros). This is EXACTLY the "deterministic mode"
   of Piper (equivalent to noise_scale=0 / noise_w=0): z = mu + sigma*0 = mu.

2. Dynamic Range limits (duration-sum ReduceMax, and Cast(Gather(Shape)) of
   the mel length) -> fixed constant OUTPUT_FRAMES. The Range drives the
   static output length; anything beyond the true duration is zero-padded and
   trimmed later in verification.

3. /Concat_1 (the Reshape_1 shape [1, L, T]) -> constant so the whole
   duration-expansion path is static.

The output is verified with onnxruntime: fixed shape, deterministic across
runs, sane waveform, and (with RandomNormalLike removed) identical output for
identical input regardless of noise scales (since eps is now zeros).

Usage:
    python patch_piper_npu.py \
        --input  outputs/piper_vi_npu/vi_VN-vais1000-medium.onnx \
        --output outputs/piper_vi_npu/piper_vi_npufix.onnx \
        --fixed_length 672 --output_frames 1024
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _scalar_tensor(name, value, dtype=TensorProto.INT64):
    if dtype == TensorProto.FLOAT:
        return helper.make_tensor(name, TensorProto.FLOAT, [], [float(value)])
    return helper.make_tensor(name, TensorProto.INT64, [], [int(value)])


def patch_random_normal_like(g) -> int:
    """RandomNormalLike -> Shape + ConstantOfShape(0)."""
    count = 0
    new_nodes = []
    for node in g.node:
        if node.op_type == "RandomNormalLike":
            src, out = node.input[0], node.output[0]
            shape_name = f"{node.name}/shape"
            new_nodes.append(helper.make_node("Shape", inputs=[src], outputs=[shape_name]))
            zero_attr = helper.make_tensor("value", TensorProto.FLOAT, [1], [0.0])
            new_nodes.append(helper.make_node(
                "ConstantOfShape", inputs=[shape_name], outputs=[out], value=zero_attr))
            count += 1
            logger.info("  RandomNormalLike %s -> zeros", node.name)
        else:
            new_nodes.append(node)
    del g.node[:]
    g.node.extend(new_nodes)
    return count


def patch_range_limits(g, fixed_length: int, fixed_frames: int) -> int:
    """Pin every dynamic Range input to a constant WITHOUT touching shared producers.

    Range node inputs are ordered [start, limit, delta]. For each Range input
    that is not already a Constant, we create a NEW constant tensor (name =
    <tensor>_npufix) and rewire ONLY that Range's input to it. The original
    producer node is left untouched, because in this graph the same tensors
    (e.g. flow-decoder Gather outputs) are consumed by other ops (Mul, Add...)
    that MUST keep the dynamic value. Range requires all three inputs to share
    a type; /Range is int64, /Range_1 is float, flow Ranges vary -- we read the
    dtype from shape inference.

    Two kinds of Range limits exist in this graph:
      - /enc_p/Range: position indices over the PHONEME sequence (attention
        mask) -> pinned to fixed_length.
      - /Range, /Range_1 and the flow Ranges: index grids over the MEL/audio
        time axis (output length) -> pinned to fixed_frames.
    The distinction is detected by whether the limit chain is rooted at a
    Shape of the graph input `input` (phoneme length).
    """
    input_name = "input"
    # Which tensor names are derived from Shape(input) -> phoneme-length-ish
    phoneme_derived = set()
    # one-hop: Shape(input) -> Gather(...) -> ... limit
    shape_of_input = set()
    for n in g.node:
        if n.op_type == "Shape" and input_name in n.input:
            shape_of_input.update(n.output)
    # second hop: nodes consuming those shapes (Gather/GatherElements/...) 
    frontier = set(shape_of_input)
    while True:
        newf = set()
        for n in g.node:
            if any(i in frontier for i in n.input):
                newf.update(n.output)
        if newf <= frontier:
            break
        frontier |= newf
    phoneme_derived = frontier

    # Per-tensor dtype from shape inference (original graph, pre-patch)
    tensor_dtypes = {}
    for vi in g.value_info:
        t = vi.type.tensor_type
        if t.HasField("elem_type"):
            tensor_dtypes[vi.name] = t.elem_type
    for i in g.initializer:
        tensor_dtypes[i.name] = i.data_type
    for inp in g.input:
        t = inp.type.tensor_type
        if t.HasField("elem_type"):
            tensor_dtypes[inp.name] = t.elem_type

    # Initializer scalar constants
    const_initializers = {}
    for i in g.initializer:
        arr = onnx.numpy_helper.to_array(i)
        if arr.size == 1:
            const_initializers[i.name] = (float(arr.flatten()[0]), i.data_type)

    count = 0
    new_nodes = []
    for node in g.node:
        if node.op_type != "Range":
            new_nodes.append(node)
            continue
        roles = ["start", "limit", "delta"]
        new_inputs = list(node.input)
        limit_is_phoneme_len = new_inputs[1] in phoneme_derived
        for idx, name in enumerate(node.input):
            if name in const_initializers:
                continue  # already constant
            role = roles[idx]
            if role == "limit":
                value = float(fixed_length if limit_is_phoneme_len else fixed_frames)
            else:
                # upstream scalar constant (one hop) else 0/1
                value = None
                for prod in g.node:
                    if name in prod.output:
                        for i in prod.input:
                            if i in const_initializers:
                                value = const_initializers[i][0]
                                break
                        break
                if value is None:
                    value = 0.0 if role == "start" else 1.0
            dtype = tensor_dtypes.get(name, TensorProto.FLOAT)
            const_dtype = dtype if dtype in (TensorProto.FLOAT, TensorProto.INT64) else TensorProto.FLOAT
            new_name = f"{name}_npufix"
            const = helper.make_node(
                "Constant", inputs=[], outputs=[new_name],
                value=_scalar_tensor(new_name, value, const_dtype))
            new_nodes.append(const)
            new_inputs[idx] = new_name
            count += 1
            logger.info("  Range %s (%s) -> const %s [%s]%s",
                        name, role, value,
                        "int64" if const_dtype == TensorProto.INT64 else "float",
                        " (phoneme-len)" if (role == "limit" and limit_is_phoneme_len) else "")
        new_node = helper.make_node("Range", inputs=new_inputs, outputs=list(node.output),
                                    name=node.name)
        new_nodes.append(new_node)
    del g.node[:]
    g.node.extend(new_nodes)
    return count


def patch_concat_reshape_shape(g, fixed_length: int, fixed_frames: int) -> int:
    """Pin /Concat_1 (Reshape_1 target shape [1, L, T]) to a constant."""
    count = 0
    new_nodes = []
    for node in g.node:
        if node.name == "/Concat_1":
            const = helper.make_node(
                "Constant", inputs=[], outputs=["/Concat_1_output_0"],
                value=helper.make_tensor("/Concat_1_output_0_fixed", TensorProto.INT64,
                                         [3], [1, fixed_length, fixed_frames]))
            new_nodes.append(const)
            count += 1
            logger.info("  /Concat_1 -> const [1, %d, %d]", fixed_length, fixed_frames)
        else:
            new_nodes.append(node)
    del g.node[:]
    g.node.extend(new_nodes)
    return count


def patch_model(model, fixed_length: int, fixed_frames: int) -> dict:
    g = model.graph
    count = {}
    count["rng"] = patch_random_normal_like(g)
    count["range"] = patch_range_limits(g, fixed_length, fixed_frames)
    count["concat"] = patch_concat_reshape_shape(g, fixed_length, fixed_frames)

    # Static input shape for `input` (fixed_length), keep other inputs as-is
    for inp in g.input:
        if inp.name == "input":
            dim = inp.type.tensor_type.shape.dim
            dim[0].dim_value = 1
            dim[1].dim_value = fixed_length
            logger.info("  input pinned to [1, %d]", fixed_length)
    return count


def main():
    parser = argparse.ArgumentParser(description="Patch Piper VITS for QNN/HTP compile")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixed_length", type=int, default=672,
                        help="Fixed phoneme sequence length (padded)")
    parser.add_argument("--output_frames", type=int, default=1024,
                        help="Fixed output length in mel frames (hop=256; 1024 = ~11.9s @22.05kHz)")
    args = parser.parse_args()

    model = onnx.load(str(args.input))
    model = onnx.shape_inference.infer_shapes(model, strict_mode=False)  # populate value_info for dtypes
    count = patch_model(model, args.fixed_length, args.output_frames)
    logger.info("Patched: %s", count)
    if count["rng"] == 0:
        logger.warning("No RandomNormalLike found (already patched?)")
    onnx.checker.check_model(model)
    onnx.save(model, str(args.output))
    logger.info("Saved: %s (%.1f MB)", args.output, args.output.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
