import argparse
import logging

logging.basicConfig(level=logging.INFO)

from onnx import load_model, save_model
from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.transformers.fusion_options import FusionOptions

from lightglue_onnx.optim.onnx_model_lightglue import LightGlueOnnxModel

NUM_HEADS, HIDDEN_SIZE = 4, 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", type=str, default="weights/sift_lightglue.onnx", help="Path to LightGlue ONNX model."
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Path to output fused LightGlue ONNX model."
    )
    parser.add_argument(
        "--cpu", default=True , action="store_true", help="Whether to optimize for CPU."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    lightglue = load_model(args.input)
    optimizer = LightGlueOnnxModel(lightglue, NUM_HEADS, HIDDEN_SIZE)

    options = None
    if args.cpu:
        options = FusionOptions("unet")
        options.enable_packed_qkv = False

    optimizer.optimize(options)
    optimizer.get_fused_operator_statistics()

    output_path = args.output
    if output_path is None:
        output_path = args.input.replace(".onnx", "_fused.onnx")
        if args.cpu:
            output_path = output_path.replace(".onnx", "_cpu.onnx")

    optimizer.save_model_to_file(output_path)

    save_model(
        SymbolicShapeInference.infer_shapes(load_model(output_path), auto_merge=True),
        output_path,
    )

    if args.cpu:
        print("CPU does not support fp16. Skipping..")
        exit()

    optimizer.convert_float_to_float16(
        keep_io_types=True,
    )

    output_path = output_path.replace(".onnx", "_fp16.onnx")
    optimizer.save_model_to_file(output_path)

    save_model(
        SymbolicShapeInference.infer_shapes(load_model(output_path), auto_merge=True),
        output_path,
    )
