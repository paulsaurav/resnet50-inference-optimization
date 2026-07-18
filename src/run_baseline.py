"""
Steps 1-2 of the project: FP32 baseline and ONNX export.

Run this first on your GPU machine. It produces:
  - The FP32 eager-mode baseline (your reference number).
  - An FP16 variant (usually a big, nearly-free win on GPU).
  - An ONNX export + ONNX Runtime CUDA benchmark.
  - results/baseline.json with everything, for the report table.

Usage:
  python src/run_baseline.py --model resnet50 --batch-sizes 1 8 32
"""

import json
import argparse
from pathlib import Path

import torch
import numpy as np
import torchvision.models as models

from benchmark import benchmark_torch, benchmark_ort


def get_model(name):
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"unknown model {name}")
    return m.eval()


def export_onnx(model, sample, path):
    torch.onnx.export(
        model,
        sample,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=17,
    )
    print(f"exported ONNX -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "This project targets a local NVIDIA GPU."
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = get_model(args.model).to(device)
    Path(args.outdir).mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)

    all_results = []

    for bs in args.batch_sizes:
        x = torch.randn(bs, 3, 224, 224, device=device)

        # 1. FP32 eager baseline.
        all_results.append(
            benchmark_torch(model, x, f"pytorch_fp32").as_row()
        )

        # 2. FP16 — cast model and input to half precision.
        model_fp16 = model.half()
        x_fp16 = x.half()
        all_results.append(
            benchmark_torch(model_fp16, x_fp16, f"pytorch_fp16").as_row()
        )
        model = model.float()  # cast back for ONNX export

    # 3. ONNX export (fixed sample) + ONNX Runtime CUDA.
    onnx_path = f"models/{args.model}.onnx"
    export_onnx(model, torch.randn(1, 3, 224, 224, device=device), onnx_path)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(
            onnx_path, providers=["CUDAExecutionProvider"]
        )
        in_name = sess.get_inputs()[0].name
        for bs in args.batch_sizes:
            xn = np.random.randn(bs, 3, 224, 224).astype(np.float32)
            all_results.append(
                benchmark_ort(sess, xn, in_name, "onnxruntime_fp32").as_row()
            )
    except ImportError:
        print("onnxruntime-gpu not installed; skipping ORT benchmark.")

    out = Path(args.outdir) / "baseline.json"
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()