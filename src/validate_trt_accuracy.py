"""
Measure top-1 accuracy of a saved TensorRT engine on ImageNet val.

Completes the tradeoff table: every latency number from build_trt_engine.py
gets its accuracy partner here. Most important for INT8, where quantization
actually costs accuracy and the whole point is measuring how much.

We reuse the SAME preprocessing and the SAME ImageFolder label mapping as
validate_accuracy.py, so the TRT numbers are directly comparable to the
PyTorch FP32 baseline (80.35%).

Usage:
  python src/validate_trt_accuracy.py \
      --engine models/resnet50_int8_bs32.engine \
      --data imagenet/ILSVRC/Data/CLS-LOC/val \
      --batch-size 32 --max-batches 100
"""

import argparse

import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(path):
    runtime = trt.Runtime(TRT_LOGGER)
    with open(path, "rb") as f:
        return runtime.deserialize_cuda_engine(f.read())


def get_loader(data_dir, batch_size):
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    ds = datasets.ImageFolder(data_dir, tf)
    # drop_last so every batch matches the engine's fixed batch size.
    return DataLoader(
        ds, batch_size=batch_size, num_workers=4,
        pin_memory=True, drop_last=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, required=True,
                    help="MUST match the batch size the engine was built for")
    ap.add_argument("--max-batches", type=int, default=100,
                    help="limit for speed; 100 x bs images is plenty")
    args = ap.parse_args()

    engine = load_engine(args.engine)
    context = engine.create_execution_context()
    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)

    bs = args.batch_size
    context.set_input_shape(inp_name, (bs, 3, 224, 224))
    out_shape = tuple(context.get_tensor_shape(out_name))

    d_input = cuda.mem_alloc(bs * 3 * 224 * 224 * 4)
    d_output = cuda.mem_alloc(int(np.prod(out_shape)) * 4)
    context.set_tensor_address(inp_name, int(d_input))
    context.set_tensor_address(out_name, int(d_output))
    stream = cuda.Stream()

    loader = get_loader(args.data, bs)

    correct = total = 0
    for i, (images, labels) in enumerate(loader):
        if i >= args.max_batches:
            break
        batch = np.ascontiguousarray(images.numpy().astype(np.float32))
        cuda.memcpy_htod_async(d_input, batch, stream)
        context.execute_async_v3(stream.handle)
        host_out = np.empty(out_shape, dtype=np.float32)
        cuda.memcpy_dtoh_async(host_out, d_output, stream)
        stream.synchronize()

        preds = host_out.argmax(axis=1)
        correct += (preds == labels.numpy()).sum()
        total += bs

    acc = 100.0 * correct / total
    print(f"engine: {args.engine}")
    print(f"top-1 accuracy over {total} images: {acc:.2f}%")


if __name__ == "__main__":
    main()