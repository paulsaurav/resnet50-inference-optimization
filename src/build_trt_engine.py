"""
Build TensorRT engines (FP16 and INT8) from the ResNet-50 ONNX export,
benchmark them, and save the engines for the accuracy step.

Pipeline: ONNX (from run_baseline.py)  ->  TRT engine  ->  timed inference.
This is the production-standard path: the ONNX file is a real, portable
deployment artifact, and TensorRT specializes it into a hardware-tuned engine.

FP16   : half precision. Big speedup on tensor cores, accuracy ~unchanged.
INT8   : 8-bit. Biggest speedup, but weights/activations must be *calibrated*
         on real images so TensorRT picks good quantization scales. Without
         calibration INT8 accuracy collapses, so we feed it a few hundred
         ImageNet images via an IInt8EntropyCalibrator2.

Written for TensorRT 10.x/11.x (build_serialized_network API).

Usage:
  # FP16 only (no data needed):
  python src/build_trt_engine.py --onnx models/resnet50.onnx --precision fp16

  # INT8 (needs a folder of ImageNet images for calibration):
  python src/build_trt_engine.py --onnx models/resnet50.onnx --precision int8 \
      --calib-dir imagenet/ILSVRC/Data/CLS-LOC/val --calib-images 512
"""

import os
import json
import argparse
from pathlib import Path

import numpy as np
import tensorrt as trt

# TensorRT needs an explicit CUDA memory API. pycuda is the usual choice;
# cuda-python also works. We use pycuda here.
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401  (initializes CUDA context on import)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# --------------------------------------------------------------------------
# INT8 calibrator: feeds batches of preprocessed images to TensorRT so it can
# measure activation ranges and choose INT8 scales.
# --------------------------------------------------------------------------
class ImageNetCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, image_dir, cache_file, batch_size=32, max_images=512):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size

        self.images = self._gather_images(image_dir, max_images)
        self.n_batches = len(self.images) // batch_size
        self.current = 0
        # Device buffer for one batch of NCHW float32 224x224.
        self.device_input = cuda.mem_alloc(
            batch_size * 3 * 224 * 224 * np.dtype(np.float32).itemsize
        )
        print(f"calibrator: {len(self.images)} images, {self.n_batches} batches")

    def _gather_images(self, image_dir, max_images):
        exts = {".jpeg", ".jpg", ".png"}
        paths = []
        for root, _, files in os.walk(image_dir):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    paths.append(os.path.join(root, f))
                    if len(paths) >= max_images:
                        return paths
        return paths

    def _preprocess(self, path):
        # Match the eval transform used everywhere else: resize 256,
        # center-crop 224, ImageNet mean/std normalization.
        from PIL import Image
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = Image.open(path).convert("RGB")
        # resize shorter side to 256
        w, h = img.size
        scale = 256 / min(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR)
        # center crop 224
        w, h = img.size
        left, top = (w - 224) // 2, (h - 224) // 2
        img = img.crop((left, top, left + 224, top + 224))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean) / std
        return arr.transpose(2, 0, 1)  # HWC -> CHW

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current >= self.n_batches:
            return None
        batch_paths = self.images[
            self.current * self.batch_size:(self.current + 1) * self.batch_size
        ]
        batch = np.ascontiguousarray(
            np.stack([self._preprocess(p) for p in batch_paths]).astype(np.float32)
        )
        cuda.memcpy_htod(self.device_input, batch)
        self.current += 1
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


# --------------------------------------------------------------------------
# Engine building
# --------------------------------------------------------------------------
def build_engine(onnx_path, precision, batch_size, calibrator=None):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2 GB

    # Fixed input shape via optimization profile (batch_size, 3, 224, 224).
    profile = builder.create_optimization_profile()
    inp = network.get_input(0)
    shape = (batch_size, 3, 224, 224)
    profile.set_shape(inp.name, shape, shape, shape)
    config.add_optimization_profile(profile)

    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # allow fp16 fallback
        config.int8_calibrator = calibrator
        config.set_calibration_profile(profile)

    print(f"building {precision} engine (bs={batch_size}) ... this can take minutes")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")
    return serialized


def benchmark_engine(serialized, batch_size, n_warmup=20, n_iters=200):
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()

    inp_name = engine.get_tensor_name(0)
    out_name = engine.get_tensor_name(1)
    context.set_input_shape(inp_name, (batch_size, 3, 224, 224))

    out_shape = tuple(context.get_tensor_shape(out_name))
    h_input = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
    h_input = np.ascontiguousarray(h_input)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(int(np.prod(out_shape)) * np.dtype(np.float32).itemsize)

    context.set_tensor_address(inp_name, int(d_input))
    context.set_tensor_address(out_name, int(d_output))
    stream = cuda.Stream()

    cuda.memcpy_htod(d_input, h_input)

    for _ in range(n_warmup):
        context.execute_async_v3(stream.handle)
    stream.synchronize()

    import time
    timings = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        context.execute_async_v3(stream.handle)
        stream.synchronize()
        timings.append((time.perf_counter() - t0) * 1000.0)

    timings.sort()
    p50 = float(np.percentile(timings, 50))
    p95 = float(np.percentile(timings, 95))
    p99 = float(np.percentile(timings, 99))
    mean = float(np.mean(timings))
    throughput = batch_size * 1000.0 / mean
    print(
        f"[tensorrt bs={batch_size:<3d}] "
        f"p50={p50:7.2f}ms  p95={p95:7.2f}ms  p99={p99:7.2f}ms  "
        f"{throughput:8.1f} img/s"
    )
    return dict(
        batch_size=batch_size, p50_ms=round(p50, 3), p95_ms=round(p95, 3),
        p99_ms=round(p99, 3), mean_ms=round(mean, 3),
        throughput_ips=round(throughput, 1),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default="models/resnet50.onnx")
    ap.add_argument("--precision", choices=["fp16", "int8"], required=True)
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--calib-dir", help="image dir for INT8 calibration")
    ap.add_argument("--calib-images", type=int, default=512)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    Path(args.outdir).mkdir(exist_ok=True)
    Path("models").mkdir(exist_ok=True)
    results = []

    for bs in args.batch_sizes:
        calibrator = None
        if args.precision == "int8":
            if not args.calib_dir:
                raise SystemExit("--calib-dir is required for INT8")
            calibrator = ImageNetCalibrator(
                args.calib_dir,
                cache_file=f"models/calib_bs{bs}.cache",
                batch_size=min(bs, 32),
                max_images=args.calib_images,
            )

        serialized = build_engine(args.onnx, args.precision, bs, calibrator)

        # Save engine for the accuracy step.
        engine_path = f"models/resnet50_{args.precision}_bs{bs}.engine"
        with open(engine_path, "wb") as f:
            f.write(serialized)
        print(f"saved {engine_path}")

        row = benchmark_engine(serialized, bs)
        row["name"] = f"tensorrt_{args.precision}"
        results.append(row)

    out = Path(args.outdir) / f"tensorrt_{args.precision}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()