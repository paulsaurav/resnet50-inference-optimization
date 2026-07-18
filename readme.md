# ResNet-50 inference optimization

Reducing ResNet-50 inference latency on an NVIDIA GPU by applying, and separately
measuring, four optimizations: FP16, ONNX Runtime, TensorRT FP16, and TensorRT
INT8. Each is benchmarked against the same FP32 PyTorch baseline, and every
latency number is paired with a top-1 accuracy number measured on the same
images, so the speed/accuracy tradeoff is explicit rather than assumed.

Best result: **TensorRT INT8 is 11.6x faster than FP32 at batch size 1
(6.87 ms to 0.59 ms), for a 0.28-point drop in top-1 accuracy.**

## Results

Hardware: NVIDIA GPU (Ampere-class), CUDA 12.1, TensorRT 10.13, PyTorch (CUDA
12.1 build). 200 timed iterations per measurement, 20 warm-up iterations
discarded. Latency is per-batch wall-clock time with `torch.cuda.synchronize()`
around each timed region.

### Batch-1 latency (single-request)

| Variant       | p50 (ms) | p95 (ms) | Speedup vs FP32 | Top-1 |
|---------------|---------:|---------:|----------------:|------:|
| PyTorch FP32  |     6.87 |     7.07 |            1.0x  | 85.25% |
| PyTorch FP16  |     7.92 |     8.08 |            0.87x | 85.25% |
| ONNX Runtime  |     2.67 |     2.72 |            2.6x  | -      |
| TensorRT FP16 |     0.74 |     5.71 |            9.3x  | 85.25% |
| TensorRT INT8 |     0.59 |     2.62 |           11.6x  | 84.97% |

### Batch-32 throughput

| Variant       | p50 (ms) | Throughput (img/s) | vs FP32 |
|---------------|---------:|-------------------:|--------:|
| PyTorch FP32  |    29.11 |               1068 |   1.0x  |
| PyTorch FP16  |    15.10 |               2091 |   2.0x  |
| ONNX Runtime  |    32.27 |                975 |   0.9x  |
| TensorRT FP16 |     6.28 |               5118 |   4.8x  |
| TensorRT INT8 |     3.00 |              10021 |   9.4x  |

Accuracy was measured on a fixed 3,200-image slice of the ImageNet validation
set (the first 100 batches of 32, in sorted order), identical across every
variant. The absolute figure runs higher than the canonical ResNet-50 number
(~80.3% on the full 50k set) because this slice is easier; the relative gap
between variants is what the accuracy column is for, and that comparison is
valid because every variant saw the same images.

## What each step does

FP16 casts weights and activations to half precision. On this GPU it does
nothing for single-request latency. At batch 1 the model is latency-bound, not
compute-bound, so the tensor cores sit idle between layers and the half-precision
cast is pure overhead (7.92 ms vs 6.87 ms). Under load it pays off: at batch 32
it roughly doubles throughput (2091 vs 1068 img/s). Worth knowing before reaching
for FP16 as a latency fix, because it isn't one.

ONNX Runtime exports the model to the ONNX graph format and runs it through a
leaner engine with operator fusion. It cuts batch-1 latency 2.6x with no
precision change and no accuracy cost. The win is almost entirely per-call
overhead reduction, which is why it shrinks at larger batches. At batch 32 it is
actually slower than PyTorch (975 vs 1068 img/s), because the fixed overhead it
removes is amortized across the batch anyway.

TensorRT compiles the ONNX graph into a hardware-specific engine, selecting and
fusing kernels for this exact GPU. FP16 here gets a 9.3x batch-1 speedup at no
accuracy cost. The same precision change that hurt in eager PyTorch helps
enormously once the surrounding graph is compiled to match.

TensorRT INT8 quantizes to 8-bit, calibrated on 512 ImageNet images so the
quantization scales match the real activation distribution. It is the fastest
variant (11.6x at batch 1, 9.4x throughput at batch 32) and costs 0.28 points of
top-1 accuracy. That tradeoff, an order-of-magnitude speedup for a fraction of a
percent, is the headline finding.

## Measurement notes

The things that make the numbers trustworthy, and the mistakes worth flagging.

Timing uses `torch.cuda.synchronize()` before and after each timed region.
CUDA kernels launch asynchronously; without synchronization you measure the time
to queue the work, not to run it, which produces latency numbers that are wrong
by an order of magnitude and always too low.

Latency is reported as percentiles, not a mean. p50 is the headline; the tail
matters because production latency targets are written against p95/p99. At batch
1, the TensorRT p95 (5.71 ms for FP16) is far above its p50 (0.74 ms). That gap
is real but not a property of the model. At sub-millisecond kernel times,
host-side scheduling jitter and GPU clock ramp dominate the tail. The p50 is the
honest single-request figure; the p95 at this scale is mostly noise, and more
iterations tighten it.

Accuracy nearly went in wrong. The first TensorRT accuracy run reported 85.25%,
higher than the supposed 80.3% FP32 baseline, which is impossible for a lossless
precision change. The cause: the TensorRT validator ran on the first 3,200 images
while the baseline ran on all 50,000, so the two numbers described different
image sets. Re-running the baseline on the same 3,200-image slice brought it to
85.25%, matching TensorRT FP16 exactly and confirming FP16 is lossless here. Any
accuracy comparison across variants has to hold the evaluation set fixed, or the
comparison is meaningless.

## Reproduce

Install torch/torchvision for your CUDA version (see pytorch.org), then:

```
pip install -r requirements.txt
```

TensorRT INT8 calibration here uses the implicit-calibration API, which requires
TensorRT 10.x. TensorRT 11.x removed it in favor of explicit QDQ quantization via
ModelOpt:

```
pip install "tensorrt-cu12==10.13.3.9"
```

Get the ImageNet validation set (Kaggle `imagenet-object-localization-challenge`),
then sort the flat val images into per-class WNID folders:

```
python src/prepare_imagenet_val.py \
    --val-dir imagenet/ILSVRC/Data/CLS-LOC/val \
    --csv imagenet/LOC_val_solution.csv
```

Run the pipeline:

```
python src/run_baseline.py --model resnet50 --batch-sizes 1 8 32
python src/build_trt_engine.py --onnx models/resnet50.onnx --precision fp16 --batch-sizes 1 8 32
python src/build_trt_engine.py --onnx models/resnet50.onnx --precision int8 --batch-sizes 1 8 32 \
    --calib-dir imagenet/ILSVRC/Data/CLS-LOC/val --calib-images 512
```

Measure accuracy (use the same `--max-batches` on all of them for comparable
numbers):

```
python src/validate_accuracy.py --data imagenet/ILSVRC/Data/CLS-LOC/val --batch-size 32 --max-batches 100
python src/validate_trt_accuracy.py --engine models/resnet50_fp16_bs32.engine \
    --data imagenet/ILSVRC/Data/CLS-LOC/val --batch-size 32 --max-batches 100
python src/validate_trt_accuracy.py --engine models/resnet50_int8_bs32.engine \
    --data imagenet/ILSVRC/Data/CLS-LOC/val --batch-size 32 --max-batches 100
```

Generate the charts:

```
python src/make_charts.py
```

## Files

```
src/benchmark.py             timing harness: warm-up, cuda sync, percentiles
src/run_baseline.py          FP32/FP16 baseline, ONNX export, ONNX Runtime bench
src/build_trt_engine.py      builds and benchmarks TensorRT FP16/INT8 engines
src/prepare_imagenet_val.py  sorts flat ImageNet val images into WNID folders
src/validate_accuracy.py     top-1 accuracy for PyTorch FP32/FP16
src/validate_trt_accuracy.py top-1 accuracy for a saved TensorRT engine
src/make_charts.py           renders result JSON into PNG charts
results/                     benchmark JSON and charts
models/                      exported ONNX and built TRT engines
```

## Known limitations

The batch-1 p95 numbers are noisy and should not be read as tail-latency
guarantees. They need a CUDA-events timing path and more iterations to be
meaningful at sub-millisecond scale.

Accuracy is measured on a 3,200-image subset for speed. The relative comparison
between variants is valid, but the absolute numbers are not the canonical
full-set figures. Running every accuracy command without `--max-batches` gives
the full-set numbers at the cost of a longer run.

The INT8 path uses TensorRT 10.x implicit calibration, which is deprecated. The
current-practice equivalent on TensorRT 11.x is explicit QDQ quantization via
NVIDIA ModelOpt, which inserts quantization nodes into the ONNX graph before the
engine is built. That migration is not done here.