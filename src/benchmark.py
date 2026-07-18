import time
import statistics
from dataclasses import dataclass, asdict

import numpy as np
import torch


@dataclass
class LatencyResult:
    name: str
    batch_size: int
    device: str
    dtype: str
    n_iters: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_ips: float  # images per second

    def as_row(self):
        return asdict(self)


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def benchmark_torch(
    model,
    input_tensor,
    name,
    n_warmup=20,
    n_iters=200,
):
    """Benchmark a PyTorch model (eager or scripted)."""
    device = input_tensor.device
    model.eval()

    with torch.no_grad():
        # Warm-up: discard these entirely.
        for _ in range(n_warmup):
            _ = model(input_tensor)
        _sync(device)

        # Timed runs. Time each iteration individually so we get a
        # distribution, not just a total.
        timings = []
        for _ in range(n_iters):
            _sync(device)
            t0 = time.perf_counter()
            _ = model(input_tensor)
            _sync(device)
            t1 = time.perf_counter()
            timings.append((t1 - t0) * 1000.0)  # ms

    return _summarize(timings, name, input_tensor, n_iters)


def benchmark_ort(
    session,
    input_numpy,
    input_name,
    name,
    n_warmup=20,
    n_iters=200,
):
    """Benchmark an ONNX Runtime InferenceSession."""
    for _ in range(n_warmup):
        _ = session.run(None, {input_name: input_numpy})

    timings = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: input_numpy})
        t1 = time.perf_counter()
        timings.append((t1 - t0) * 1000.0)

    batch_size = input_numpy.shape[0]
    return _summarize_raw(timings, name, batch_size, "cuda/ort", "varies", n_iters)


def _summarize(timings, name, input_tensor, n_iters):
    batch_size = input_tensor.shape[0]
    device = input_tensor.device.type
    dtype = str(input_tensor.dtype).replace("torch.", "")
    return _summarize_raw(timings, name, batch_size, device, dtype, n_iters)


def _summarize_raw(timings, name, batch_size, device, dtype, n_iters):
    timings_sorted = sorted(timings)
    p50 = np.percentile(timings_sorted, 50)
    p95 = np.percentile(timings_sorted, 95)
    p99 = np.percentile(timings_sorted, 99)
    mean = statistics.mean(timings_sorted)
    # Throughput uses mean per-batch latency across the whole batch.
    throughput = (batch_size * 1000.0) / mean

    result = LatencyResult(
        name=name,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
        n_iters=n_iters,
        p50_ms=round(p50, 3),
        p95_ms=round(p95, 3),
        p99_ms=round(p99, 3),
        mean_ms=round(mean, 3),
        throughput_ips=round(throughput, 1),
    )
    print(
        f"[{name:28s}] bs={batch_size:<3d} "
        f"p50={p50:7.2f}ms  p95={p95:7.2f}ms  p99={p99:7.2f}ms  "
        f"{throughput:8.1f} img/s"
    )
    return result