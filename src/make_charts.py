import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed, just write files
import matplotlib.pyplot as plt


RESULTS = Path("results")

# Friendly display names + a stable color per variant.
DISPLAY = {
    "pytorch_fp32":     ("PyTorch FP32", "#4C72B0"),
    "pytorch_fp16":     ("PyTorch FP16", "#55A868"),
    "onnxruntime_fp32": ("ONNX Runtime", "#C44E52"),
    "tensorrt_fp16":    ("TensorRT FP16", "#8172B3"),
    "tensorrt_int8":    ("TensorRT INT8", "#CCB974"),
}
# Order variants left-to-right from slowest to fastest for readability.
ORDER = ["pytorch_fp32", "pytorch_fp16", "onnxruntime_fp32",
         "tensorrt_fp16", "tensorrt_int8"]


def load_all():
    """Return {name: {batch_size: row}} across all result files."""
    rows = []
    for fname in ["baseline.json", "tensorrt_fp16.json", "tensorrt_int8.json"]:
        p = RESULTS / fname
        if p.exists():
            rows.extend(json.loads(p.read_text()))
        else:
            print(f"note: {p} not found, skipping")

    data = {}
    for r in rows:
        data.setdefault(r["name"], {})[r["batch_size"]] = r
    return data


def bar_chart(data, batch_size, metric, ylabel, title, outfile, annotate_fmt):
    names, values, colors = [], [], []
    for name in ORDER:
        if name in data and batch_size in data[name]:
            label, color = DISPLAY[name]
            names.append(label)
            values.append(data[name][batch_size][metric])
            colors.append(color)

    if not names:
        print(f"no data for {title}, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, edgecolor="black", linewidth=0.6)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    for b, v in zip(bars, values):
        ax.annotate(annotate_fmt.format(v),
                    (b.get_x() + b.get_width() / 2, v),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(RESULTS / outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {RESULTS / outfile}")


def line_chart(data, outfile):
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = False
    for name in ORDER:
        if name not in data:
            continue
        label, color = DISPLAY[name]
        batches = sorted(data[name].keys())
        lat = [data[name][b]["p50_ms"] for b in batches]
        ax.plot(batches, lat, marker="o", label=label, color=color, linewidth=2)
        plotted = True

    if not plotted:
        return

    ax.set_xlabel("Batch size")
    ax.set_ylabel("p50 latency (ms)")
    ax.set_title("Latency vs batch size", fontweight="bold")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 8, 32])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / outfile, dpi=150)
    plt.close(fig)
    print(f"wrote {RESULTS / outfile}")


def main():
    data = load_all()
    if not data:
        raise SystemExit("no result JSON files found in results/")

    bar_chart(
        data, batch_size=1, metric="p50_ms",
        ylabel="p50 latency (ms)  — lower is better",
        title="Batch-1 inference latency (ResNet-50)",
        outfile="latency_bs1.png", annotate_fmt="{:.2f} ms",
    )
    bar_chart(
        data, batch_size=32, metric="throughput_ips",
        ylabel="throughput (img/s)  — higher is better",
        title="Batch-32 throughput (ResNet-50)",
        outfile="throughput_bs32.png", annotate_fmt="{:.0f}",
    )
    line_chart(data, "latency_vs_batch.png")


if __name__ == "__main__":
    main()