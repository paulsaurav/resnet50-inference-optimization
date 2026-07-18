"""
Accuracy validation.

A speedup number is meaningless without an accuracy number next to it.
"3x faster" could mean "3x faster and completely broken." This module
measures top-1 accuracy on an ImageNet validation subset so every optimized
variant can be reported as (latency, accuracy) — the only honest way to
present the tradeoff.

Point --data at an ImageFolder-structured directory:
    val/
      n01440764/  *.JPEG
      n01443537/  *.JPEG
      ...
A 1000-image subset of ImageNet val is plenty for a portfolio project.
"""

import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision.models as models


def get_loader(data_dir, batch_size=64):
    # Standard ImageNet eval preprocessing.
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
    # shuffle=False so the image order matches the TensorRT validator exactly
    # when --max-batches is used. Both iterate ImageFolder in the same sorted
    # order, so the same --max-batches limit selects the same images, making
    # accuracy numbers directly comparable. drop_last=True keeps every batch
    # at full size, matching the fixed-batch TRT engines.
    return DataLoader(
        ds, batch_size=batch_size, num_workers=4,
        pin_memory=True, shuffle=False, drop_last=True,
    )


@torch.no_grad()
def top1_accuracy(model, loader, device, half=False, max_batches=None):
    model.eval()
    correct = total = 0
    for i, (images, labels) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = images.to(device)
        if half:
            images = images.half()
        labels = labels.to(device)
        out = model(images)
        pred = out.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="ImageFolder val dir")
    ap.add_argument("--model", default="resnet50")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="match the TRT validator (default 32)")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="limit batches to match a TRT subset run "
                         "(e.g. 100 -> first 100*bs images)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = get_loader(args.data, batch_size=args.batch_size)

    n = "full val set" if args.max_batches is None \
        else f"first {args.max_batches * args.batch_size} images"
    print(f"evaluating on {n}")

    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m = m.to(device)
    acc = top1_accuracy(m, loader, device, max_batches=args.max_batches)
    print(f"FP32 top-1 accuracy: {acc:.2f}%")

    acc_fp16 = top1_accuracy(m.half(), loader, device, half=True,
                             max_batches=args.max_batches)
    print(f"FP16 top-1 accuracy: {acc_fp16:.2f}%")
    print(f"accuracy delta (fp16 - fp32): {acc_fp16 - acc:+.2f} pts")


if __name__ == "__main__":
    main()