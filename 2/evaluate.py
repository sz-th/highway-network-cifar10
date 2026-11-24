import argparse
import numpy as np
import torch
import torch.nn as nn

from dataset import CIFAR10_CLASSES, get_cifar10_loaders
from model import count_parameters, resnet56
from train import evaluate, evaluate_per_class


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="./results/resnet56_best.pt")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--shortcut", type=str, choices=["A", "B"], default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    shortcut = args.shortcut or ckpt.get("args", {}).get("shortcut", "A")

    model = resnet56(num_classes=10, shortcut_option=shortcut).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    _, test_loader, _ = get_cifar10_loaders(data_dir=args.data_dir)
    criterion = nn.CrossEntropyLoss()

    test_loss, test_error = evaluate(model, test_loader, criterion, device)
    per_class = evaluate_per_class(model, test_loader, device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"ResNet-56 (Option {shortcut}) parameters: {count_parameters(model):,}")
    print(f"Test loss: {test_loss:.4f} | Test error: {test_error:.2f}% | Test accuracy: {100.0 - test_error:.2f}%")
    print("Per-class accuracy:")
    for name in CIFAR10_CLASSES:
        print(f"  {name:12s}: {per_class[name]:5.2f}%")


if __name__ == "__main__":
    main()
