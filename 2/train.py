import argparse
import json
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from dataset import CIFAR10_CLASSES, get_cifar10_loaders
from model import count_parameters, resnet56


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        preds = outputs.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    error_pct = 100.0 * (1.0 - correct / total)
    return avg_loss, error_pct


@torch.no_grad()
def evaluate_per_class(model, loader, device):
    model.eval()
    class_correct = np.zeros(len(CIFAR10_CLASSES), dtype=np.int64)
    class_total = np.zeros(len(CIFAR10_CLASSES), dtype=np.int64)

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(images).argmax(dim=1)
        for c in range(len(CIFAR10_CLASSES)):
            mask = targets == c
            class_total[c] += mask.sum().item()
            class_correct[c] += (preds[mask] == targets[mask]).sum().item()

    return {
        CIFAR10_CLASSES[c]: 100.0 * class_correct[c] / class_total[c]
        for c in range(len(CIFAR10_CLASSES))
    }


def save_plots(history, output_path):
    iters = [h["iteration"] / 1e4 for h in history]
    train_err = [h["train_error"] for h in history]
    test_err = [h["test_error"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    test_loss = [h["test_loss"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(iters, train_err, linestyle="--", label="Train error (%)")
    axes[0].plot(iters, test_err, linewidth=2.0, label="Test error (%)")
    axes[0].axhline(6.97, color="gray", linestyle=":", linewidth=1.2, label="Paper ResNet-56 (6.97%)")
    axes[0].set_xlabel("Iteration (1e4)")
    axes[0].set_ylabel("Error (%)")
    axes[0].set_title("ResNet-56 on CIFAR-10: Error")
    axes[0].set_ylim(0, 25 if min(test_err) < 20 else min(100, max(test_err) + 5))
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(iters, train_loss, linestyle="--", label="Train loss")
    axes[1].plot(iters, test_loss, linewidth=2.0, label="Test loss")
    axes[1].set_xlabel("Iteration (1e4)")
    axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].set_title("ResNet-56 on CIFAR-10: Loss")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-iters", type=int, default=64000)
    parser.add_argument("--milestones", type=int, nargs="+", default=[32000, 48000])
    parser.add_argument("--shortcut", type=str, choices=["A", "B"], default="A")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./results")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, test_loader, _ = get_cifar10_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    model = resnet56(num_classes=10, shortcut_option=args.shortcut).to(device)
    num_params = count_parameters(model)
    print(f"Device: {device} (AMP: {use_amp})")
    print(f"ResNet-56 (Option {args.shortcut}) parameters: {num_params:,} ({num_params / 1e6:.2f}M)")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.milestones, gamma=0.1
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history = []
    best_test_error = float("inf")
    best_iter = 0
    iteration = 0
    epoch = 0

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        iteration = ckpt["iteration"]
        epoch = ckpt["epoch"]
        best_test_error = ckpt.get("best_test_error", float("inf"))
        best_iter = ckpt.get("best_iter", 0)
        history = ckpt.get("history", [])
        print(f"Resumed from {args.resume} at epoch {epoch}, iter {iteration}")

    start_time = time.time()

    while iteration < args.max_iters:
        epoch += 1
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            if iteration >= args.max_iters:
                break

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            iteration += 1
            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += batch_size

        train_loss = running_loss / total
        train_error = 100.0 * (1.0 - correct / total)
        test_loss, test_error = evaluate(model, test_loader, criterion, device)
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "iteration": iteration,
            "lr": current_lr,
            "train_loss": train_loss,
            "train_error": train_error,
            "test_loss": test_loss,
            "test_error": test_error,
        }
        history.append(record)

        if test_error < best_test_error:
            best_test_error = test_error
            best_iter = iteration
            torch.save(
                {
                    "iteration": iteration,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "test_error": test_error,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "resnet56_best.pt"),
            )

        if epoch == 1 or epoch % 10 == 0 or iteration >= args.max_iters:
            elapsed = time.time() - start_time
            print(
                f"Epoch {epoch:3d} | Iter {iteration:5d}/{args.max_iters} | "
                f"LR {current_lr:.4f} | "
                f"Train err {train_error:5.2f}% (loss {train_loss:.4f}) | "
                f"Test err {test_error:5.2f}% (loss {test_loss:.4f}) | "
                f"Best {best_test_error:5.2f}% | {elapsed:.1f}s",
                flush=True,
            )

        if epoch % 20 == 0:
            torch.save(
                {
                    "iteration": iteration,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_test_error": best_test_error,
                    "best_iter": best_iter,
                    "test_error": test_error,
                    "history": history,
                    "args": vars(args),
                },
                os.path.join(args.output_dir, "resnet56_last.pt"),
            )

    torch.save(
        {
            "iteration": iteration,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_test_error": best_test_error,
            "best_iter": best_iter,
            "test_error": history[-1]["test_error"],
            "history": history,
            "args": vars(args),
        },
        os.path.join(args.output_dir, "resnet56_last.pt"),
    )

    best_ckpt = torch.load(
        os.path.join(args.output_dir, "resnet56_best.pt"),
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_ckpt["model_state_dict"])
    per_class_acc = evaluate_per_class(model, test_loader, device)

    summary = {
        "parameters": num_params,
        "best_test_error": best_test_error,
        "best_test_accuracy": 100.0 - best_test_error,
        "best_iteration": best_iter,
        "final_test_error": history[-1]["test_error"],
        "final_test_accuracy": 100.0 - history[-1]["test_error"],
        "elapsed_seconds": time.time() - start_time,
        "per_class_accuracy": per_class_acc,
        "history": history,
    }

    with open(os.path.join(args.output_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    save_plots(history, os.path.join(args.output_dir, "training_curves.png"))

    print("-" * 80)
    print(
        f"Training finished in {summary['elapsed_seconds']:.1f}s | "
        f"Best test error: {best_test_error:.2f}% (acc {100.0 - best_test_error:.2f}%) at iter {best_iter}"
    )
    print(
        f"Final test error: {history[-1]['test_error']:.2f}% (acc {100.0 - history[-1]['test_error']:.2f}%)"
    )
    print("Per-class accuracy (best checkpoint):")
    for cls_name, acc in per_class_acc.items():
        print(f"  {cls_name:12s}: {acc:5.2f}%")


if __name__ == "__main__":
    main()
