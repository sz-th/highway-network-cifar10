import argparse
import math
import os
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms

from highway_network import (
    DeepFCHighwayNet,
    DeepFCPlainNet,
    build_highway11_cifar10,
    build_highway19_cifar10,
    count_parameters,
)


class TensorCIFAR10Dataset(Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


def get_cifar10_loaders(data_dir="./data", batch_size=128, val_fraction=0.1, num_workers=0, seed=42):
    already_extracted = os.path.isdir(os.path.join(data_dir, "cifar-10-batches-py"))
    raw_train = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=not already_extracted)
    raw_test = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=not already_extracted)

    indices = np.arange(len(raw_train.targets))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        stratify=raw_train.targets,
        random_state=seed,
    )

    x_train_raw = torch.from_numpy(raw_train.data[train_idx]).permute(0, 3, 1, 2).float().div_(255.0)
    y_train = torch.tensor(np.array(raw_train.targets)[train_idx], dtype=torch.long)

    x_val_raw = torch.from_numpy(raw_train.data[val_idx]).permute(0, 3, 1, 2).float().div_(255.0)
    y_val = torch.tensor(np.array(raw_train.targets)[val_idx], dtype=torch.long)

    x_test_raw = torch.from_numpy(raw_test.data).permute(0, 3, 1, 2).float().div_(255.0)
    y_test = torch.tensor(np.array(raw_test.targets), dtype=torch.long)

    train_mean = x_train_raw.mean(dim=(0, 2, 3), keepdim=True)
    train_std = x_train_raw.std(dim=(0, 2, 3), keepdim=True)

    x_train = (x_train_raw - train_mean) / train_std
    x_val = (x_val_raw - train_mean) / train_std
    x_test = (x_test_raw - train_mean) / train_std

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
    )

    train_set = TensorCIFAR10Dataset(x_train, y_train, transform=train_transform)
    val_set = TensorCIFAR10Dataset(x_val, y_val, transform=None)
    test_set = TensorCIFAR10Dataset(x_test, y_test, transform=None)

    pin_memory = torch.cuda.is_available()
    persistent = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    return train_loader, val_loader, test_loader


def compute_dummy_baseline(train_loader, val_loader, num_classes=10):
    y_train = train_loader.dataset.targets.numpy()
    y_val = val_loader.dataset.targets.numpy()
    counts = np.bincount(y_train, minlength=num_classes)
    majority_class = int(np.argmax(counts))
    dummy_preds = np.full_like(y_val, majority_class)
    dummy_acc = 100.0 * float((dummy_preds == y_val).mean())
    dummy_loss = math.log(num_classes)
    dummy_f1 = 100.0 * float(f1_score(y_val, dummy_preds, average="macro", zero_division=0))
    return {
        "majority_class": majority_class,
        "val_loss": dummy_loss,
        "val_acc": dummy_acc,
        "val_macro_f1": dummy_f1,
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_detailed=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        all_preds.append(predicted.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    total = len(y_true)
    avg_loss = running_loss / total
    correct_vec = (y_pred == y_true).astype(np.float64)
    acc = 100.0 * float(correct_vec.mean())

    if not return_detailed:
        return avg_loss, acc

    macro_prec = 100.0 * float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    macro_rec = 100.0 * float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    macro_f1 = 100.0 * float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)
    ci_low, ci_high = stats.t.interval(
        0.95, len(correct_vec) - 1, loc=correct_vec.mean(), scale=stats.sem(correct_vec)
    )

    return {
        "loss": avg_loss,
        "acc": acc,
        "ci_95": (100.0 * float(ci_low), 100.0 * float(ci_high)),
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
    }


@torch.no_grad()
def analyze_highway_gates(model, loader, device, save_path="highway_gates_analysis.png"):
    model.eval()
    hw_layers = model.get_highway_layers()
    if not hw_layers:
        return

    num_layers = len(hw_layers)
    gate_sums = [0.0 for _ in range(num_layers)]
    total_samples = 0
    single_gate = None
    single_out = None

    for batch_idx, (inputs, _) in enumerate(loader):
        inputs = inputs.to(device)
        _ = model(inputs)
        batch_size = inputs.size(0)
        total_samples += batch_size

        for i, layer in enumerate(hw_layers):
            g = layer.last_gate
            if g.dim() == 4:
                g_mean_spatial = g.mean(dim=(2, 3))
            else:
                g_mean_spatial = g
            gate_sums[i] = gate_sums[i] + g_mean_spatial.sum(dim=0).cpu().numpy()

        if batch_idx == 0:
            single_gate = []
            single_out = []
            for layer in hw_layers:
                g0 = layer.last_gate[0]
                o0 = layer.last_output[0]
                if g0.dim() == 3:
                    g0 = g0.mean(dim=(1, 2))
                    o0 = o0.mean(dim=(1, 2))
                single_gate.append(g0.cpu().numpy())
                single_out.append(o0.cpu().numpy())

        if total_samples >= 2000:
            break

    mean_gates = [g / total_samples for g in gate_sums]
    biases = [layer.T.bias.detach().cpu().numpy() for layer in hw_layers]

    max_blocks = max(len(b) for b in biases)

    def pad_to_matrix(arr_list):
        mat = np.full((len(arr_list), max_blocks), np.nan)
        for idx, arr in enumerate(arr_list):
            mat[idx, : len(arr)] = arr
        return mat

    bias_mat = pad_to_matrix(biases)
    mean_gate_mat = pad_to_matrix(mean_gates)
    single_gate_mat = pad_to_matrix(single_gate)
    single_out_mat = pad_to_matrix(single_out)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    titles = [
        "Transform Gate Biases",
        "Mean Transform Gate Outputs",
        "Single Sample Gate Outputs",
        "Single Sample Block Outputs",
    ]
    mats = [bias_mat, mean_gate_mat, single_gate_mat, single_out_mat]
    cmaps = ["viridis", "magma", "magma", "coolwarm"]

    for ax, title, mat, cmap in zip(axes, titles, mats, cmaps):
        im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xlabel("Channel / Block")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    axes[0].set_ylabel("Highway Layer Depth")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_training_history(history, dummy_acc=10.0, save_path="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.set_title("Loss Curve (Train vs Validation)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], label="Validation Accuracy")
    ax2.axhline(dummy_acc, color="gray", linestyle=":", linewidth=1.2, label=f"Dummy Baseline ({dummy_acc:.1f}%)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy Curve (Train vs Validation)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def build_model(arch, transform_bias, use_bn, fc_layers=19):
    if arch == "highway11":
        return build_highway11_cifar10(transform_bias=transform_bias, use_bn=use_bn)
    if arch == "highway19":
        return build_highway19_cifar10(transform_bias=transform_bias, use_bn=use_bn)
    if arch == "fc_highway":
        return DeepFCHighwayNet(
            input_dim=3072, hidden_dim=50, num_highway_layers=fc_layers, num_classes=10, transform_bias=transform_bias
        )
    if arch == "fc_plain":
        return DeepFCPlainNet(input_dim=3072, hidden_dim=71, num_hidden_layers=fc_layers, num_classes=10)
    raise ValueError(f"Unknown architecture: {arch}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--arch", type=str, default="highway11", choices=["highway11", "highway19", "fc_highway", "fc_plain"]
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--transform-bias", type=float, default=-2.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--fc-layers", type=int, default=19)
    parser.add_argument("--no-bn", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--save-path", type=str, default="best_highway_cifar10.pt")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, test_loader = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        num_workers=args.num_workers,
    )
    print(
        f"Split contract: {len(train_loader.dataset)} train / "
        f"{len(val_loader.dataset)} val (stratified) / "
        f"{len(test_loader.dataset)} test"
    )

    dummy = compute_dummy_baseline(train_loader, val_loader)
    print(
        f"Dummy baseline (majority class={dummy['majority_class']}): "
        f"Val Loss: {dummy['val_loss']:.4f} | Val Acc: {dummy['val_acc']:.2f}% | "
        f"Val Macro F1: {dummy['val_macro_f1']:.2f}%"
    )

    model = build_model(
        args.arch, transform_bias=args.transform_bias, use_bn=not args.no_bn, fc_layers=args.fc_layers
    ).to(device)
    print(f"Architecture: {args.arch} | Parameters: {count_parameters(model):,} | Transform Bias: {args.transform_bias}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        start_t = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, return_detailed=True)
        scheduler.step()
        elapsed = time.time() - start_t

        val_loss = val_metrics["loss"]
        val_acc = val_metrics["acc"]
        val_f1 = val_metrics["macro_f1"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "arch": args.arch,
                    "model_state_dict": model.state_dict(),
                    "val_acc": best_val_acc,
                    "val_macro_f1": val_f1,
                    "transform_bias": args.transform_bias,
                },
                args.save_path,
            )

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Val F1: {val_f1:.2f}% | "
            f"Best Val: {best_val_acc:.2f}% | Time: {elapsed:.1f}s"
        )

    plot_training_history(history, dummy_acc=dummy["val_acc"], save_path="training_curves.png")

    best_ckpt = torch.load(args.save_path, map_location=device, weights_only=True)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion, device, return_detailed=True)
    ci_lo, ci_hi = test_metrics["ci_95"]

    best_ckpt["test_acc"] = test_metrics["acc"]
    best_ckpt["test_loss"] = test_metrics["loss"]
    best_ckpt["test_ci_95"] = test_metrics["ci_95"]
    best_ckpt["test_macro_precision"] = test_metrics["macro_precision"]
    best_ckpt["test_macro_recall"] = test_metrics["macro_recall"]
    best_ckpt["test_macro_f1"] = test_metrics["macro_f1"]
    torch.save(best_ckpt, args.save_path)

    analyze_highway_gates(model, test_loader, device, save_path="highway_gates_analysis.png")

    print("\n--- Final Single Test Set Evaluation (Selected Best Val Epoch: " f"{best_epoch}) ---")
    print(f"Test Loss:            {test_metrics['loss']:.4f}")
    print(f"Test Accuracy:        {test_metrics['acc']:.2f}% (95% CI [{ci_lo:.2f}%, {ci_hi:.2f}%])")
    print(f"Test Macro Precision: {test_metrics['macro_precision']:.2f}%")
    print(f"Test Macro Recall:    {test_metrics['macro_recall']:.2f}%")
    print(f"Test Macro F1:        {test_metrics['macro_f1']:.2f}%")
    print("Confusion Matrix (10x10):")
    print(test_metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
