import math
import os
import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from highway_network import (
    DeepFCHighwayNet,
    DeepFCPlainNet,
    HighwayConv2d,
    HighwayLinear,
    PlainTransitionConv2d,
    build_highway11_cifar10,
    build_highway19_cifar10,
)
from train_cifar10 import compute_dummy_baseline, evaluate, get_cifar10_loaders


class TestHighwayNetwork(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_highway_linear_shapes_and_gate_bounds(self):
        layer = HighwayLinear(features=64, transform_bias=-2.0).to(self.device)
        layer.eval()
        x = torch.randn(16, 64, device=self.device)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(layer.last_gate)
        self.assertEqual(layer.last_gate.shape, x.shape)
        self.assertTrue(torch.all(layer.last_gate > 0.0) and torch.all(layer.last_gate < 1.0))

    def test_highway_conv2d_shapes_and_zero_padding(self):
        layer = HighwayConv2d(channels=32, kernel_size=3, transform_bias=-2.0).to(self.device)
        layer.eval()
        x = torch.randn(8, 32, 32, 32, device=self.device)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)
        self.assertIsNotNone(layer.last_gate)
        self.assertEqual(layer.last_gate.shape, x.shape)
        self.assertTrue(torch.all(layer.last_gate > 0.0) and torch.all(layer.last_gate < 1.0))

    def test_equation_4_carry_and_transform_regimes(self):
        x = torch.randn(10, 32, device=self.device)

        carry_layer = HighwayLinear(features=32, transform_bias=-100.0).to(self.device)
        nn.init.zeros_(carry_layer.T.weight)
        y_carry = carry_layer(x)
        self.assertTrue(torch.allclose(y_carry, x, atol=1e-6))

        trans_layer = HighwayLinear(features=32, transform_bias=100.0).to(self.device)
        nn.init.zeros_(trans_layer.T.weight)
        expected_h = F.relu(trans_layer.H(x))
        y_trans = trans_layer(x)
        self.assertTrue(torch.allclose(y_trans, expected_h, atol=1e-6))

    def test_equation_5_jacobian_identity_when_gate_closed(self):
        x = torch.randn(4, 16, device=self.device, requires_grad=True)
        layer = HighwayLinear(features=16, transform_bias=-100.0).to(self.device)
        nn.init.zeros_(layer.T.weight)

        y = layer(x)
        upstream_grad = torch.randn_like(y)
        y.backward(upstream_grad)
        self.assertTrue(torch.allclose(x.grad, upstream_grad, atol=1e-6))

    def test_negative_bias_initialization(self):
        for bias_val in [-1.0, -2.0, -4.0]:
            fc_layer = HighwayLinear(features=50, transform_bias=bias_val)
            conv_layer = HighwayConv2d(channels=32, kernel_size=3, transform_bias=bias_val)
            self.assertTrue(torch.allclose(fc_layer.T.bias, torch.full_like(fc_layer.T.bias, bias_val)))
            self.assertTrue(torch.allclose(conv_layer.T.bias, torch.full_like(conv_layer.T.bias, bias_val)))

    def test_optimization_in_50_layer_highway_vs_plain(self):
        hw_net = DeepFCHighwayNet(
            input_dim=64, hidden_dim=50, num_highway_layers=49, num_classes=10, transform_bias=-4.0
        ).to(self.device)
        plain_net = DeepFCPlainNet(
            input_dim=64, hidden_dim=71, num_hidden_layers=49, num_classes=10
        ).to(self.device)

        x = torch.randn(64, 64, device=self.device)
        target = torch.randint(0, 10, (64,), device=self.device)
        criterion = nn.CrossEntropyLoss()

        opt_hw = torch.optim.SGD(hw_net.parameters(), lr=0.02, momentum=0.9)
        opt_plain = torch.optim.SGD(plain_net.parameters(), lr=0.02, momentum=0.9)

        for _ in range(150):
            opt_hw.zero_grad()
            loss_hw = criterion(hw_net(x), target)
            loss_hw.backward()
            opt_hw.step()

            opt_plain.zero_grad()
            loss_plain = criterion(plain_net(x), target)
            loss_plain.backward()
            opt_plain.step()

        self.assertLess(loss_hw.item(), 0.5)
        self.assertLess(loss_hw.item(), loss_plain.item() * 0.25)

    def test_plain_transition_conv_changes_dimensionality(self):
        trans = PlainTransitionConv2d(in_channels=3, out_channels=48, kernel_size=3).to(self.device)
        x = torch.randn(4, 3, 32, 32, device=self.device)
        y = trans(x)
        self.assertEqual(y.shape, (4, 48, 32, 32))

    def test_conv_highway_architectures_forward_backward(self):
        x = torch.randn(4, 3, 32, 32, device=self.device)
        target = torch.randint(0, 10, (4,), device=self.device)
        criterion = nn.CrossEntropyLoss()

        for builder, expected_hw_count in [(build_highway11_cifar10, 7), (build_highway19_cifar10, 15)]:
            model = builder(transform_bias=-2.0).to(self.device)
            self.assertEqual(len(model.get_highway_layers()), expected_hw_count)
            out = model(x)
            self.assertEqual(out.shape, (4, 10))
            loss = criterion(out, target)
            loss.backward()
            grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
            self.assertEqual(len(grad_norms), len(list(model.parameters())))
            self.assertTrue(all(g > 0.0 for g in grad_norms))

    def test_dataset_contract_stratification_and_dummy_baseline(self):
        train_loader, val_loader, test_loader = get_cifar10_loaders(
            data_dir="./data", batch_size=256, val_fraction=0.1, num_workers=0
        )
        self.assertEqual(len(train_loader.dataset), 45000)
        self.assertEqual(len(val_loader.dataset), 5000)
        self.assertEqual(len(test_loader.dataset), 10000)

        train_counts = np.bincount(train_loader.dataset.targets.numpy(), minlength=10)
        val_counts = np.bincount(val_loader.dataset.targets.numpy(), minlength=10)
        self.assertTrue(np.all(train_counts == 4500))
        self.assertTrue(np.all(val_counts == 500))

        train_imgs = train_loader.dataset.images
        self.assertTrue(torch.allclose(train_imgs.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4))
        self.assertTrue(torch.allclose(train_imgs.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-4))

        dummy = compute_dummy_baseline(train_loader, val_loader)
        self.assertAlmostEqual(dummy["val_acc"], 10.0, places=4)
        self.assertAlmostEqual(dummy["val_loss"], math.log(10), places=5)

    def test_trained_checkpoint_on_cifar10(self):
        ckpt_path = "best_highway_cifar10.pt"
        if not os.path.isfile(ckpt_path):
            self.skipTest("Trained checkpoint not found")

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model = build_highway11_cifar10(transform_bias=ckpt.get("transform_bias", -2.0)).to(self.device)
        model.load_state_dict(ckpt["model_state_dict"])

        train_loader, val_loader, test_loader = get_cifar10_loaders(data_dir="./data", batch_size=256, num_workers=0)
        dummy = compute_dummy_baseline(train_loader, val_loader)
        criterion = nn.CrossEntropyLoss()
        metrics = evaluate(model, test_loader, criterion, self.device, return_detailed=True)
        ci_lo, ci_hi = metrics["ci_95"]

        print(
            f"\n[Checkpoint Evaluation] Epoch: {ckpt['epoch']} | "
            f"Val Acc: {ckpt.get('val_acc', 0.0):.2f}% | "
            f"Test Loss: {metrics['loss']:.4f} | "
            f"Test Acc: {metrics['acc']:.2f}% (95% CI [{ci_lo:.2f}%, {ci_hi:.2f}%]) | "
            f"Macro F1: {metrics['macro_f1']:.2f}%"
        )
        self.assertAlmostEqual(metrics["acc"], ckpt["test_acc"], places=1)
        self.assertGreater(metrics["acc"], dummy["val_acc"] * 7.5)
        self.assertGreater(metrics["macro_f1"], 75.0)
        self.assertEqual(metrics["confusion_matrix"].shape, (10, 10))
        self.assertTrue(os.path.isfile("training_curves.png"))
        self.assertTrue(os.path.isfile("highway_gates_analysis.png"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
