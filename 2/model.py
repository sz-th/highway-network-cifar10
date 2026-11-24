import torch
import torch.nn as nn
import torch.nn.functional as F


class ShortcutOptionA(nn.Module):
    def __init__(self, out_planes):
        super().__init__()
        self.pad_channels = out_planes // 4

    def forward(self, x):
        return F.pad(
            x[:, :, ::2, ::2],
            (0, 0, 0, 0, self.pad_channels, self.pad_channels),
            mode="constant",
            value=0.0,
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, shortcut_option="A"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if shortcut_option == "A":
                self.shortcut = ShortcutOptionA(planes)
            elif shortcut_option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, planes, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(planes),
                )
            else:
                raise ValueError(f"Unknown shortcut option: {shortcut_option}")

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, shortcut_option="A"):
        super().__init__()
        self.in_planes = 16
        self.shortcut_option = shortcut_option

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks, stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks, stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks, stride=2)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, self.shortcut_option))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


def resnet56(num_classes=10, shortcut_option="A"):
    return ResNetCIFAR(
        BasicBlock, num_blocks=9, num_classes=num_classes, shortcut_option=shortcut_option
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
