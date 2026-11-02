import torch
import torch.nn as nn
import torch.nn.functional as F


class HighwayLinear(nn.Module):
    def __init__(self, features, transform_bias=-2.0, activation=F.relu):
        super().__init__()
        self.H = nn.Linear(features, features)
        self.T = nn.Linear(features, features)
        self.activation = activation
        self.last_gate = None
        self.last_output = None
        self._init_weights(transform_bias)

    def _init_weights(self, transform_bias):
        nn.init.kaiming_normal_(self.H.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.H.bias)
        nn.init.kaiming_normal_(self.T.weight, mode="fan_in", nonlinearity="linear")
        nn.init.constant_(self.T.bias, transform_bias)

    def forward(self, x):
        h = self.activation(self.H(x))
        t = torch.sigmoid(self.T(x))
        out = h * t + x * (1.0 - t)
        if not self.training:
            self.last_gate = t.detach()
            self.last_output = out.detach()
        return out


class HighwayConv2d(nn.Module):
    def __init__(self, channels, kernel_size=3, transform_bias=-2.0, use_bn=True):
        super().__init__()
        padding = kernel_size // 2
        self.H = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=not use_bn)
        self.bn_h = nn.BatchNorm2d(channels) if use_bn else nn.Identity()
        self.T = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.last_gate = None
        self.last_output = None
        self._init_weights(transform_bias)

    def _init_weights(self, transform_bias):
        nn.init.kaiming_normal_(self.H.weight, mode="fan_out", nonlinearity="relu")
        if self.H.bias is not None:
            nn.init.zeros_(self.H.bias)
        nn.init.kaiming_normal_(self.T.weight, mode="fan_out", nonlinearity="linear")
        nn.init.constant_(self.T.bias, transform_bias)

    def forward(self, x):
        h = F.relu(self.bn_h(self.H(x)))
        t = torch.sigmoid(self.T(x))
        out = h * t + x * (1.0 - t)
        if not self.training:
            self.last_gate = t.detach()
            self.last_output = out.detach()
        return out


class PlainTransitionConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, use_bn=True):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_out", nonlinearity="relu")
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class ConvHighwayNet(nn.Module):
    def __init__(
        self,
        num_classes=10,
        stage_channels=(32, 64, 128),
        stage_highway_layers=(3, 4, 3),
        transform_bias=-2.0,
        dropout_p=0.2,
        use_bn=True,
    ):
        super().__init__()
        stages = []
        in_ch = 3
        for stage_idx, (out_ch, num_hw) in enumerate(zip(stage_channels, stage_highway_layers)):
            layers = [PlainTransitionConv2d(in_ch, out_ch, kernel_size=3, use_bn=use_bn)]
            for _ in range(num_hw):
                layers.append(HighwayConv2d(out_ch, kernel_size=3, transform_bias=transform_bias, use_bn=use_bn))
            if stage_idx < len(stage_channels) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
                if dropout_p > 0:
                    layers.append(nn.Dropout2d(p=dropout_p))
            stages.append(nn.Sequential(*layers))
            in_ch = out_ch

        self.stages = nn.ModuleList(stages)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(stage_channels[-1], num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_highway_layers(self):
        return [m for m in self.modules() if isinstance(m, HighwayConv2d)]


class DeepFCHighwayNet(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=50, num_highway_layers=19, num_classes=10, transform_bias=-2.0):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(self.input_layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.input_layer.bias)

        self.highway_layers = nn.ModuleList(
            [HighwayLinear(hidden_dim, transform_bias=transform_bias) for _ in range(num_highway_layers)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.input_layer(x))
        for layer in self.highway_layers:
            x = layer(x)
        return self.classifier(x)

    def get_highway_layers(self):
        return list(self.highway_layers)


class DeepFCPlainNet(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=71, num_hidden_layers=19, num_classes=10):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.kaiming_normal_(self.input_layer.weight, mode="fan_in", nonlinearity="relu")
        nn.init.zeros_(self.input_layer.bias)

        layers = []
        for _ in range(num_hidden_layers):
            lin = nn.Linear(hidden_dim, hidden_dim)
            nn.init.kaiming_normal_(lin.weight, mode="fan_in", nonlinearity="relu")
            nn.init.zeros_(lin.bias)
            layers.append(lin)
        self.hidden_layers = nn.ModuleList(layers)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, mode="fan_in", nonlinearity="linear")
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.input_layer(x))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.classifier(x)

    def get_highway_layers(self):
        return []


def build_highway11_cifar10(transform_bias=-2.0, use_bn=True):
    return ConvHighwayNet(
        num_classes=10,
        stage_channels=(32, 48, 64),
        stage_highway_layers=(2, 3, 2),
        transform_bias=transform_bias,
        dropout_p=0.1,
        use_bn=use_bn,
    )


def build_highway19_cifar10(transform_bias=-2.0, use_bn=True):
    return ConvHighwayNet(
        num_classes=10,
        stage_channels=(64, 96, 128),
        stage_highway_layers=(5, 5, 5),
        transform_bias=transform_bias,
        dropout_p=0.1,
        use_bn=use_bn,
    )


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
