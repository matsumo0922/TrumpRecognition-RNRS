from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            activation: Optional[nn.ReLU] = nn.ReLU,
            norm: Optional[nn.BatchNorm2d] = nn.BatchNorm2d,
    ):
        super().__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.relu = activation(inplace=True) if activation is nn.ReLU else None
        self.norm = norm(out_ch) if norm is nn.BatchNorm2d else None

    def forward(self, x):
        if self.relu is None and self.norm is None:
            return self.conv(x)

        if self.norm is None:
            return self.relu(self.conv(x))

        if self.relu is None:
            return self.norm(self.conv(x))

        return self.relu(self.norm(self.conv(x)))


class StemConv(nn.Module):
    def __init__(self, in_ch, stem_width, is_deep=False):
        super(StemConv, self).__init__()

        inplanes = stem_width * 2
        layers = list()

        if is_deep:
            layers.extend([
                Conv(in_ch, stem_width, kernel_size=3, stride=2, padding=1),
                Conv(stem_width, stem_width, kernel_size=3, stride=1, padding=1),
                Conv(stem_width, inplanes, kernel_size=3, stride=1, padding=1),
                Conv(inplanes, inplanes, kernel_size=3, stride=2, padding=2)
            ])
        else:
            layers.append(
                Conv(in_ch, inplanes, kernel_size=7, stride=2, padding=3)
            )

        self.stem_conv = nn.Sequential(*layers)
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.max_pooling(x)

        return x


class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=0.25):
        super(SEBlock, self).__init__()

        reduced_channels = int(channels * reduction_ratio)

        self.conv1 = nn.Conv2d(channels, reduced_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(reduced_channels, channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x

        x = x.mean((2, 3), keepdim=True)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.sigmoid(x)

        return identity * x


class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, norm=nn.BatchNorm2d):
        super(DownsampleBlock, self).__init__()

        self.downsample = nn.Sequential(
            nn.Identity() if stride == 1 else nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=1, padding=0, bias=False),
            norm(out_ch)
        )

    def forward(self, x):
        return self.downsample(x)


class BottleneckBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, downsample=None, is_se_block=True, reduction_ratio=0.25, stochastic_depth_ratio=0.0):
        super(BottleneckBlock, self).__init__()

        self.conv1 = Conv(inplanes, planes, kernel_size=1)
        self.conv2 = Conv(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.conv3 = Conv(planes, planes * 4, kernel_size=1, activation=None)

        self.downsample = downsample
        self.se = SEBlock(planes * 4, reduction_ratio) if is_se_block else None
        self.drop_path = DropPath(stochastic_depth_ratio)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = self.downsample(x) if self.downsample is not None else x.clone()

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        if self.se is not None:
            x = self.se(x)

        if self.drop_path.drop_ratio:
            x = self.drop_path(x)

        x += identity

        return self.relu(x)


class DropPath(nn.Module):
    def __init__(self, drop_ratio=None):
        super(DropPath, self).__init__()
        self.drop_ratio = drop_ratio

    def forward(self, x):
        if self.drop_ratio is None or self.drop_ratio == 0 or not self.training:
            return x

        keep_ratio = 1 - self.drop_ratio
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_ratio + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = random_tensor.floor_()

        return x.div(keep_ratio) * random_tensor


class ResNetRs(nn.Module):
    def __init__(self, n_classes):
        super(ResNetRs, self).__init__()

        blocks = [3, 4, 6, 3]
        input_channels = [64, 256, 512, 1024]
        output_channels = [64, 128, 256, 512]

        self.dropout_ratio = 0.0
        self.stochastic_depth_ratio = 0.0
        self.total_blocks = sum(blocks)

        self.conv1 = StemConv(3, 32, is_deep=True)
        self.conv2_x = self.get_layer(BottleneckBlock, blocks[0], input_channels[0], output_channels[0], stride=1)
        self.conv3_x = self.get_layer(BottleneckBlock, blocks[1], input_channels[1], output_channels[1], stride=2)
        self.conv4_x = self.get_layer(BottleneckBlock, blocks[2], input_channels[2], output_channels[2], stride=2)
        self.conv5_x = self.get_layer(BottleneckBlock, blocks[3], input_channels[3], output_channels[3], stride=1)

        self.avg_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * 4, n_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pooling(x)

        x = x.flatten(1, -1)

        if self.dropout_ratio > 0.:
            x = nn.functional.dropout(x, p=self.dropout_ratio, training=self.training)

        x = self.linear(x)

        return x

    def get_layer(self, diff_block: Type[BottleneckBlock], n_blocks, input_channels, output_channel, stride, is_se_block=True, reduction_ratio=0.2):
        layer = list()

        for i in range(n_blocks):
            stride = stride if i == 0 else 1
            downsample = DownsampleBlock(input_channels, output_channel * 4, kernel_size=1, stride=stride) if i == 0 else None
            drop_ratio = (self.stochastic_depth_ratio * i) / (self.total_blocks - 1)

            layer.append(diff_block(
                input_channels,
                output_channel,
                stride=stride,
                downsample=downsample,
                is_se_block=is_se_block,
                reduction_ratio=reduction_ratio,
                stochastic_depth_ratio=drop_ratio
            ))

            input_channels = output_channel * 4

        return nn.Sequential(*layer)
