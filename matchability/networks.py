import torch
from torch import nn


class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, norm=True, relu=True):
        super().__init__()

        if padding > 0:
            self.add_module('pad', nn.ReflectionPad2d(padding))

        self.add_module('conv',
                        nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias))
        if norm:
            self.add_module('norm',
                            nn.BatchNorm2d(out_channels, affine=True))
        if relu:
            self.add_module('relu', nn.PReLU())


class ConvTransposeLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, output_padding=0,
                 bias=True, dilation=1,
                 norm=True, relu=True):
        super().__init__()

        if padding > 0:
            self.add_module('pad', nn.ReflectionPad2d(padding))
        self.add_module('conv',
                        nn.ConvTranspose2d(in_channels=in_channels,
                                           out_channels=out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride,
                                           padding=0,
                                           output_padding=output_padding,
                                           bias=bias,
                                           dilation=dilation))
        if norm:
            self.add_module('norm',
                            nn.BatchNorm2d(out_channels, affine=True))
        if relu:
            self.add_module('relu', nn.PReLU())


class UpsampleConvLayer(nn.Sequential):
    def __init__(self, scale_factor, mode,
                 in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, norm=True, relu=True):
        super().__init__()

        self.scale_factor = scale_factor
        self.mode = mode

        if padding > 0:
            self.add_module('pad', nn.ReflectionPad2d(padding))
        self.add_module('conv',
                        nn.Conv2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  padding=0,
                                  dilation=dilation,
                                  groups=groups,
                                  bias=bias))
        if norm:
            self.add_module('norm',
                            nn.BatchNorm2d(out_channels, affine=True))
        if relu:
            self.add_module('relu', nn.PReLU())

    def forward(self, x):
        x = nn.functional.interpolate(x,
                                      scale_factor=self.scale_factor,
                                      mode=self.mode)
        out = super().forward(x)
        return out


class ResidualBlock(nn.Sequential):
    def __init__(self, outer_channels, inner_channels,
                 kernel_size=3, stride=1, dilation=1, padding=1,
                 groups=1, bias=True, norm=True):
        super().__init__()
        self.add_module('conv0',
                        ConvLayer(outer_channels, inner_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  dilation=dilation,
                                  padding=padding,
                                  groups=groups,
                                  bias=bias, norm=norm, relu=True))
        self.add_module('conv1',
                        ConvLayer(inner_channels, outer_channels,
                                  kernel_size=kernel_size,
                                  stride=stride,
                                  dilation=dilation,
                                  padding=padding,
                                  groups=groups,
                                  bias=bias, norm=norm, relu=True))

    def forward(self, x):
        res = super().forward(x)
        return x + res


class SiameseNet(nn.Sequential):
    def __init__(self, source_channels, output_channels,
                 init_channels=64, feature_blocks=5, concat_blocks=5,
                 max_channels=256, final_kernel_size=1):
        super().__init__()

        self.features = nn.Sequential()
        self.features.add_module('conv0',
                                 ConvLayer(source_channels, init_channels,
                                           kernel_size=5, stride=2, padding=2,
                                           bias=True, norm=True, relu=True))
        self.features.add_module('res0',
                                 ResidualBlock(init_channels, init_channels,
                                               kernel_size=3, stride=1, padding=1,
                                               bias=True, norm=True))
        channels = init_channels
        for i in range(1, feature_blocks):
            new_channels = min(channels*2, max_channels)
            self.features.add_module('conv{}'.format(i),
                                     ConvLayer(channels, new_channels,
                                               kernel_size=3, stride=2, padding=1,
                                               bias=True, norm=True, relu=True))
            self.features.add_module('res{}'.format(i),
                                     ResidualBlock(new_channels, new_channels,
                                                   kernel_size=3, stride=1, padding=1,
                                                   bias=True, norm=True))
            channels = new_channels

        channels = channels*2
        self.concat = nn.Sequential()

        for i in range(0, concat_blocks):
            new_channels = min(channels*2, max_channels)
            self.concat.add_module('conv{}'.format(i),
                                   ConvLayer(channels, new_channels,
                                             kernel_size=3, stride=2, padding=1,
                                             bias=True, norm=True, relu=True))
            self.concat.add_module('res{}'.format(i),
                                   ResidualBlock(new_channels, new_channels,
                                                 kernel_size=3, stride=1, padding=1,
                                                 bias=True, norm=True))
            channels = new_channels

        self.concat.add_module('conv{}'.format(concat_blocks),
                               ConvLayer(channels, output_channels,
                                         kernel_size=final_kernel_size,
                                         stride=1, padding=0,
                                         bias=True, norm=False, relu=False))

    def forward(self, x1, x2):
        y1 = self.features(x1)
        y2 = self.features(x2)
        y = torch.cat([y1, y2], dim=1)
        y = self.concat(y)
        return y


class TransformerNet(nn.Sequential):
    def __init__(self, source_channels, output_channels,
                 init_channels=64, max_channels=4096, num_layers=8):
        super().__init__()

        self.add_module('conv0',
                        ConvLayer(source_channels, init_channels,
                                  kernel_size=1, stride=1, padding=0,
                                  bias=True, norm=False, relu=True))
        channels = init_channels

        for i in range(1, num_layers):
            new_channels = min(channels*2, max_channels)
            self.add_module('conv{}'.format(i),
                            ConvLayer(channels, new_channels,
                                      kernel_size=1, stride=1, padding=0,
                                      bias=True, norm=False, relu=True))
            channels = new_channels

        self.add_module('conv{}'.format(num_layers),
                        ConvLayer(channels, output_channels,
                                  kernel_size=1, stride=1, padding=0,
                                  bias=True, norm=False, relu=False))


class Constant(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weight = nn.Parameter(torch.Tensor(dim))
        nn.init.uniform_(self.weight, -0.2, 0.2)

    def forward(self, batch_size):
        return self.weight.unsqueeze(dim=0).expand(
            batch_size, self.dim).unsqueeze(dim=2).unsqueeze(dim=3)

    def extra_repr(self):
        return 'dim={}'.format(self.dim)
