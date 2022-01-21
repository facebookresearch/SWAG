# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/regnet.py

import math
from collections import OrderedDict
from enum import Enum, auto
from typing import Optional, Sequence

import numpy as np
import torch.nn as nn


RELU_IN_PLACE = True
NORMALIZE_L2 = "l2"


def is_pos_int(number: int) -> bool:
    """
    Returns True if a number is a positive integer.
    """
    return type(number) == int and number >= 0


class FullyConnectedHead(nn.Module):
    """This head defines a 2d average pooling layer
    (:class:`torch.nn.AdaptiveAvgPool2d`) followed by a fully connected
    layer (:class:`torch.nn.Linear`).
    """

    def __init__(
        self,
        num_classes: Optional[int],
        in_plane: int,
        conv_planes: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        zero_init_bias: bool = False,
        normalize_inputs: Optional[str] = None,
    ):
        """Constructor for FullyConnectedHead
        Args:
            num_classes: Number of classes for the head. If None, then the fully
                connected layer is not applied.
            in_plane: Input size for the fully connected layer.
            conv_planes: If specified, applies a 1x1 convolutional layer to the input
                before passing it to the average pooling layer. The convolution is also
                followed by a BatchNorm and an activation.
            activation: The activation to be applied after the convolutional layer.
                Unused if `conv_planes` is not specified.
            zero_init_bias: Zero initialize the bias
            normalize_inputs: If specified, normalize the inputs after performing
                average pooling using the specified method. Supports "l2" normalization.
        """
        super().__init__()
        assert num_classes is None or is_pos_int(num_classes)
        assert is_pos_int(in_plane)
        if conv_planes is not None and activation is None:
            raise TypeError("activation cannot be None if conv_planes is specified")
        if normalize_inputs is not None and normalize_inputs != NORMALIZE_L2:
            raise ValueError(
                f"Unsupported value for normalize_inputs: {normalize_inputs}"
            )
        self.conv = (
            nn.Conv2d(in_plane, conv_planes, kernel_size=1, bias=False)
            if conv_planes
            else None
        )
        self.bn = nn.BatchNorm2d(conv_planes) if conv_planes else None
        self.activation = activation
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = (
            None
            if num_classes is None
            else nn.Linear(
                in_plane if conv_planes is None else conv_planes, num_classes
            )
        )
        self.normalize_inputs = normalize_inputs

        if zero_init_bias:
            self.fc.bias.data.zero_()

    def forward(self, x):
        out = x
        if self.conv is not None:
            out = self.activation(self.bn(self.conv(x)))

        out = self.avgpool(out)

        out = out.flatten(start_dim=1)

        if self.normalize_inputs is not None:
            if self.normalize_inputs == NORMALIZE_L2:
                out = nn.functional.normalize(out, p=2.0, dim=1)

        if self.fc is not None:
            out = self.fc(out)

        return out


# The different possible blocks
class BlockType(Enum):
    VANILLA_BLOCK = auto()
    RES_BASIC_BLOCK = auto()
    RES_BOTTLENECK_BLOCK = auto()
    RES_BOTTLENECK_LINEAR_BLOCK = auto()


# The different possible Stems
class StemType(Enum):
    RES_STEM_CIFAR = auto()
    RES_STEM_IN = auto()
    SIMPLE_STEM_IN = auto()


# The different possible activations
class ActivationType(Enum):
    RELU = auto()
    SILU = auto()


class SqueezeAndExcitationLayer(nn.Module):
    """Squeeze and excitation layer, as per https://arxiv.org/pdf/1709.01507.pdf"""

    def __init__(
        self,
        in_planes,
        reduction_ratio: Optional[int] = 16,
        reduced_planes: Optional[int] = None,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Either reduction_ratio is defined, or out_planes is defined,
        # neither both nor none of them
        assert bool(reduction_ratio) != bool(reduced_planes)

        if activation is None:
            activation = nn.ReLU()

        reduced_planes = (
            in_planes // reduction_ratio if reduced_planes is None else reduced_planes
        )
        self.excitation = nn.Sequential(
            nn.Conv2d(in_planes, reduced_planes, kernel_size=1, stride=1, bias=True),
            activation,
            nn.Conv2d(reduced_planes, in_planes, kernel_size=1, stride=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_squeezed = self.avgpool(x)
        x_excited = self.excitation(x_squeezed)
        x_scaled = x * x_excited
        return x_scaled


class BasicTransform(nn.Sequential):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ):
        super().__init__()

        self.a = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
            nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False),
        )

        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 2


class ResStemCifar(nn.Sequential):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )
        self.depth = 2


class ResStemIN(nn.Sequential):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(width_in, width_out, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.depth = 3


class SimpleStemIN(nn.Sequential):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
    ):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )
        self.depth = 2


class VanillaBlock(nn.Sequential):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.a = nn.Sequential(
            nn.Conv2d(width_in, width_out, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        self.b = nn.Sequential(
            nn.Conv2d(width_out, width_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        self.depth = 2


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.proj_block = (width_in != width_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(
                width_in, width_out, 1, stride=stride, padding=0, bias=False
            )
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BasicTransform(
            width_in, width_out, stride, bn_epsilon, bn_momentum, activation
        )
        self.activation = activation

        # The projection and transform happen in parallel,
        # and ReLU is not counted with respect to depth
        self.depth = self.f.depth

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)

        return self.activation(x)


class BottleneckTransform(nn.Sequential):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int,
        bottleneck_multiplier: float,
        se_ratio: Optional[float],
    ):
        super().__init__()
        w_b = int(round(width_out * bottleneck_multiplier))
        g = w_b // group_width

        self.a = nn.Sequential(
            nn.Conv2d(width_in, w_b, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        self.b = nn.Sequential(
            nn.Conv2d(w_b, w_b, 3, stride=stride, padding=1, groups=g, bias=False),
            nn.BatchNorm2d(w_b, eps=bn_epsilon, momentum=bn_momentum),
            activation,
        )

        if se_ratio:
            # The SE reduction ratio is defined with respect to the
            # beginning of the block
            width_se_out = int(round(se_ratio * width_in))
            self.se = SqueezeAndExcitationLayer(
                in_planes=w_b,
                reduction_ratio=None,
                reduced_planes=width_se_out,
                activation=activation,
            )

        self.c = nn.Conv2d(w_b, width_out, 1, stride=1, padding=0, bias=False)
        self.final_bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.depth = 3 if not se_ratio else 4


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int = 1,
        bottleneck_multiplier: float = 1.0,
        se_ratio: Optional[float] = None,
    ):
        super().__init__()

        # Use skip connection with projection if shape changes
        self.proj_block = (width_in != width_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2d(
                width_in, width_out, 1, stride=stride, padding=0, bias=False
            )
            self.bn = nn.BatchNorm2d(width_out, eps=bn_epsilon, momentum=bn_momentum)
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            bn_epsilon,
            bn_momentum,
            activation,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )
        self.activation = activation

        # The projection and transform happen in parallel,
        # and activation is not counted with respect to depth
        self.depth = self.f.depth

    def forward(self, x, *args):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        return self.activation(x)


class ResBottleneckLinearBlock(nn.Module):
    """Residual linear bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        bn_epsilon: float,
        bn_momentum: float,
        activation: nn.Module,
        group_width: int = 1,
        bottleneck_multiplier: float = 4.0,
        se_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.has_skip = (width_in == width_out) and (stride == 1)
        self.f = BottleneckTransform(
            width_in,
            width_out,
            stride,
            bn_epsilon,
            bn_momentum,
            activation,
            group_width,
            bottleneck_multiplier,
            se_ratio,
        )

        self.depth = self.f.depth

    def forward(self, x):
        return x + self.f(x) if self.has_skip else self.f(x)


class AnyStage(nn.Sequential):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(
        self,
        width_in: int,
        width_out: int,
        stride: int,
        depth: int,
        block_constructor: nn.Module,
        activation: nn.Module,
        group_width: int,
        bottleneck_multiplier: float,
        params: "AnyNetParams",
        stage_index: int = 0,
    ):
        super().__init__()
        self.stage_depth = 0

        for i in range(depth):
            block = block_constructor(
                width_in if i == 0 else width_out,
                width_out,
                stride if i == 0 else 1,
                params.bn_epsilon,
                params.bn_momentum,
                activation,
                group_width,
                bottleneck_multiplier,
                params.se_ratio,
            )

            self.stage_depth += block.depth
            self.add_module(f"block{stage_index}-{i}", block)


class AnyNetParams:
    def __init__(
        self,
        depths: Sequence[int],
        widths: Sequence[int],
        group_widths: Sequence[int],
        bottleneck_multipliers: Sequence[int],
        strides: Sequence[int],
        stem_type: StemType = StemType.SIMPLE_STEM_IN,
        stem_width: int = 32,
        block_type: BlockType = BlockType.RES_BOTTLENECK_BLOCK,
        activation: ActivationType = ActivationType.RELU,
        use_se: bool = True,
        se_ratio: float = 0.25,
        bn_epsilon: float = 1e-05,
        bn_momentum: bool = 0.1,
        num_classes: Optional[int] = None,
    ):
        self.depths = depths
        self.widths = widths
        self.group_widths = group_widths
        self.bottleneck_multipliers = bottleneck_multipliers
        self.strides = strides
        self.stem_type = stem_type
        self.stem_width = stem_width
        self.block_type = block_type
        self.activation = activation
        self.use_se = use_se
        self.se_ratio = se_ratio if use_se else None
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.num_classes = num_classes
        self.relu_in_place = RELU_IN_PLACE

    def get_expanded_params(self):
        """Return an iterator over AnyNet parameters for each stage."""
        return zip(
            self.widths,
            self.strides,
            self.depths,
            self.group_widths,
            self.bottleneck_multipliers,
        )


class AnyNet(nn.Module):
    """Implementation of an AnyNet.
    See https://arxiv.org/abs/2003.13678 for details.
    """

    def __init__(self, params: AnyNetParams):
        super().__init__()

        activation = {
            ActivationType.RELU: nn.ReLU(params.relu_in_place),
            ActivationType.SILU: nn.SiLU(),
        }[params.activation]

        if activation is None:
            raise RuntimeError("SiLU activation is only supported since PyTorch 1.7")

        assert params.num_classes is None or is_pos_int(params.num_classes)

        # Ad hoc stem
        self.stem = {
            StemType.RES_STEM_CIFAR: ResStemCifar,
            StemType.RES_STEM_IN: ResStemIN,
            StemType.SIMPLE_STEM_IN: SimpleStemIN,
        }[params.stem_type](
            3,
            params.stem_width,
            params.bn_epsilon,
            params.bn_momentum,
            activation,
        )

        # Instantiate all the AnyNet blocks in the trunk
        block_fun = {
            BlockType.VANILLA_BLOCK: VanillaBlock,
            BlockType.RES_BASIC_BLOCK: ResBasicBlock,
            BlockType.RES_BOTTLENECK_BLOCK: ResBottleneckBlock,
            BlockType.RES_BOTTLENECK_LINEAR_BLOCK: ResBottleneckLinearBlock,
        }[params.block_type]

        current_width = params.stem_width

        self.trunk_depth = 0

        blocks = []
        for i, (
            width_out,
            stride,
            depth,
            group_width,
            bottleneck_multiplier,
        ) in enumerate(params.get_expanded_params()):
            blocks.append(
                (
                    f"block{i+1}",
                    AnyStage(
                        current_width,
                        width_out,
                        stride,
                        depth,
                        block_fun,
                        activation,
                        group_width,
                        bottleneck_multiplier,
                        params,
                        stage_index=i + 1,
                    ),
                )
            )

            self.trunk_depth += blocks[-1][1].stage_depth

            current_width = width_out

        self.trunk_output = nn.Sequential(OrderedDict(blocks))

        # Init weights and good to go
        self.init_weights()

        # If head, create
        if params.num_classes is not None:
            self.head = FullyConnectedHead(
                num_classes=params.num_classes, in_plane=current_width
            )
        else:
            self.head = None

    def forward(self, x, *args, **kwargs):
        x = self.stem(x)
        x = self.trunk_output(x)
        if self.head is None:
            return x
        x = self.head(x)

        return x

    def init_weights(self):
        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()


def _quantize_float(f, q):
    """Converts a float to closest non-zero int divisible by q."""
    return int(round(f / q) * q)


def _adjust_widths_groups_compatibilty(stage_widths, bottleneck_ratios, group_widths):
    """Adjusts the compatibility of widths and groups,
    depending on the bottleneck ratio."""
    # Compute all widths for the current settings
    widths = [int(w * b) for w, b in zip(stage_widths, bottleneck_ratios)]
    groud_widths_min = [min(g, w_bot) for g, w_bot in zip(group_widths, widths)]

    # Compute the adjusted widths so that stage and group widths fit
    ws_bot = [_quantize_float(w_bot, g) for w_bot, g in zip(widths, groud_widths_min)]
    stage_widths = [int(w_bot / b) for w_bot, b in zip(ws_bot, bottleneck_ratios)]
    return stage_widths, groud_widths_min


class RegNetParams(AnyNetParams):
    def __init__(
        self,
        depth: int,
        w_0: int,
        w_a: float,
        w_m: float,
        group_width: int,
        bottleneck_multiplier: float = 1.0,
        stem_type: StemType = StemType.SIMPLE_STEM_IN,
        stem_width: int = 32,
        block_type: BlockType = BlockType.RES_BOTTLENECK_BLOCK,
        activation: ActivationType = ActivationType.RELU,
        use_se: bool = True,
        se_ratio: float = 0.25,
        bn_epsilon: float = 1e-05,
        bn_momentum: bool = 0.1,
        num_classes: Optional[int] = None,
    ):
        assert (
            w_a >= 0 and w_0 > 0 and w_m > 1 and w_0 % 8 == 0
        ), "Invalid RegNet settings"
        self.depth = depth
        self.w_0 = w_0
        self.w_a = w_a
        self.w_m = w_m
        self.group_width = group_width
        self.bottleneck_multiplier = bottleneck_multiplier
        self.stem_type = stem_type
        self.block_type = block_type
        self.activation = activation
        self.stem_width = stem_width
        self.use_se = use_se
        self.se_ratio = se_ratio if use_se else None
        self.bn_epsilon = bn_epsilon
        self.bn_momentum = bn_momentum
        self.num_classes = num_classes
        self.relu_in_place = RELU_IN_PLACE

    def get_expanded_params(self):
        """Programatically compute all the per-block settings,
        given the RegNet parameters.
        The first step is to compute the quantized linear block parameters,
        in log space. Key parameters are:
        - `w_a` is the width progression slope
        - `w_0` is the initial width
        - `w_m` is the width stepping in the log space
        In other terms
        `log(block_width) = log(w_0) + w_m * block_capacity`,
        with `bock_capacity` ramping up following the w_0 and w_a params.
        This block width is finally quantized to multiples of 8.
        The second step is to compute the parameters per stage,
        taking into account the skip connection and the final 1x1 convolutions.
        We use the fact that the output width is constant within a stage
        """

        QUANT = 8
        STRIDE = 2

        # Compute the block widths. Each stage has one unique block width
        widths_cont = np.arange(self.depth) * self.w_a + self.w_0
        block_capacity = np.round(np.log(widths_cont / self.w_0) / np.log(self.w_m))
        block_widths = (
            np.round(np.divide(self.w_0 * np.power(self.w_m, block_capacity), QUANT))
            * QUANT
        )
        num_stages = len(np.unique(block_widths))
        block_widths = block_widths.astype(int).tolist()

        # Convert to per stage parameters
        split_helper = zip(
            block_widths + [0],
            [0] + block_widths,
            block_widths + [0],
            [0] + block_widths,
        )
        splits = [w != wp or r != rp for w, wp, r, rp in split_helper]

        stage_widths = [w for w, t in zip(block_widths, splits[:-1]) if t]
        stage_depths = np.diff([d for d, t in enumerate(splits) if t]).tolist()

        strides = [STRIDE] * num_stages
        bottleneck_multipliers = [self.bottleneck_multiplier] * num_stages
        group_widths = [self.group_width] * num_stages

        # Adjust the compatibility of stage widths and group widths
        stage_widths, group_widths = _adjust_widths_groups_compatibilty(
            stage_widths, bottleneck_multipliers, group_widths
        )

        return zip(
            stage_widths, strides, stage_depths, group_widths, bottleneck_multipliers
        )


class RegNet(AnyNet):
    """Implementation of RegNet, a particular form of AnyNets.
    See https://arxiv.org/abs/2003.13678 for introduction to RegNets, and details about
    RegNetX and RegNetY models.
    See https://arxiv.org/abs/2103.06877 for details about RegNetZ models.
    """

    def __init__(self, params: RegNetParams):
        super().__init__(params)

    def forward(self, x, *args, **kwargs):
        x = self.stem(x)
        x = self.trunk_output(x)
        if self.head is None:
            return x
        x = self.head(x)

        return x

    def init_weights(self):
        # Performs ResNet-style weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Note that there is no bias due to BN
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=math.sqrt(2.0 / fan_out))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                m.bias.data.zero_()


class RegNetY16gf(RegNet):
    def __init__(self, **kwargs):
        # Output size: 3024 feature maps
        super().__init__(
            RegNetParams(
                depth=18, w_0=200, w_a=106.23, w_m=2.48, group_width=112, **kwargs
            )
        )


class RegNetY32gf(RegNet):
    def __init__(self, **kwargs):
        # Output size: 3712 feature maps
        super().__init__(
            RegNetParams(
                depth=20, w_0=232, w_a=115.89, w_m=2.53, group_width=232, **kwargs
            )
        )


class RegNetY128gf(RegNet):
    def __init__(self, **kwargs):
        # Output size: 7392 feature maps
        super().__init__(
            RegNetParams(
                depth=27, w_0=456, w_a=160.83, w_m=2.52, group_width=264, **kwargs
            )
        )
