# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Optional

from torch.hub import load_state_dict_from_url

from .regnet import RegNetY32gf, RegNetY16gf, RegNetY128gf
from .vision_transformer import ViTB16, ViTH14, ViTL16


class ModelCheckpoints(Enum):
    regnety_16gf = ""
    regnety_16gf_in1k = ""
    regnety_32gf = ""
    regnety_32gf_in1k = ""
    regnety_128gf = ""
    regnety_128gf_in1k = ""
    vit_b16 = ""
    vit_b16_in1k = ""
    vit_l16 = ""
    vit_l16_in1k = ""
    vit_h14 = ""
    vit_h14_in1k = ""


IN1K_CLASSES = 1000


def build_model(
    cls: type,
    checkpoint_path: str,
    num_classes: Optional[int] = None,
    pretrained: bool = True,
    map_location: Any = None,
    progress: bool = True,
    **kwargs
):
    #model = cls(num_classes=num_classes, **kwargs)
    model = cls(**kwargs)
    if pretrained:
        checkpoint = load_state_dict_from_url(
            checkpoint_path, map_location=map_location, progress=progress
        )
        model.load_state_dict(checkpoint)
    return model


def regnety_16gf(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        RegNetY16gf,
        ModelCheckpoints.regnety_16gf,
        num_classes=None,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def regnety_16gf_in1k(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        RegNetY16gf,
        ModelCheckpoints.regnety_16gf_in1k,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def regnety_32gf(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        RegNetY32gf,
        ModelCheckpoints.regnety_32gf,
        num_classes=None,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def regnety_32gf_in1k(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        RegNetY32gf,
        ModelCheckpoints.regnety_32gf_in1k,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def regnety_128gf(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        RegNetY128gf,
        ModelCheckpoints.regnety_128gf,
        num_classes=None,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def regnety_128gf_in1k(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        RegNetY32gf,
        ModelCheckpoints.regnety_128gf_in1k,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def vit_b16(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        ViTB16,
        ModelCheckpoints.vit_b16,
        num_classes=None,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def vit_b16_in1k(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        ViTB16,
        ModelCheckpoints.vit_b16_in1k,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def vit_l16(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        ViTL16,
        ModelCheckpoints.vit_l16,
        num_classes=None,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def vit_l16_in1k(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        ViTL16,
        ModelCheckpoints.vit_l16_in1k,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def vit_h14(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        ViTH14,
        ModelCheckpoints.vit_h14,
        num_classes=None,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )


def vit_h14_in1k(
    pretrained: bool = True, map_location: Any = None, progress: bool = True, **kwargs
):
    return build_model(
        ViTH14,
        ModelCheckpoints.vit_h14_in1k,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        map_location=map_location,
        progress=progress,
        **kwargs
    )
