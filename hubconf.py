#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


dependencies = ["torch"]


from models import (
    regnety_16gf,
    regnety_16gf_in1k,
    regnety_32gf,
    regnety_32gf_in1k,
    regnety_128gf,
    regnety_128gf_in1k,
    vit_b16,
    vit_b16_in1k,
    vit_l16,
    vit_l16_in1k,
    vit_h14,
    vit_h14_in1k,
)
