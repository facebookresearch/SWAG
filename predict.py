# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


"""
Download the weights in ./checkpoints and ImageNet 1K ID to class mappings beforehand
wget https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json -O in_cls_idx.json
"""
import json
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
import cog

from models.vision_transformer import ViTB16, ViTH14, ViTL16
from models.regnet import RegNetY32gf, RegNetY16gf, RegNetY128gf


class Predictor(cog.Predictor):
    def setup(self):
        IN1K_CLASSES = 1000
        self.device = "cuda:0"
        self.resolution = {
            'vit_h14_in1k': 518,
            'vit_l16_in1k': 512,
            'vit_b16_in1k': 384,
            'regnety_16gf_in1k': 384,
            'regnety_32gf_in1k': 384,
            'regnety_128gf_in1k': 384
        }

        vit_h14_in1k_model = ViTH14(image_size=518, num_classes=IN1K_CLASSES)
        vit_h14_in1k_model.load_state_dict(torch.load('checkpoints/vit_h14_in1k.torch'))
        vit_l16_in1k_model = ViTL16(image_size=512, num_classes=IN1K_CLASSES)
        vit_l16_in1k_model.load_state_dict(torch.load('checkpoints/vit_l16_in1k.torch'))
        vit_b16_in1k_model = ViTB16(image_size=384, num_classes=IN1K_CLASSES)
        vit_b16_in1k_model.load_state_dict(torch.load('checkpoints/vit_b16_in1k.torch'))
        regnety_16gf_in1k_model = RegNetY16gf(num_classes=IN1K_CLASSES)
        regnety_16gf_in1k_model.load_state_dict(torch.load('checkpoints/regnety_16gf_in1k.torch'))
        regnety_32gf_in1k_model = RegNetY32gf(num_classes=IN1K_CLASSES)
        regnety_32gf_in1k_model.load_state_dict(torch.load('checkpoints/regnety_32gf_in1k.torch'))
        regnety_128gf_in1k_model = RegNetY128gf(num_classes=IN1K_CLASSES)
        regnety_128gf_in1k_model.load_state_dict(torch.load('checkpoints/regnety_128gf_in1k.torch'))
        self.models = {
            'vit_h14_in1k': vit_h14_in1k_model,
            'vit_l16_in1k': vit_l16_in1k_model,
            'vit_b16_in1k': vit_b16_in1k_model,
            'regnety_16gf_in1k': regnety_16gf_in1k_model,
            'regnety_32gf_in1k': regnety_32gf_in1k_model,
            'regnety_128gf_in1k': regnety_128gf_in1k_model
        }

        with open("in_cls_idx.json", "r") as f:
            self.imagenet_id_to_name = {int(cls_id): name for cls_id, (label, name) in json.load(f).items()}

    @cog.input(
        "image",
        type=Path,
        help="input image",
    )
    @cog.input(
        "model_name",
        type=str,
        default='vit_h14_in1k',
        options=['vit_h14_in1k', 'vit_l16_in1k', 'vit_b16_in1k', 'regnety_16gf_in1k', 'regnety_32gf_in1k', 'regnety_128gf_in1k'],
        help="Choose a model type",
    )
    @cog.input(
        "topk",
        type=int,
        min=1,
        max=10,
        default=5,
        help="Choose top k predictions to return.",
    )
    def predict(self, image, model_name, topk):
        resolution = self.resolution[model_name]
        model = self.models[model_name]
        model.to(self.device)
        model.eval()

        image = Image.open(str(image)).convert("RGB")
        image = transform_image(image, resolution)
        image = image.to(self.device)
        # we do not need to track gradients for inference
        with torch.no_grad():
            _, preds = model(image).topk(topk)
        preds = preds.tolist()[0]
        return [self.imagenet_id_to_name[cls_id] for cls_id in preds]


def transform_image(image, resolution):
    transform = transforms.Compose([
        transforms.Resize(
            resolution,
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ])
    image = transform(image)
    # we also add a batch dimension to the image since that is what the model expects
    image = image[None, :]
    return image
