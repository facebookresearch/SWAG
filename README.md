# SWAG: Supervised Weakly from hashtAGs

This repository contains SWAG models from the paper [Revisiting Weakly Supervised Pre-Training of Visual Perception Models](https://arxiv.org/abs/2201.08371).

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weakly-supervised-pre-training-of/image-classification-on-places365-standard)](https://paperswithcode.com/sota/image-classification-on-places365-standard?p=revisiting-weakly-supervised-pre-training-of)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weakly-supervised-pre-training-of/image-classification-on-inaturalist-2018)](https://paperswithcode.com/sota/image-classification-on-inaturalist-2018?p=revisiting-weakly-supervised-pre-training-of)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weakly-supervised-pre-training-of/fine-grained-image-classification-on-cub-200-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200-1?p=revisiting-weakly-supervised-pre-training-of)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weakly-supervised-pre-training-of/image-classification-on-objectnet)](https://paperswithcode.com/sota/image-classification-on-objectnet?p=revisiting-weakly-supervised-pre-training-of)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/revisiting-weakly-supervised-pre-training-of/image-classification-on-imagenet-v2)](https://paperswithcode.com/sota/image-classification-on-imagenet-v2?p=revisiting-weakly-supervised-pre-training-of)


## Requirements
This code has been tested to work with Python 3.8, PyTorch 1.10.1 and torchvision 0.11.2. 

Note that CUDA support is not required for the tutorials.

To setup PyTorch and torchvision, please follow PyTorch's getting started [instructions](https://pytorch.org/get-started/locally/). If you are using conda on a linux machine, you can follow the following setup instructions  - 
```console
conda create --name swag python=3.8
conda activate swag
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Model Zoo

We share checkpoints for all the pretrained models in the paper, and their ImageNet-1k finetuned counterparts. The models are available via [torch.hub](https://pytorch.org/docs/stable/hub.html), and we also share URLs to all the checkpoints. 

The details of the models, their torch.hub names / checkpoint links, and their performance on Imagenet-1k (IN-1K) are listed below.

| Model | Pretrain Resolution | Pretrained Model | Finetune Resolution | IN-1K Finetuned Model | IN-1K Top-1 | IN-1K Top-5 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| RegNetY 16GF | 224 x 224 | [regnety_16gf](https://dl.fbaipublicfiles.com/SWAG/regnety_16gf.torch) | 384 x 384 | [regnety_16gf_in1k](https://dl.fbaipublicfiles.com/SWAG/regnety_16gf_in1k.torch) | 86.02% | 98.05% |
| RegNetY 32GF | 224 x 224 | [regnety_32gf](https://dl.fbaipublicfiles.com/SWAG/regnety_32gf.torch) | 384 x 384 | [regnety_32gf_in1k](https://dl.fbaipublicfiles.com/SWAG/regnety_32gf_in1k.torch) | 86.83% | 98.36% |
| RegNetY 128GF | 224 x 224 | [regnety_128gf](https://dl.fbaipublicfiles.com/SWAG/regnety_128gf.torch) | 384 x 384 | [regnety_128gf_in1k](https://dl.fbaipublicfiles.com/SWAG/regnety_128gf_in1k.torch) | 88.23% | 98.69% |
| ViT B/16 | 224 x 224 | [vit_b16](https://dl.fbaipublicfiles.com/SWAG/vit_b16.torch) | 384 x 384 | [vit_b16_in1k](https://dl.fbaipublicfiles.com/SWAG/vit_b16_in1k.torch) | 85.29% | 97.65% |
| ViT L/16 | 224 x 224 | [vit_l16](https://dl.fbaipublicfiles.com/SWAG/vit_l16.torch) | 512 x 512 | [vit_l16_in1k](https://dl.fbaipublicfiles.com/SWAG/vit_l16_in1k.torch) | 88.07% | 98.51% |
| ViT H/14 | 224 x 224 | [vit_h14](https://dl.fbaipublicfiles.com/SWAG/vit_h14.torch) | 518 x 518 | [vit_h14_in1k](https://dl.fbaipublicfiles.com/SWAG/vit_h14_in1k.torch) | 88.55% | 98.69% |

The models can be loaded via torch hub using the following command -

```python
model = torch.hub.load("facebookresearch/swag", model="vit_b16_in1k")
```

## Inference Tutorial

For a tutorial with step-by-step instructions to perform inference, follow our [inference tutorial](inference_tutorial.ipynb) and run it locally, or [![Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/facebookresearch/swag/blob/main/inference_tutorial.ipynb).

## Live Demo

SWAG has been integrated into [Huggingface Spaces 🤗](https://huggingface.co/spaces) using [Gradio](https://github.com/gradio-app/gradio). Try out the web demo on [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/akhaliq/SWAG).

Credits: [AK391](https://github.com/AK391)

## ImageNet 1K Evaluation

We also provide a script to evaluate the accuracy of our models on ImageNet 1K, [imagenet_1k_eval.py](imagenet_1k_eval.py). This script is a slightly modified version of the PyTorch ImageNet [example](https://github.com/pytorch/examples/blob/master/imagenet/main.py) which supports our models.

To evaluate the RegNetY 16GF IN1K model on a single node (one or more GPUs), one can simply run the following command -
```console
python imagenet_1k_eval.py -m regnety_16gf_in1k -r 384 -b 400 /path/to/imagenet_1k/root/
```

Note that we specify a `384 x 384` resolution since that was the model's training resolution, and also specify a mini-batch size of `400`, which is distributed over all the GPUs in the node. For larger models or with fewer GPUs, the batch size will need to be reduced. See the PyTorch ImageNet example [README](https://github.com/pytorch/examples/tree/master/imagenet) for more details.

## Citation

If you use the SWAG models or if the work is useful in your research, please give us a star and cite:  

```bibtex
@misc{singh2022revisiting,
      title={Revisiting Weakly Supervised Pre-Training of Visual Perception Models}, 
      author={Singh, Mannat and Gustafson, Laura and Adcock, Aaron and Reis, Vinicius de Freitas and Gedik, Bugra and Kosaraju, Raj Prateek and Mahajan, Dhruv and Girshick, Ross and Doll{\'a}r, Piotr and van der Maaten, Laurens},
      journal={arXiv preprint arXiv:2201.08371},
      year={2022}
}
```

## License
SWAG models are released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details.
