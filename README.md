# SWAG

This repository contains model checkpoints for the SWAG models (**S**upervised **W**eakly from hasht**AG**s), from the paper [Revisiting Weakly Supervised Pre-Training of Visual Perception Models]().

## Model Zoo 

We share checkpoints for all the pretrained models in the paper, and their ImageNet-1k (IN-1K) finetuned counterparts. 

The details of the models and their [torch hub](https://pytorch.org/docs/stable/hub.html) names are listed below.

| Model | Pretrained Name | Pretrain Resolution | IN-1K Finetuned Name | Finetune Resolution | IN-1K Top-1 | 
| :--- | :--- | :--- | :--- | :--- | :--- |
| ViT H/14 | vit_h14 | 224 x 224 | vit_h14_in1k | 518 x 518 | 88.6% |
| RegNetY 128GF | regnety_128gf | 224 x 224 | regnety_128gf_in1k | 384 x 384 | 88.2% |

The models can be loaded via torch hub using the following command -

```
model = torch.hub.load("facebookresearch/swag", model="vit_b16")
```

## Citation

If you use the SWAG models or if the work is useful in your research, please cite:  

```bibtex
@misc{singh2022revisiting,
      title={Revisiting Weakly Supervised Pre-Training of Visual Perception Models}, 
      author={Mannat Singh, Laura Gustafson, Aaron Adcock, Vinicius de Freitas Reis, Bugra Gedik, Raj Prateek Kosaraju, Dhruv Mahajan, Ross Girshick, Piotr Doll\'ar, Laurens van der Maaten},
      journal={arXiv preprint arXiv:?},
      year={2022},
}
```

## License
SWAG models are released under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for additional details.
