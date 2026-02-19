# GRAIL for ResNet, ViT, and CLIP


## Checkpoints
- resnet18 - trained (private)
- clipvit-b32-model-soups (M. Wortsman)
  - https://github.com/mlfoundations/model-soups/releases/ 
  - https://arxiv.org/abs/2203.05482 
- preactresnet18 and vit-exp (M. Andriushchenko)
  - https://arxiv.org/pdf/2302.07011
  - https://drive.google.com/drive/folders/1LmthJCb3RXBFWjeTOC4UOOl7Ppgg2h7n
  - https://github.com/tml-epfl/sharpness-vs-generalization/tree/main
  - Comments:
    - pip install vit-pytorch==0.40.2 
    - see: https://github.com/tml-epfl/sharpness-vs-generalization/commit/6d73be94eb88dae6d3096647bb24b92244fae18f
    - see training metrics on Google Drive

## Evaluating pipeline on individual checkpoints
- `CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.eval_resnet_compression`
- `CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.eval_preact_resnet_compression`
- `CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.eval_vit_compression`
- `CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.eval_clip_compression`

## Testing pipelines
- `CUDA_VISIBLE_DEVICES=0 python3 -m pipelines.test_resnet_compression --ckpt_dir ../checkpoints/resnet18/adam --method fold`
- `CUDA_VISIBLE_DEVICES=6 python3 -m pipelines.test_iterative_resnet_compression --ckpt_dir ../checkpoints/resnet18/iterative --method fold --prune_fraction 0.2 --iterations 7 --epochs 1 --lr 1e-4 >>output/resnet18/iterative-fold-p0.2-i7-e1-lr0.0001`

## Publication
This part is adopted from [Folding as Projection](https://github.com/osaukh/folding_as_projection). Please cite this work below, if you use the code.

```bibtex
@inproceedings{saukh2026cutless,
  title     = {Cut Less, Fold More: Model Compression through the Lens of Projection Geometry},
  author    = {Saukh, Olga and Wang, Dong and {\v{S}}iki{\'c}, Haris and Cheng, Yun and Thiele, Lothar},
  booktitle = {Proceedings of the International Conference on Learning Representations (ICLR)},
  year      = {2026}
}
