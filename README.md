<p align="center">
<img src="grail-llm/figures/grail_logo.png" width="20%"> <br>
</p>

<div align="center">
<h1>GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks</h1>
</div>

## Overview

**GRAIL** is a post-compression weight compensation framework that recovers the performance of structured compressed models. It leverages Gram matrix statistics and ridge regression to compensate for information loss without expensive retraining.
<p align="center">
<img width="100%" alt="image" src="grail-llm/figures/main.png">    
</p>

This repository contains implementations for:
1. **Large Language Models (LLMs)**: LLaMA, LLaMA-2
2. **Vision Models**: ResNet, ViT, CLIP

## Repository Structure

The project is organized into two main modules:

### 1. [GRAIL for LLMs](./grail-llm)
Located in `grail-llm/`, this module supports:
- **Models**: LLaMA-1, LLaMA-2
- **Pruning Methods**: FLAP, Wanda-SP, SlimGPT, Wanda++
- **Features**: Weight & Bias compensation, Zero-shot evaluation
- The code folder is modified from https://github.com/TWWinde/GRAIL_LLM and https://github.com/nanguoyu/simple_model_folding

**[👉 Go to GRAIL-LLM Documentation](./grail-llm/README.md)**

### 2. [GRAIL for Vision](./grail-vision)
Located in `grail-vision/`, this module supports:
- **Models**: ResNet18, ViT, CLIP
- **Features**: Model soups, specialized compensation pipelines
- The code folder is modified from  https://github.com/osaukh/folding_as_projection

**[👉 Go to GRAIL-Vision Documentation](./grail-vision/README.md)**

## Quick Start

### For LLMs
```bash
cd grail-llm
pip install -r requirements.txt  # If available, or see installation in README
python main.py --model meta-llama/Llama-2-7b-hf --prune_method flap --compensate
```

### For Vision Models
```bash
cd grail-vision
# Follow instructions in grail-vision/README.md
```

## Citation

If you use GRAIL in your research, please cite:

```bibtex
@inproceedings{Tang2026GRAIL,
  author    = {Tang, Wenwu. and Wang, Dong and Thiele, Lothar. and Saukh, Olga.},
  title     = {GRAIL: Post-hoc Compensation by Linear Reconstruction for Compressed Networks},
  booktitle = {Proceedings of the Conference on Parsimony and Learning (CPAL)},
  year      = {2026},
  note      = {Accepted (Proceedings Track)}
}
```


## License

MIT
