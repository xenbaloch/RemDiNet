# Remove the Distraction (RemDiNet)

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RemDiNet is a PyTorch LLIE framework that suppresses distraction by fusing SNR (Signal-to-Noise Ratio) cues with class-agnostic saliency, enabling robust, color-consistent enhancement and effective learning with limited paired supervision.

> ## 📄 Paper 

[![Paper](https://img.shields.io/badge/Paper-Pattern%20Recognition%202026-blue)](https://www.sciencedirect.com/science/article/pii/S0031320326006084)


## 🔧 Installation

### Requirements

- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration, highly recommended)
- 8GB+ GPU memory (16GB recommended for batch training)

### Step 1: Clone the Repository

```bash
git clone https://github.com/xenbaloch/RemDiNet.git
cd RemDiNet
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using conda
conda create -n remdinet python=3.9
conda activate remdinet

# OR using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements. txt
```

**Core dependencies:**
- PyTorch >= 1.12.0
- torchvision
- numpy
- Pillow
- opencv-python

**Optional (for full metrics):**
- lpips
- pytorch-msssim
- scikit-image
- scipy

---


## 📁 Dataset Preparation

### Directory Structure

Organize your data as follows:

```
your_dataset/
├── train_data/        # Low-light training images
│   ├── img001.png
│   ├── img002.png
│   └── ...
├── train_gt/          # Ground truth (optional for unsupervised)
│   ├── img001.png
│   ├── img002.png
│   └── ...
├── val_data/          # Low-light validation images (optional)
│   └── ...
├── val_gt/            # Validation ground truth (optional)
│   └── ...
├── test_data/         # Low-light test images
│   └── ... 
└── test_gt/           # Ground truth for evaluation (optional)
    └── ...
```

**Note:** 
- If `val_data/` and `val_gt/` are not provided, the training script will automatically split `train_data/` using the `--val_split` ratio (default: 0.2).
- Ground truth directories (`train_gt/`, `val_gt/`, `test_gt/`) are optional for unsupervised training but recommended for evaluation.

### Supported Datasets

RemDiNet works with popular low-light datasets:

- **LOL** (Low-Light dataset)
- **LOL-v2** (Real and Synthetic)
- **SID (sRGB)**
- Custom low-light datasets 

### Data Pairing Strategies

The dataloader supports three GT pairing modes:

1. **`strict`**: Exact filename matching (e.g., `img001.png` ↔ `img001.png`)
2. **`flexible`**: Smart matching with suffix removal (handles `_low`, `_dark`, etc.)
3. **`none`**: Fully unsupervised (no GT required)

Specify via `--gt_pairing` argument in training.

---


## Train/ Test


### Default train run (lowlight_train.py)

--data_root
./data
--experiment_name
RemoveDistraction
--batch_size
8
--num_epochs
200
--use_semantic_guidance
--use_snr_awareness
--use_learnable_snr
--gt_pairing
flexible
--num_workers
4
--stage1_frac
0.10
--stage3_frac
0.25

---

### Default test run (lowlight_test.py)

--model_path
./experiments/RemoveDistraction/checkpoints/best_model.pth
--test_dir
./data/test_data
--output_dir
./results
--gt_dir
./data/test_gt

---

### Training stages

RemDiNet uses a **three-stage training strategy**: 

1. **Stage 1**: Reconstruction warm-up with basic color preservation
2. **Stage 2**: Structure learning with SSIM, edge losses, and color diversity
3. **Stage 3**: Perceptual refinement with VGG-based loss

Smooth transitions between stages prevent training instability.

---

## 📜 License

This project is released under the [MIT License](LICENSE).

```
MIT License

Copyright (c) 2026 Zain Baloch (Muhammad Zain Ul Abideen)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including, without limitation, the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software... 
```

---

## 📧 Contact

**Zain**  
📧 Email: [mzain771@outlook.com](mailto:mzain771@outlook.com)  
🐙 GitHub: [xenbaloch](https://github.com/xenbaloch)  

---


## 🔄 Updates

- **2026-01**:  Initial public release
- Pre-trained models (coming soon)

---

## 📌 Citation

If you find this work useful, please consider citing:

```bibtex
@article{ZAINULABIDEEN2026113643,
  title = {Remove the distraction: Semantic-SNR guided low-light image enhancement under flexible supervision},
  journal = {Pattern Recognition},
  volume = {179},
  pages = {113643},
  year = {2026},
  doi = {https://doi.org/10.1016/j.patcog.2026.113643},
  url = {https://www.sciencedirect.com/science/article/pii/S0031320326006084},
  author = {Muhammad {Zain Ul Abideen} and Benzhuang Zhang and Risheng Liu}
}

