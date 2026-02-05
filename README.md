# Remove the Distraction (RemDiNet)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue. svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RemDiNet is a PyTorch implementation for low-light image enhancement that leverages semantic information and SNR (signal-to-noise ratio) guidance for robust, color-preserving improvements, even with limited paired data. 

> **Paper (In Submission)**: *Semantic-SNR Guided Low-Light Image Enhancement under Flexible Supervision*  
> **Code**: [https://github.com/xenbaloch/RemDiNet](https://github.com/xenbaloch/RemDiNet)

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
- **MIT-Adobe FiveK (sRGB)**
- **SID (sRGB)**
- Custom low-light datasets

### Data Pairing Strategies

The dataloader supports three GT pairing modes:

1. **`strict`**: Exact filename matching (e.g., `img001.png` ↔ `img001.png`)
2. **`flexible`**: Smart matching with suffix removal (handles `_low`, `_dark`, etc.)
3. **`none`**: Fully unsupervised (no GT required)

Specify via `--gt_pairing` argument in training.

---


### Training Stages

RemDiNet uses a **three-stage training strategy**: 

1. **Stage 1 (25% epochs)**: Reconstruction warm-up with basic color preservation
2. **Stage 2 (55% epochs)**: Structure learning with SSIM, edge losses, and color diversity
3. **Stage 3 (20% epochs)**: Perceptual refinement with VGG-based loss

Smooth transitions between stages prevent training instability.

---

## 📂 Project Structure

```
RemDiNet/
├── model. py                  # Main model architecture
├── Myloss.py                  # Custom loss functions
├── lowlight_train.py          # Training script
├── lowlight_test.py           # Testing/inference script
├── dataloader.py              # Dataset loader with augmentations
├── metrics.py                 # Image quality metrics
├── metrics_core.py            # Core PSNR/SSIM implementations
├── loss_weights.py            # Loss weight configurations
├── color_enhancement.py       # Color enhancement modules
├── histo_chart.py             # RGB histogram visualization
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── LICENSE                    # MIT License
```

---

### Performance Tips

- Use GPU for 10-100× speedup
- Use mixed precision training (add `torch.cuda.amp` if needed)
- Preprocess/resize large images before batch processing
- Use gradient accumulation for larger effective batch sizes

---


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
🔗 Project:  [https://github.com/xenbaloch/RemDiNet](https://github.com/xenbaloch/RemDiNet)

---

## 🙏 Acknowledgments

- ZeroDCE 
- MobileNetV3 pre-trained weights from torchvision
- Metrics implementations inspired by LPIPS, MS-SSIM libraries
- Dataset support for LOL, LOL-v2, and related benchmarks

---

## 🔄 Updates

- **2026-01**:  Initial public release
- Pre-trained models (coming soon)
- Paper submission (in progress)

---

**Happy Enhancing! 🌟**

If you find this work useful, please consider giving it a ⭐ on GitHub! 

The source code and pre-trained models will be available at https://github.com/xenbaloch/RemDiNet. 

