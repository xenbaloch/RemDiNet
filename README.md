# Remove the Distraction (RemDiNet)

**Semantic-SNR Guided Low-Light Image Enhancement under Flexible Supervision**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue. svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-red.svg)](https://pytorch.org/)
[![License:  MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

RemDiNet is a PyTorch implementation for low-light image enhancement that leverages semantic information and SNR (signal-to-noise ratio) guidance for robust, color-preserving improvements, even with limited paired data. 

> **Paper**: *Semantic-SNR Guided Low-Light Image Enhancement under Flexible Supervision*  
> **Code & Pre-trained Models**: [https://github.com/xenbaloch/RemDiNet](https://github.com/xenbaloch/RemDiNet)


## ✨ Features

- **🎯 Semantic-Aware Enhancement**: Uses MobileNetV3 backbone for semantic guidance
- **📡 SNR-Based Distraction Removal**: Intelligent noise and distraction detection
- **🎨 Advanced Color Preservation**: Multiple color consistency mechanisms with trainable global color affine transformation
- **🔄 Flexible Supervision**: Works with paired, unpaired, or mixed training data
- **🎓 Staged Training**: Three-stage training strategy (reconstruction → structure → perceptual)
- **⚡ Efficient Architecture**: Lightweight with gradient accumulation support
- **🔧 Post-Processing**: Optional color enhancement at inference time
- **📈 Analysis Tools**: RGB histogram visualization and detailed metrics logging

---

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
├── test_data/         # Low-light test images
│   └── ... 
└── test_gt/           # Ground truth for evaluation (optional)
    └── ...
```

### Supported Datasets

RemDiNet works with popular low-light datasets:

- **LOL** (Low-Light dataset)
- **LOL-v2** (Real and Synthetic)
- **MIT-Adobe FiveK**
- **SICE**
- Custom datasets

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

### Monitoring Training

Training progress is displayed in real-time: 

```
Epoch | Stage | Train PSNR | Val PSNR | Val SSIM | Saturation | Mix | Status
--------------------------------------------------------------------------------
    0 |     1 |       15.3 |     16.2 |   0.7234 |       0.21 | 1.00 | OK
    5 |     1 |       18.7 |     19.1 |   0.7891 |       0.23 | 1.00 | OK
   15 |     2 |       21.4 |     21.8 |   0.8456 |       0.19 | 0.50 | OK (transition)
```

## 💾 Pre-trained Models

Pre-trained weights will be available at:
- [GitHub Releases](https://github.com/xenbaloch/RemDiNet/releases)

**Coming soon:**
- Model trained on LOL dataset
- Model trained on LOL-v2-real
- Lightweight variant (no semantic guidance)

---

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

Copyright (c) 2026 Zain Mehdi

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

- MobileNetV3 pre-trained weights from torchvision
- Metrics implementations inspired by LPIPS, MS-SSIM libraries
- Dataset support for LOL, LOL-v2, and related benchmarks

---

## 🔄 Updates

- **2026-01**:  Initial public release
- Pre-trained models coming soon
- Paper submission in progress

---

**Happy Enhancing! 🌟**

If you find this work useful, please consider giving it a ⭐ on GitHub! 

The source code and pre-trained models will be available at https://github.com/xenbaloch/RemDiNet. 

