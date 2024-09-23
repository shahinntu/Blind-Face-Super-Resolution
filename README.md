# Blind Face Super-Resolution

This project focuses on the super-resolution of low-quality facial images using deep learning models. It combines real-time image degradation and advanced super-resolution techniques to enhance the quality of facial images.

## Table of Contents
- [Project Overview](#project-overview)
- [Folder Structure](#folder-structure)
- [Setup and Installation](#setup-and-installation)
- [Data Preparation](#data-preparation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Training Process](#training-process)
- [Evaluation](#evaluation)

## Project Overview

This project involves the application of various neural network architectures, including SwinIR, to enhance the resolution of degraded facial images. It examines the balance between performance and computational efficiency within specified model constraints.

## Folder Structure

```
Blind Face Super-Resolution/
│
├── configs/                                # Training configuration files
│   ├── train_config_gan.json               # GAN training configurations
│   ├── train_config_net.json               # PSNR-based training configurations
│
├── data/                                   # Dataset directories
│   ├── results/                            # Output results from models
│   ├── test/                               # Test datasets
│   │   └── LQ/                             # Test low-quality images
│   ├── test_real/                          # Real-world test images
│   │   └── LQ/                             # Real-world test low-quality images
│   ├── train/                              # Training datasets
│   │   ├── GT/                             # Ground truth high-quality images
│   │   └── meta_info_FFHQ5000sub_GT.txt    # Metadata for training images
│   ├── val/                                # Validation datasets
│   │   ├── GT/                             # Validation high-quality images
│   │   └── LQ/                             # Validation low-quality images
│
├── model_logs/                             # Logs and model output files
├── notebooks/                              # Jupyter notebooks for analysis
├── report/                                 # Contains the final project report
├── scripts/                                # Shell scripts for training models
├── src/                                    # Source code
│   ├── data/                               # Data handling modules
│   ├── model/                              # Model definitions and utilities
│
├── .gitignore                              # Gitignore file
├── Blind Face Super-Resolution.code-workspace # Workspace file
├── README.md                               # This README file
├── requirements.txt                        # Python dependencies
└── train_model.py                          # Script to train models
```

## Setup and Installation

1. Clone the repository and navigate into the project directory:

   ```bash
   git clone https://github.com/shahinntu/Blind-Face-Super-Resolution.git
   cd Blind-Face-Super-Resolution
   ```
2. Install required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

Before starting the training, ensure the training and validation datasets are correctly set up:
- Place training images in the `data/train/GT/` directory. These images should be high-quality facial images.
- Metadata for training images should be in `data/train/meta_info_FFHQ5000sub_GT.txt` in the following format.

  ```
  00000.png (512x512x3)
  00001.png (512x512x3)
  ...
  00010.png (512x512x3)
  ```
- Place corresponding low-quality (LQ) and ground truth (GT) validation images in `data/val/LQ/` and `data/val/GT/` directories, respectively.

## Usage

### Training

To initiate training for the PSNR-based or GAN-based models, run the corresponding script from the `scripts/` directory:
```bash
bash scripts/train_model_net.sh  # For PSNR-based model
bash scripts/train_model_gan.sh  # For GAN-based model
```

## Configuration

Edit the configuration files in the `configs/` directory to tweak the training parameters like batch size, learning rate, epochs, etc.

## Training Process

Training details and progress can be monitored through the outputs in the `model_logs/` directory.

Here's the amended "Evaluation" section for your `README.md`:

## Evaluation

To evaluate the models, follow these steps:

1. **Input Correct Model Versions**: Make sure to load the correct restored versions of the trained models from the `model_logs/` directory. You can do this by specifying the paths to the saved models within the evaluation and test notebooks.

2. **Evaluate Models**: Use the `notebooks/06-evaluate-models.ipynb` notebook to evaluate the models based on PSNR (Peak Signal-to-Noise Ratio).

3. **Testing on New Data**:
   - Place the low-quality test images in the `data/test/LQ/` folder.
   - For real-world images, place low-quality test images in the `data/test_real/LQ/` folder.
   
4. **Generate Test Results**: Use the `notebooks/07-generate-test-results.ipynb` to generate test results based on the test images in the folders specified above.
