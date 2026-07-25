Official implementation of **"An NSNP-Inspired Multi-Scale Feature Purification Network for Image Forgery Localization"**.

FPNet-NSNP is an image forgery localization framework inspired by the nonlinear spiking neural P system (NSNP). The model introduces NSNP-inspired computational mechanisms into deep neural networks and integrates multi-scale feature perception, cross-layer feature fusion, and feature purification strategies to improve the localization accuracy of tampered regions in digital images.

The proposed framework is designed for **pixel-level image forgery localization**, aiming to identify and accurately locate manipulated regions in forged images.

## Network Architecture

The overall architecture of FPNet-NSNP is shown in:
FPNet-NSNP/Figure

The network structure diagram and related experimental figures are provided in the `Figure` folder of this repository.

## Dataset

The datasets used in our experiments can be downloaded from the following link:

**Dataset download:**

Baidu Netdisk:

Link:https://pan.baidu.com/s/1gdl06ip1HdVpWWrkwVdE2g?pwd=ejgy

Extraction code:
ejgy


The dataset contains the image forgery localization datasets used in our experiments.

## Environment Configuration

The experiments were conducted under the following environment:

- Python 3.10
- TensorFlow 2.13.1
- NumPy 1.24.3
- Pandas 2.0.3
- Matplotlib 3.7.5
- Scikit-learn 1.3.2
- SciPy 1.10.1
- CUDA 11.4

GPU environment:

- NVIDIA GPU
- NVIDIA Driver 470.223.02


## Training Details

The FPNet-NSNP model was trained from scratch without using any pretrained backbone.

Main experimental settings:

- Task: Image forgery localization
- Training strategy: End-to-end training
- Loss function: Binary Cross Entropy (BCE) + Dice Loss
- Optimizer: Adam
- Data augmentation: Applied during training
- Weight decay: Applied to reduce overfitting
- Evaluation metrics:
  - Precision
  - Recall
  - F1-score
  - AUC


## Contact

For questions or discussions about this work, please contact the authors.
