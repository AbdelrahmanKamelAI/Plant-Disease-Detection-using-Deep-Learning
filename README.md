# Plant-Disease-Detection-using-Deep-Learning
This Jupyter Notebook implements an end-to-end deep learning pipeline for multi-class plant disease classification using computer vision.


# Plant Disease Classification via Transfer Learning 
## Overview
This repository contains a complete deep learning workflow designed to identify and classify plant diseases from leaf imagery. Using a dataset of over 116,000 images, the pipeline leverages **InceptionResNetV2** transfer learning combined with customized classification layers to achieve high-accuracy multi-class disease diagnosis across various agricultural crops.

## Features
- **Comprehensive Crop Coverage:** Classifies healthy and diseased conditions across crops including Apple, Corn, Potato, Rice, Tomato, Watermelon, and more.
- **Advanced Architecture:** Uses pre-trained InceptionResNetV2 weights for robust feature extraction.
- **Robust Pipeline:**
  - Dynamic dataset generation from structured folder hierarchies.
  - Automatic class weighting to address dataset imbalance.
  - Training safeguards including early stopping, learning rate reduction on plateaus, and best-model checkpointing.
  - Comprehensive performance metrics including classification reports and confusion matrices.

## Prerequisites & Installation

Ensure you have Python 3.8+ installed along with the required libraries:

```bash
pip install tensorflow opencv-python pandas numpy scikit-learn matplotlib seaborn
