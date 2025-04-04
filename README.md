# Alzheimer's Detection from Brain MRI Scans - APS360 Deep Learning Project

A deep learning-based project to detect Alzheimer's Disease using brain MRI scans. This project leverages convolutional neural networks (CNNs) and transfer learning techniques (ResNet152) to classify MRI images into respective Alzheimer's stages with high accuracy.

---

## 📚 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Acknowledgements](#acknowledgements)
- [Disclaimer & Citation](#disclaimer--citation)

---

## 🧾 Overview

Alzheimer's is a progressive neurodegenerative disorder. Early detection can help with better disease management. This project applies state-of-the-art computer vision techniques to identify stages of Alzheimer's from brain MRI scans.

The pipeline involves:

- Data cleaning and preprocessing.
- One-hot encoding of labels.
- CNN training from scratch.
- Transfer learning with ResNet152.
- Evaluation using confusion matrix and accuracy metrics.
- Hyperparameter tuning and data augmentation for performance improvements.

---

## 📂 Dataset

The dataset consists of labeled brain MRI scans categorized into:

- **Mild Demented**
- **Moderate Demented**
- **Non-Demented**
- **Very Mild Demented**

### Sources:
The dataset used is publicly available from [Kaggle](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset) or similar repositories. Ensure compliance with the dataset's license if used elsewhere.

---

## 🧠 Approach

We started with a basic CNN architecture and achieved ~81% validation accuracy. Upon applying transfer learning using **ResNet152**, our accuracy improved significantly.

### Techniques Used:
- **Data Augmentation**
- **Transfer Learning (ResNet152)**
- **Confusion Matrix Analysis**
- **Optimization with Adam**
- **Model Evaluation with Validation Split**

---

## 🏗️ Model Architecture

### 1. Baseline CNN:
- 3 convolutional layers
- MaxPooling + ReLU
- Dense + Softmax
- Accuracy: ~85% (after tuning)

### 2. Transfer Learning with ResNet152:
- Removed dropout for better convergence
- Fine-tuned ResNet layers
- Added custom classification head
- **Final Accuracy:** ✅ **99% on training**, **~95% on validation**

---

## 📊 Results

| Model           | Training Acc | Validation Acc | Confusion Matrix |
|----------------|--------------|----------------|------------------|
| CNN (Baseline) | 81%          | 85%            | Moderate         |
| ResNet152 TL   | 99%          | **95+%**       | Excellent        |

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/Alzheimer-MRI-Detection.git
cd Alzheimer-MRI-Detection

# (Optional) Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
