# Alzheimer's Detection from Brain MRI Scans

A deep learning-based project to detect Alzheimer's Disease using brain MRI scans. This project leverages convolutional neural networks (CNNs) and transfer learning techniques (ResNet152) to classify MRI images into respective Alzheimer's stages with high accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Disclaimer & Citation](#disclaimer--citation)
- [Contact](#contact)

---

## Overview

Alzheimer's is a progressive neurodegenerative disorder that affects memory, cognitive function, and behavior. Early detection can significantly improve quality of life and treatment options. This project applies state-of-the-art computer vision and deep learning techniques to identify stages of Alzheimer's from brain MRI scans.

Key goals:
- Build an end-to-end deep learning pipeline for MRI scan classification.
- Use both custom CNN and ResNet152-based transfer learning models.
- Achieve high accuracy, ideally >95%, on validation data.
- Provide clean code, reproducibility, and detailed evaluation.

---

## Dataset

The dataset consists of labeled brain MRI scans categorized into four stages:

- **Non-Demented**
- **Very Mild Demented**
- **Mild Demented**
- **Moderate Demented**

### Dataset Source:
[Kaggle - Alzheimer's MRI Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset)  
(Ensure you follow the dataset license when reusing or redistributing.)

---

## Approach

The pipeline includes:

1. **Data Preprocessing**
   - Resize, normalize, and split dataset into train/val/test
   - One-hot encode the class labels
2. **Model Training**
   - Baseline CNN model
   - Transfer learning using pretrained **ResNet152**
   - Experimentation with dropout, data augmentation, learning rate tuning
3. **Model Evaluation**
   - Confusion matrix, accuracy, precision, recall
   - Visualization of predictions

---

## Model Architecture

### Baseline CNN:
- 3x Conv2D + ReLU + MaxPooling
- Flatten + Dense + Softmax
- Achieved ~85% accuracy after tuning

### Transfer Learning with ResNet152:
- Pretrained ResNet152 (from ImageNet)
- Removed dropout (improved validation accuracy)
- Custom classification head:
  - GlobalAvgPool → Dense(256) → ReLU → Dense(4) + Softmax
- Final Accuracy: **99% training**, **>95% validation**

---

## Results

| Model           | Training Accuracy | Validation Accuracy | Notes                           |
|----------------|-------------------|----------------------|----------------------------------|
| CNN (Custom)   | 81%               | ~85%                | Good baseline                   |
| ResNet152 (TL) | 99%               | **95+%**            | With augmentation & tuning      |

---

## Installation

```bash
# Clone the repo
git clone https://github.com/your-username/Alzheimer-MRI-Detection.git
cd Alzheimer-MRI-Detection

# (Optional) Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## ⚠️ Disclaimer & Citation

> ⚠️ **Academic Integrity & Usage Notice**  
This project was developed for educational purposes as part of the University of Toronto's APS360 deep learning course and individual research initiative. **Do not plagiarize** or submit this as your own in academic settings, coursework, competitions, or hiring evaluations. You may reuse or reference the code **only with proper credit** and acknowledgment.

Using this work without appropriate citation may violate academic integrity policies and result in disciplinary action.

---

### 🔖 Citation

If this work contributes to your academic, professional, or personal projects, please cite it as:
@misc{alzheimermri2025,
  title={Alzheimer’s Detection with Deep Learning},
  author={Hitansh Bhatt, Hwang (William) Wei Ju, Muhammad Irfan, Aryan Ghosh},
  year={2025},
  howpublished={\url{https://github.com/HitanshBhatt/Alzheimer-Detection-APS360-Project}},
  note={APS360 Deep learning coursework project}
}

