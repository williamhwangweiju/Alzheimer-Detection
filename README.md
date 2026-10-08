# Alzheimer's Detection from Brain MRI Scans

A deep learning-based project to detect Alzheimer's Disease using brain MRI scans. This project leverages convolutional neural networks (CNNs) and transfer learning techniques (ResNet152) to classify MRI images into respective Alzheimer's stages with high accuracy.

---

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Approach](#approach)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Disclaimer & Citation](#disclaimer-and-citation)
- [Contact](#contact)

---

## Overview

Alzheimer's is a progressive neurodegenerative disorder that affects memory, cognitive function, and behavior. Early detection can significantly improve quality of life and treatment options. This project applies computer vision and deep learning techniques to identify stages of Alzheimer's from brain MRI scans.

Key goals:
- Build an end-to-end deep learning pipeline for MRI scan classification.
- Compare a Random Forest baseline, a custom CNN, and ResNet152-based transfer learning.
- Achieve high accuracy, ideally >95%, on validation data.
- Provide clean code, reproducibility, and detailed evaluation.

---

## Dataset

The dataset consists of labeled brain MRI scans categorized into four stages:

- **Non-Demented**
- **Very Mild Demented**
- **Mild Demented**
- **Moderate Demented**

We used 6,000 images per stage (**24,000 images** in total), resized to 128 × 128.

### Dataset Source:
[Kaggle - Augmented Alzheimer MRI Dataset](https://www.kaggle.com/datasets/uraninjo/augmented-alzheimer-mri-dataset), derived from the OASIS brain MRI collection. The dataset provider had already applied augmentation.
(Ensure you follow the dataset license when reusing or redistributing.)

---

## Approach

The pipeline includes:

1. **Data Preprocessing**
   - Resize images to 128 × 128
   - Encode the four class labels
   - Split the dataset into training and validation sets
2. **Model Training**
   - Random Forest baseline
   - Custom CNN (AlzheimerNet)
   - Transfer learning using pretrained **ResNet152**
3. **Hyperparameter Search**
   - Optuna search over learning rate, batch size, and epochs, maximizing weighted recall on the validation set
4. **Model Evaluation**
   - Accuracy, weighted precision, recall, and F1-score
   - Confusion matrix and visualization of predictions

---

## Model Architecture

### Random Forest Baseline:
- 100 trees trained on flattened 128 × 128 × 3 images, with pixel values normalized to [0, 1]

### Custom CNN (AlzheimerNet):
- Two convolutional blocks, each with two Conv2D layers, batch normalization, max-pooling, and dropout (0.25)
- Dense(512) + dropout (0.5) → Dense(4)

### Transfer Learning with ResNet152:
- Pretrained ResNet152 (from ImageNet)
- Early layers frozen; residual stages `layer2`–`layer4` fine-tuned
- Custom classification head: Dense(2048) → Dense(4096) → Dense(1024) → Dense(4), with ReLU activations and dropout (0.5) after the first two layers
- Trained with AdamW (weight decay 1e-4) and label-smoothed cross-entropy (0.1)
- Final settings: learning rate ≈ 7.7e-5, batch size 32, 10 epochs

---

## Results

Metrics are on the validation set.

| Model                         | Training Accuracy | Validation Accuracy | Weighted F1 |
|-------------------------------|-------------------|---------------------|-------------|
| Random Forest (baseline)      | —                 | 74.3%               | ≈74.1%      |
| ResNet152 (transfer learning) | 99.1%             | **96.5%**           | **96.5%**   |

ResNet152 also reached 96.7% weighted precision and 96.5% weighted recall. Recall by stage:

| Stage              | Recall |
|--------------------|--------|
| Non-Demented       | 88.7%  |
| Very Mild Demented | 98.6%  |
| Mild Demented      | 98.7%  |
| Moderate Demented  | 100.0% |

The model was also tested qualitatively on unused images from the same source and on external MRI images; see Section 9 of the final report.

---

## Usage

The full pipeline is in `Team27_APS360.ipynb` (also available on [Google Colab](https://colab.research.google.com/drive/1dK2eVPZ8EtbMdCNKKIENec7bjbIFtX2t?usp=sharing)).

1. Clone the repo:
   ```bash
   git clone https://github.com/williamhwangweiju/Alzheimer-Detection.git
   ```
2. Download the dataset from Kaggle (link above).
3. Open the notebook in Google Colab (a GPU runtime is recommended) or Jupyter, set the dataset path, and run the cells in order.

Main dependencies: PyTorch, torchvision, scikit-learn, Optuna, NumPy, Pillow, Matplotlib, and seaborn.

---

## Project Structure

| File | Description |
|------|-------------|
| `Team27_APS360.ipynb` | Full pipeline: data preparation, baseline, custom CNN, ResNet152 training, Optuna search, and evaluation |
| `Team27_APS360.pdf` | PDF export of the notebook |
| `APS360 Project Proposal - Team27.pdf` | Project proposal |
| `APS360 Progrees Report - Team 27.pdf` | Progress report |
| `APS360 Final Report.pdf` | Final report |

---

## Disclaimer and Citation

> **Academic Integrity & Usage Notice**  
This project was developed for educational purposes as part of the University of Toronto's APS360 deep learning course and individual research initiative. **Do not plagiarize** or submit this as your own in academic settings, coursework, competitions, or hiring evaluations. You may reuse or reference the code **only with proper credit** and acknowledgment.

Using this work without appropriate citation may violate academic integrity policies and result in disciplinary action.

---

### Citation

If this work contributes to your academic, professional, or personal projects, please cite it as:
```bibtex
@misc{alzheimermri2025,
  title={Alzheimer’s Detection with Deep Learning},
  author={Hitansh Bhatt and Hwang (William) Wei Ju and Muhammad Irfan and Aryan Ghosh},
  year={2025},
  howpublished={\url{https://github.com/HitanshBhatt/Alzheimer-Detection-APS360-Project}},
  note={APS360 Deep learning coursework project}
}
```

## Contact

If you have any questions, suggestions, or want to collaborate, feel free to reach out:

- **Email:** hwangweiju@gmail.com
