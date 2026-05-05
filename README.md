# Project

# Pneumonia Detection from Chest X‑Rays

## Team Information

| **Name**          | **ID**       | 
| ----------------- | ------------ | 
| Moparthi Aparna   | 700772027    | 

**Institution:** University of Central Missouri 
**Course:**  Computer Science

---

## 📖 Short Introduction

Pneumonia is a leading cause of death among children under five and the elderly, especially in low‑resource settings where expert radiologists are scarce. Chest X‑ray (CXR) is the primary diagnostic tool, but manual interpretation is subjective, time‑consuming, and prone to high miss rates (up to 30% even by experienced radiologists).

This project implements and compares two deep learning models for automated pneumonia detection from paediatric chest X‑rays:

1. **Custom CNN** – a convolutional neural network built from scratch with L2 regularization and dropout.
2. **ResNet50 (Transfer Learning)** – a 50‑layer residual network pretrained on ImageNet, fine‑tuned using a two‑phase strategy (freeze then unfreeze last 20 layers).

We used the public **Chest X‑Ray Images (Pneumonia)** dataset (5,863 images). Extensive preprocessing (resize to 224×224, normalization, augmentation) and a stratified 50/50 train‑validation split were applied to ensure stable evaluation. The fine‑tuned ResNet50 achieved an **F1 score of 0.893** and **AUC of 0.957**, outperforming the custom CNN (F1=0.888, AUC=0.932). The code is fully reproducible, commented, and ready for further research or clinical prototyping.

---

## 🎯 Project Objectives

- Design and train a custom CNN from scratch for binary pneumonia classification.
- Fine‑tune a pretrained ResNet50 model on the same dataset using a two‑phase transfer learning strategy.
- Compare both models using five metrics: **Accuracy, Precision, Recall, F1 Score, and AUC‑ROC**.
- Provide a reproducible pipeline with proper train/validation/test splits (fixing the original 16‑image validation set).
- Demonstrate the clinical advantage of higher recall for screening applications.

---


---

## 📊 Dataset

- **Source:** [Chest X‑Ray Images (Pneumonia) by Paul Mooney](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) on Kaggle.
- **Total images:** 5,863 paediatric chest X‑rays (ages 1–5) from Guangzhou Women and Children’s Medical Center.
- **Original split:**
  - Training: 5,216 images (3,883 pneumonia, 1,333 normal)
  - Test: 624 images (390 pneumonia, 234 normal)
  - Validation: **16 images** (too small – we created a new 50/50 split)
- **New split (stratified):**
  - Training: 2,607 images (1,941 pneumonia, 666 normal)
  - Validation: 2,609 images (1,942 pneumonia, 667 normal)
  - Test: unchanged (624 images)

---

## ⚙️ Methodology

### Preprocessing & Augmentation
- Resize all images to 224×224 pixels.
- Normalization: [0,1] for custom CNN; ImageNet mean/std for ResNet50.
- Data augmentation (training only): random rotations (±15°), width/height shifts (±10%), horizontal flips.

### Model 1: Custom CNN
- Three convolutional blocks (32, 64, 128 filters) each followed by MaxPooling.
- Flatten → Dense(128) with L2 regularization (λ=0.001) → Dropout(0.5) → Dense(1, sigmoid).
- Optimizer: Adam (learning rate 1e‑4), loss: binary cross‑entropy.
- **Trainable parameters:** 11,169,089 (≈42.6 MB).

### Model 2: ResNet50 (Transfer Learning)
- Base: ResNet50 pretrained on ImageNet (output layer removed).
- Custom top: GlobalAveragePooling2D → Dense(256, ReLU) → Dropout(0.5) → Dense(128, ReLU) → Dropout(0.3) → Dense(1, sigmoid).
- **Two‑phase fine‑tuning:**
  - **Phase 1 (5 epochs):** Base frozen; train only top layers with lr=1e‑3.
  - **Phase 2 (5 epochs):** Unfreeze last 20 layers; fine‑tune with lr=1e‑5.
- **Trainable parameters after unfreezing:** ≈2.3 million.

### Training Environment
- Hardware: NVIDIA Tesla T4 GPU (16 GB VRAM) via Google Colab.
- Batch size: 32.
- Epochs: 10 for both models (no early stopping to ensure full training).

### Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score, Area Under the ROC Curve (AUC‑ROC).

---

## 📈 Results

After training on the new 50/50 split and evaluating on the original test set (624 images), we obtained the following performance:

| Model                     | Accuracy | Precision | Recall | F1 Score | AUC‑ROC |
|---------------------------|----------|-----------|--------|----------|---------|
| Custom CNN                | 0.8526   | 0.8465    | 0.9333 | 0.8878   | 0.9322  |
| **ResNet50 (fine‑tuned)** | **0.8542** | 0.8257    | **0.9718** | **0.8928** | **0.9568** |

### Key Insights
- Both models achieve clinically viable performance (F1 > 0.88).
- ResNet50 substantially improves **recall** (97.2% vs 93.3%), reducing missed pneumonia cases from 6.7% to only 2.8% – a critical advantage for screening.
- ResNet50 also shows better overall discrimination (AUC 0.957 vs 0.932).
- Custom CNN has slightly higher precision (fewer false alarms), but the lower recall makes it less suitable for high‑stakes screening.

---

## 🚀 How to Run the Code

### Option 1: Google Colab (Recommended)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `pneumonia_detection.py` or copy its content into a new notebook.
3. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU).
4. Run all cells. The script will automatically download the dataset, preprocess, train both models, and output the comparison table.

### Option 2: Local Machine (with GPU)
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Pneumonia-Detection-Project.git
   cd Pneumonia-Detection-Project
