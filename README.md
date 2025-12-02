# 🫁 Lung Cancer Detection using EfficientNetB0

This project utilizes Deep Learning and Transfer Learning to classify various types of lung cancer from Chest CT scan images. Built upon the **EfficientNetB0** architecture and implemented using **TensorFlow/Keras**, the model is designed to assist in the early detection and categorization of lung anomalies.

## 📌 Overview

Lung cancer remains a leading cause of cancer-related mortality globally. Automated analysis of CT scans can significantly aid radiologists in early diagnosis. This project trains a Convolutional Neural Network (CNN) to classify images into four distinct categories with high precision.

**Key Features:**
* **Transfer Learning:** Leverages pre-trained *EfficientNetB0* weights (ImageNet) for robust feature extraction.
* **Data Augmentation:** Applies rotation, shifting, and shearing to training data to improve model generalization.
* **Comprehensive Evaluation:** Generates Confusion Matrices and detailed Classification Reports.

## 📂 Dataset Classes

The model is trained on the **Chest CT-Scan Cancer Dataset** to recognize the following classes:

1.  **Adenocarcinoma** (Left Lower Lobe_T2_N0_M0_Ib)
2.  **Large Cell Carcinoma** (Left Hilum_T2_N2_M0_IIIa)
3.  **Squamous Cell Carcinoma** (Left Hilum_T1_N2_M0_IIIa)
4.  **Normal** (Healthy lung tissue)

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Deep Learning:** TensorFlow 2.x, Keras
* **Data Manipulation:** NumPy
* **Visualization:** Matplotlib, Seaborn
* **Metrics:** Scikit-learn

## 🧠 Model Architecture

The architecture is designed for efficiency and accuracy:

1.  **Base Layer:** `EfficientNetB0` (Pre-trained on ImageNet, top layers removed).
2.  **Feature Extraction:** First 20 layers frozen; subsequent layers fine-tuned.
3.  **Custom Classification Head:**
    * `GlobalAveragePooling2D`
    * `Dropout(0.5)` for regularization
    * `Dense` output layer with Softmax activation (4 units).
4.  **Optimization:** Adam Optimizer ($lr=1e-4$) with `ReduceLROnPlateau`.

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/your-username/lung-cancer-detection.git](https://github.com/your-username/lung-cancer-detection.git)
cd lung-cancer-detection
pip install tensorflow matplotlib seaborn scikit-learn numpy
