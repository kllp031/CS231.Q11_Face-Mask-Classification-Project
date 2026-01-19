<p align="center">
  <a href="https://www.uit.edu.vn/" title="University of Information Technology">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="University of Information Technology (UIT)" width="400">
  </a>
</p>

<h1 align="center"><b>CS231.Q11 – Introduction to Computer Vision</b></h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white" />
</p>

---
# **CS231 Course Project: Face Mask Classification**

> This repository contains the implementation of a **Face Mask Classification System**, developed as the final project for **CS231.Q11 – Introduction to Computer Vision** at the **University of Information Technology (UIT – VNU-HCM)**.
>
> The project performs a comparative study between traditional Machine Learning approaches (using **HOG** and **LBP** feature descriptors with **SVM, KNN, and Random Forest** classifiers) and modern **Convolutional Neural Networks (CNN)** to identify individuals wearing masks versus those without masks.
>
> The primary objective of this project is to **analyze and compare the effectiveness of traditional Machine Learning pipelines versus Deep Learning approaches** for the task of **face mask detection**, a binary image classification problem with real-world relevance in public health and surveillance systems.
>
> The project emphasizes **methodological comparison**, **feature representation**, and **performance evaluation**, rather than solely maximizing accuracy through deep models.


<p align="center">
  <img src="static/images/thumbnail.png" alt="thumbnail" width="600">
</p>

---

## Team Information
| No. | Student ID | Full Name | Role | Github | Email |
|----:|:----------:|-----------|------|--------|-------|
| 1 | 23521143 | Nguyen Cong Phat | Leader | [paht2005](https://github.com/paht2005) | 23521143@gm.uit.edu.vn |
| 2 | 23521168 | Nguyen Le Phong | Member | [kllp031](https://github.com/kllp031) | 23521168@gm.uit.edu.vn  |
| 3 | 23520213 | Vu Viet Cuong | Member | [Kun05-AI](https://github.com/Kun05-AI) |  23520213@gm.uit.edu.vn  | 


---

## **Table of Contents**
- [Repository Structure](#repository-structure)
- [Problem Statement](#problem-statement)
- [System Overview](#system-overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Extraction](#feature-extraction)
- [Model Architectures](#model-architectures)
- [Training & Optimization](#training--optimization)
- [Installation](#installation)
- [Usage](#usage)
- [Demo Application](#demo-application)
- [Experimental Results](#experimental-results)
- [Discussion](#discussion)
- [Conclusion & Future Work](#conclusion--future-work)
- [License](#license)

---

## **Repository Structure**
```text
CS231.Q11_Face-Mask-Classification-Project/
├── src/            # Model Training Source Code (jupyter notebook files)
│   ├── CNN/                 	# Deep Learning CNN (Grayscale) training source code
│   ├── HOG_KNN/               # KNN training with HOG features source code
│   ├── HOG_RF/              	# Random Forest training with HOG features source code
│   ├── HOG_SVM/  				# SVM training with HOG features source code
│   ├── LBP_KNN/  				# KNN training with LBP features source code
│   ├── LBP_RF/  				# Random Forest training with LBP features source code
│   └── LBP_SVM/				# SVM training with LBP features source code
│
├── models/
│   ├── yunet.onnx              # Pre-trained Face Detection model (Included)
│   ├── mask_detector_model.h5  # Trained Mask Classification model (Included)
│   └── [others].joblib/.keras        # Large models/caches (Ignored - Download link below)
│
├── docs/                       # Report & presentation
│   ├── 23520213-23521143-23521168_Report.pdf
│   └── 23520213-23521143-23521168_Slide.pdf
│
├── static/                     # Static Assets
│   ├── images/                 # Images for Slide, Report, and Thumbnails
│   ├── results/                # Output images from Flask Web Demo
│   ├── templates/              # Web UI (index.html, indexSVM.html)
│   └── test/                   # Sample test images (e.g., test.jpg)
│
├── uploads/                    # Temporary storage for user-uploaded images
├── demo_webcam.py              # Real-time Webcam detection script
├── demoSVM_image_flask.py      # Flask Web Application script
├── requirements.txt            # Python dependencies
├── LICENSE
├──  .gitignore                  # Git ignore rules
└── README.md # Main project documentation
```

---

## **Problem Statement**

Face mask detection is a practical computer vision problem that requires **robust facial feature representation** under variations in:
- Illumination
- Pose
- Occlusion
- Mask styles and colors

The goal of this project is to:
1. Evaluate whether **hand-crafted features (HOG, LBP)** combined with classical classifiers can compete with CNN-based approaches.
2. Analyze trade-offs between **accuracy, computational cost, and deployment complexity**.
3. Develop a system capable of **real-time inference** using standard consumer hardware.

---

## **System Overview**

The proposed system consists of three main components:

1. **Offline Training Pipeline**
   - Image preprocessing
   - Feature extraction
   - Model training and hyperparameter optimization

2. **Inference Pipeline**
   - Face detection using **YuNet**
   - Feature extraction / CNN inference
   - Classification and post-processing

3. **Deployment Interfaces**
   - Flask-based web application (static image classification)
   - Real-time webcam detection

---

## **Key Features**

- **Binary Face Mask Classification** with high accuracy
- **Comparative Study** between:
  - Traditional ML: HOG/LBP + SVM, KNN, Random Forest
  - Deep Learning: Custom CNN
- **Automated Hyperparameter Tuning**
  - Optuna for ML models
  - Keras Tuner (Hyperband) for CNN
- **Real-time Detection** using webcam input
- **User-friendly Web Interface** built with Flask

---

## **Dataset**

### Face Mask 12K Images Dataset

- **Source**: Kaggle  
  🔗 https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset
- **Total Images**: Approximately **12,000 RGB images**
- **Image Characteristics**:
  - Diverse facial orientations
  - Multiple ethnicities
  - Various mask types and lighting conditions

### Dataset Structure
- **Training Set**: 10,000 images  
- **Validation Set**: 800 images  
- **Test Set**: 992 images  

The dataset is well-balanced between the two classes, making it suitable for unbiased binary classification evaluation.

---

## **Data Preprocessing**

To ensure consistency and reduce computational complexity, the following preprocessing steps were applied:

1. **Resizing**
   - All images resized to **128 × 128 pixels**

2. **Normalization**
   - Pixel intensities scaled to the range **[0, 1]**

3. **Grayscale Conversion**
   - Applied for traditional ML pipelines
   - Reduces dimensionality while preserving structural facial features

---

## **Feature Extraction**

### Histogram of Oriented Gradients (HOG)
- Captures **edge and shape information**
- Effective for representing facial geometry
- Tested configurations:
  - `6 × 3` cells
  - `8 × 2` cells (best-performing)

### Local Binary Patterns (LBP)
- Encodes **local texture patterns**
- Robust to illumination changes
- Useful for modeling fine-grained facial textures

---

## **Model Architectures**

### Traditional Machine Learning Models
- **Support Vector Machine (SVM)** with RBF kernel
- **K-Nearest Neighbors (KNN)**
- **Random Forest**

These models operate on extracted HOG or LBP feature vectors.

### Deep Learning Model
- Custom **Convolutional Neural Network (CNN)**
- Lightweight architecture optimized for grayscale input
- Designed to balance performance and training efficiency

---

## **Training & Optimization**

- **Traditional ML Models**
  - Hyperparameters optimized using **Optuna**
  - Objective: maximize validation accuracy

- **CNN**
  - Optimized using **Keras Tuner (Hyperband)**
  - Tuned parameters include:
    - Number of convolutional layers
    - Filter sizes
    - Learning rate
    - Dropout rate

---

## **Installation**

### 1. Clone repository
```bash
git clone https://github.com/paht2005/CS231.Q11_Face-Mask-Classification-Project.git
cd CS231.Q11_Face-Mask-Classification-Project
```

### 2. Create virtual environment (recommended)
```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```
--- 

## Usage
### 1. Train models
Open and run notebooks in **src/**:
```bash
jupyter notebook
```
- CNN/train-model-CNN-best-grayscale.ipynb
- HOG_KNN/train-model-HOG-KNN_6x3.ipynb
- HOG_KNN/train-model-HOG-KNN_8x2.ipynb
- HOG_RF/train-model-HOG-RF_8x2.ipynb
- HOG_RF/train-model-HOG-RF_6x3.ipynb
- HOG_SVM/train-model-HOG-SVM_6x3.ipynb
- HOG_SVM/train-model-HOG-SVM-8x2.ipynb
- LBP_KNN/train-model-LBP-KNN.ipynb
- LBP_RF/train-model-LBP-RF.ipynb
- LBP_SVM/train-model-LBP-SVM.ipynb

### 2. Run Flask demo
```bash
python demoSVM_image_flask.py
```
Open browser at:
```bash
http://127.0.0.1:5000
```
### 3. Real-time Webcam Detection
```bash
python demo_webcam.py
```
