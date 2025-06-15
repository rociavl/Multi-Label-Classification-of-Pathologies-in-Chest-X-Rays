# Multi-Label-Classification-of-Pathologies-in-Chest-X-Rays

🫁 Multi-Label Classification of Chest X-ray Pathologies
Deep Learning for Biostatistics - UPC

![image](https://github.com/user-attachments/assets/1167fde5-3107-43ba-9752-fa1ba7374aac)

👥 Authors

Rocío Ávalos - Universitat Politècnica de Catalunya (UPC)

Ainhoa Fraile - Universitat Politècnica de Catalunya (UPC)

Escola d'Enginyeria de Barcelona Est (EEBE)


📋 Project Overview

This project focuses on developing and comparing Convolutional Neural Network (CNN) architectures for multi-label classification of thoracic pathologies in chest X-ray images. Using the NIH ChestX-ray14 dataset, we implement both supervised deep learning models and unsupervised analysis techniques to understand the internal representations learned by the models.

🎯 Objectives

Primary Goal: Develop and compare CNN architectures for multi-label classification of 3-5 thoracic pathologies and explore the internal representations of these models.

Specific Objectives:

- Select and preprocess a subset of the NIH ChestX-ray14 dataset (patient-wise split, 3-5 pathologies)

- Implement and train at least two deep learning architectures for multi-label classification

- Apply transfer learning techniques

- Extract features from intermediate layers of trained models

- Apply PCA for visualization and K-Means for clustering on these features

- Evaluate classification model performance

- Compare architectures and analyze unsupervised learning results in the context of pathologies


🔬 Methodology

📊 Dataset

Source: NIH ChestX-ray14 (NIH Clinical Center, available on Kaggle)

Subset: 5,000-10,000 chest X-ray images

Target Pathologies: 3-5 selected thoracic pathologies

Split Strategy: Strict patient-wise division (train/validation/test)

🖼️ Data Preprocessing

Image resizing (224×224 pixels)

Normalization

Data augmentation (rotations, flips)

Patient-based dataset splitting to prevent data leakage

🧠 Model Architectures

1. Simple CNN from Scratch (PyTorch)

Architecture: 3-4 convolutional layers + pooling + fully connected

Features: ReLU activation, batch normalization, dropout

Framework: Pure PyTorch

2. Transfer Learning with Pre-trained CNN (fast.ai/timm)

Base Models: ResNet18/34, EfficientNet-B0

Strategy: Fine-tuning with classifier head replacement

Frameworks: fast.ai vision_learner and/or PyTorch with timm

📈 Training Configuration

Loss Function: BCEWithLogitsLoss (multi-label classification)

Optimizers: AdamW, Adam

Techniques: Learning rate scheduling, early stopping, overfitting prevention

🔍 Unsupervised Analysis

- Feature Extraction: Activations from deep intermediate layers

- Dimensionality Reduction: PCA for 2D/3D visualization

- Clustering: K-Means analysis on extracted features

- Evaluation: Cluster correspondence to pathology patterns


📊 Results & Evaluation

Performance Metrics

- Multi-label Metrics: Macro and Micro averaged

AUC-ROC, Accuracy, Precision, Recall, F1-Score

![image](https://github.com/user-attachments/assets/4dd770c4-5279-4579-ad11-d69b7ab9426f)


Visualizations: Confusion matrices, ROC curves per class

- Medical Interpretation: Per-pathology diagnostic accuracy

Model Interpretability

- Feature Visualization: PCA projections colored by pathologies

- Clustering Analysis: Pathology pattern discovery

- Medical Validation: Clinical relevance of learned features


🚀 Getting Started

Prerequisites

    bash# Core ML libraries
    torch>=2.0.0
    torchvision
    timm
    fastai

# Data processing

    pandas
    numpy
    scikit-learn
    opencv-python

# Visualization
    matplotlib
    seaborn
    plotly

# Medical imaging
    pydicom
    nibabel

🔧 Installation & Setup

Clone the repository:

    bashgit clone https://github.com/rociavl/Multi-Label-Classification-of-Pathologies-in-Chest-X-Rays
    
    cd chest-xray-classification

Open in Google Colab:

    python# Mount Google Drive for data access
    from google.colab import drive
    drive.mount('/content/drive')

# Install required packages

    !pip install timm fastai

Data Setup:

python# Download NIH ChestX-ray14 dataset
# Place in /content/drive/MyDrive/chest_xray_data/

📓 Running the Analysis

The project is organized in a comprehensive Jupyter notebook with clear sections:

Data Preprocessing & EDA
Model Implementation
Training & Evaluation
Unsupervised Analysis
Results Visualization


📁 Project Structure

    chest-xray-classification/
    ├── notebooks/
    │   ├── Clasificacion_Multi-Etiqueta_Chest_Xray.ipynb
    │   └── data_exploration.ipynb
    ├── src/
    │   ├── models/
    │   │   ├── simple_cnn.py
    │   │   └── transfer_learning.py
    │   ├── preprocessing/
    │   │   └── data_utils.py
    │   └── evaluation/
    │       └── metrics.py
    ├── data/
    │   └── processed/
    ├── results/
    │   ├── models/
    │   ├── figures/
    │   └── reports/
    └── README.md

🔬 Key Findings

Model Performance

- Transfer Learning Models: Achieved superior performance with pre-trained ImageNet weights
- 
- Simple CNN: Competitive results with faster training time
- 
- Multi-label Accuracy: High precision in pathology detection across multiple conditions

Unsupervised Insights

- PCA Visualization: Clear clustering patterns corresponding to different pathologies

- Feature Learning: Models learned clinically relevant anatomical features

- Pathology Grouping: Successful identification of co-occurring conditions


📚 Medical Relevance

This project addresses a critical challenge in automated medical diagnosis, providing:

- Radiologist Support: AI-assisted interpretation of chest X-rays

- Multi-pathology Detection: Simultaneous screening for multiple conditions

- Feature Interpretability: Understanding what the model learns about pathologies

- Clinical Validation: Medically meaningful feature representations



🤝 Contributing

We welcome contributions to improve the project:

Fork the repository

    Create a feature branch (git checkout -b feature/improvement)
    Commit changes (git commit -am 'Add new feature')
    Push to branch (git push origin feature/improvement)
    Create a Pull Request


📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
