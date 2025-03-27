# DermaMNIST Classification and Adversarial Robustness for TinyML compatible models

This project demonstrates the **training, evaluation, and adversarial robustness** of a CNN model on the **DermaMNIST** dataset using TensorFlow and the **Adversarial Robustness Toolbox (ART)**. The pipeline includes model training, adversarial attack generation (FGSM & PGD), and adversarial training for enhanced model robustness.

---

## Features

-  CNN-based classification on the DermaMNIST dataset
-  Model performance evaluation using accuracy, classification report, and confusion matrix
-  Adversarial attack generation using **FGSM** and **PGD**
-  Adversarial training to improve model resilience
-  Visualization of training & validation accuracy
-  Saves and reloads adversarial samples for reuse

---

## Dataset

- **Dataset:** [MedMNIST DermaMNIST](https://medmnist.com/)
- 7-class skin disease image classification dataset with 28×28 RGB images
- Automatically downloaded using `medmnist` Python package

---

## Installation

To run this notebook, install the following packages:

```bash
pip install medmnist seaborn matplotlib adversarial-robustness-toolbox
```

## How It Works
Load & preprocess data using medmnist
Build a CNN model with TensorFlow/Keras
Train the model and evaluate on test data
Visualize performance (accuracy, confusion matrix)
Generate adversarial examples using:
Fast Gradient Sign Method (FGSM)
Projected Gradient Descent (PGD)
Save & reload adversarial data
Adversarially retrain the model
Evaluate model robustness against attacks

## Sample Results
 Original Model Accuracy: ~75.41%
 Under FGSM Attack: ~8.28%
 Under PGD Attack: ~6.18%
 After Adversarial Training:
FGSM Accuracy: ~66.53%
PGD Accuracy: ~71.77%

## Libraries Used
TensorFlow / Keras
MedMNIST
ART (Adversarial Robustness Toolbox)
NumPy, Matplotlib, Seaborn
Scikit-learn

## Future Work
Add support for more advanced attacks (CW, AutoAttack)
Extend to other MedMNIST datasets (e.g., PathMNIST)
Build a web interface for real-time inference & demo
