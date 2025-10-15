# ✨ Importance of Feature Selection in Machine Learning

---

## 🚀 Project Overview

This repository documents a **comprehensive comparative study** on the impact of various **Feature Selection (FS)** techniques on the performance, efficiency, and interpretability of common Machine Learning classifiers.

> **💡 The Goal:** The primary objective was to demonstrate how strategically reducing the dimensionality of a dataset can **improve model performance** (maintaining or improving accuracy) while significantly **reducing computational costs and training time.**

## 🛠️ Tech Stack

A snapshot of the core technologies used in this analytical project:

| Category | Tools |
| :--- | :--- |
| **Language** | `Python` |
| **Machine Learning** | `scikit-learn` |
| **Data Handling** | `pandas`, `numpy` |
| **Visualization** | `matplotlib` (For result visualization and Confusion Matrices) |

---

## 🔬 Methodology: A 3-Way Comparative Study

The analysis utilized three distinct datasets and compared three major feature selection categories against three different classifiers.

### 1. Feature Selection Algorithms

The project evaluated techniques across all three major categories:

| Method Type | Description | Algorithms Used |
| :--- | :--- | :--- |
| **Filter Methods** | Selects features based on statistical measures (e.g., correlation), independent of the model. | **Random Forest** & **Decision Tree** Importance Ranking |
| **Wrapper Methods** | Uses the model's performance as the criterion, evaluating different feature subsets. | **Recursive Feature Elimination (RFE)** |
| **Embedded Methods** | Integrates selection into the model training process (e.g., regularization). | Lasso/Ridge techniques (Implicit in some models) |

### 2. Classifiers Tested

The selected feature subsets were benchmarked using:

* **K-Nearest Neighbors (KNN)**
* **Decision Tree Classifier**
* **Support Vector Machine (SVM)**

### 3. Datasets Used
Analysis was conducted across different data types to test generalizability:

| Dataset | Type | Key Attributes |
| :--- | :--- | :--- |
| **Iris** | Multivariate | Petal Length, Petal Width, Sepal Length, Sepal width |
| **Diabetes (Pima)** | Numeric | Pregnancies, Glucose, BMI, Age, etc. |
| **Wine** | Categorical/Complex | Fixed acidity, alcohol, total sulfur dioxide, etc. |

---

## ✅ Key Results & Best Combinations

The study concluded that the optimal combination of FS and Classifier is highly dependent on the dataset type:

| Dataset | Best Feature Selection Algorithm | Best Classifier |
| :--- | :--- | :--- |
| **Iris** (Multivariate) | **Recursive Feature Elimination (RFE)** | **K-Nearest Neighbors (KNN)** |
| **Diabetes** (Numeric) | **Decision Tree (DT) Importance** | **Support Vector Machine (SVM)** |
| **Wine** (Categorical/Complex) | **Random Forest (RF) Importance** | **Support Vector Machine (SVM)** |

---

## 📊 Comparative Accuracy Charts

Visual confirmation of the model performance across all datasets and combinations.

### Iris Dataset Accuracy

![Accuracy Chart of Iris Data](image_38b1e6.png)

### Diabetes Dataset Accuracy

![Accuracy Chart of Diabetes Data](image_38b1e8.png)

### Wine Dataset Accuracy

![Accuracy Chart of Wine Data](image_38b1ea.png)

---

## 📖 Deeper Dive

For a complete and detailed breakdown of the methodology, results, and comprehensive evaluation charts, please refer to the full **Project Report** and the **Presentation (PPT)** available in this repository.
