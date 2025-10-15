# 💡 Importance of Feature Selection in Machine Learning

This repository documents a comprehensive comparative study on the impact of various **Feature Selection (FS)** techniques on the performance, efficiency, and interpretability of common Machine Learning classifiers.

## Project Goal
The primary objective of feature selection is to improve model performance by reducing the dimensionality of the dataset while maintaining or improving the accuracy of the model. This process also helps to reduce computational costs and training time.

## Datasets
The analysis was conducted across three publicly available, distinct datasets to ensure the generalizability of the findings:

| Dataset | Type | Key Attributes |
| :--- | :--- | :--- |
| **Iris** | Multivariate | Petal Length, Petal Width, Sepal Length, Sepal width |
| **Diabetes (Pima)** | Numeric | Pregnancies, Glucose, BMI, Age, etc. |
| **Wine** | Categorical/Complex | Fixed acidity, alcohol, sulphates, total sulfur dioxide, etc. |

## Methodology

### 1. Feature Selection Algorithms
The project implemented and evaluated techniques spanning all three major categories of feature selection:

* **Filter Methods:** Selection based on statistical measures, such as correlation or mutual information, without taking the model's performance into account. (Algorithms included **Random Forest** and **Decision Tree** feature importance ranking).
* **Wrapper Methods:** Uses the model's performance as the criterion for selection by evaluating the model with different subsets of features. (Algorithm used: **Recursive Feature Elimination (RFE)**).
* **Embedded Methods:** Integrates feature selection into the model-building process, often using regularization techniques like Lasso or Ridge regression.

### 2. Classifiers
The selected feature subsets were then tested and compared using three popular classification algorithms:

* **K-Nearest Neighbors (KNN)**
* **Decision Tree Classifier**
* **Support Vector Machine (SVM)**

## Key Results & Best Combinations

The study demonstrated that the optimal combination of Feature Selection method and Classifier is highly dependent on the dataset's underlying structure and type:

| Dataset | Best Feature Selection Algorithm | Best Classifier |
| :--- | :--- | :--- |
| **Iris** (Multivariate) | Recursive Feature Elimination (RFE) | K-Nearest Neighbors (KNN) |
| **Diabetes** (Numeric) | Decision Tree (DT) Importance | Support Vector Machine (SVM) |
| **Wine** (Categorical/Complex) | Random Forest (RF) Importance | Support Vector Machine (SVM) |

## Technologies Used

* `Python`
* `scikit-learn` (For ML Algorithms and Feature Selection)
* `numpy` (For numerical operations)
* `pandas` (For data handling and processing)
* `matplotlib` (For result visualization and Confusion Matrices)

---

## 📚 Further Information

For a complete and detailed breakdown of the methodology, results, and comprehensive evaluation charts, please refer to the full **Project Report** and the **Presentation (PPT)** available in this repository.