# 🧠 Importance of Feature Selection in Machine Learning

A **comparative study and analysis** exploring how various **Feature Selection (FS)** techniques impact model accuracy, training efficiency, and interpretability across multiple datasets and classifiers.

Built with 🐍 Python, 📊 scikit-learn, and 🔍 statistical analysis — this project demonstrates the **power of dimensionality reduction** in real-world machine learning workflows.

---

## 🚀 Features

- ⚙️ **Comparative Analysis** of multiple feature selection methods (Filter, Wrapper, Embedded)
- 🤖 **Multi-Classifier Evaluation** — KNN, Decision Tree, and SVM
- 📈 **Dataset Variety** — Iris, Diabetes (Pima), and Wine datasets
- 🧩 **Feature Importance Visualization** using Random Forests and Decision Trees
- 📊 **Performance Benchmarking** with accuracy charts and confusion matrices
- 📉 **Dimensionality Reduction Insights** for efficiency and interpretability

---

## 🧱 Tech Stack

| Category | Tools |
| :--- | :--- |
| **Language** | Python |
| **Machine Learning** | scikit-learn |
| **Data Handling** | pandas, numpy |
| **Visualization** | matplotlib |

---

## 🔬 Methodology Overview

This project performs a **3-way comparative analysis** using three datasets, three classifiers, and three categories of feature selection algorithms.

### 🧠 Feature Selection Algorithms

| Method Type | Description | Algorithms Used |
| :--- | :--- | :--- |
| **Filter Methods** | Selects features based on statistical measures (e.g., correlation) independent of model. | Random Forest & Decision Tree Importance |
| **Wrapper Methods** | Evaluates feature subsets based on model performance. | Recursive Feature Elimination (RFE) |
| **Embedded Methods** | Performs selection during model training (e.g., via regularization). | Lasso / Ridge Regression |

### ⚙️ Classifiers Tested
- K-Nearest Neighbors (KNN)  
- Decision Tree Classifier  
- Support Vector Machine (SVM)

### 🧩 Datasets Used

| Dataset | Type | Key Attributes |
| :--- | :--- | :--- |
| **Iris** | Multivariate | Petal Length, Petal Width, Sepal Length, Sepal Width |
| **Diabetes (Pima)** | Numeric | Pregnancies, Glucose, BMI, Age, etc. |
| **Wine** | Categorical / Complex | Fixed acidity, Alcohol, Total sulfur dioxide, etc. |

---

## ✅ Key Results

| Dataset | Best Feature Selection | Best Classifier |
| :--- | :--- | :--- |
| **Iris** | Recursive Feature Elimination (RFE) | KNN |
| **Diabetes** | Decision Tree Importance | SVM |
| **Wine** | Random Forest Importance | SVM |

> 🏆 The optimal combination of Feature Selection and Classifier **depends on dataset characteristics** — no single method dominates all scenarios.

---

## 📊 Visual Results

### Iris Dataset Accuracy
![Accuracy Chart of Iris Data](Picture2.png)

### Diabetes Dataset Accuracy
![Accuracy Chart of Diabetes Data](Picture1.png)

### Wine Dataset Accuracy
![Accuracy Chart of Wine Data](Picture3.png)

---

## 🛠️ Setup Instructions

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/FeatureSelectionStudy.git
   cd FeatureSelectionStudy
