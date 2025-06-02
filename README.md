# BBM409 - Decision Tree & Pruning Assignment

This repository contains my implementation for **Assignment 2** of **BBM409: Introduction to Machine Learning Lab**, completed in Fall 2022 at Hacettepe University.

## 🧠 Assignment Overview

### 📌 Goal
Implement a decision tree classifier (using the ID3 algorithm) on the Employee Attrition dataset. Extend the implementation with a pruning process to prevent overfitting and improve model generalization.

---

## 🏗️ Structure

### Part 1: Decision Tree (ID3)

- **Dataset**: [Kaggle - Employee Attrition Dataset](https://www.kaggle.com/datasets/patelprashant/employee-attrition)
- **Task**: Predict employee attrition ("Yes" or "No")
- **Techniques**:
  - Implemented ID3 algorithm from scratch
  - Discretized continuous features into 10 intervals
  - Used 5-fold cross-validation
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1 Score
- **Output**:
  - Root-to-leaf rules from the best-performing tree

### Part 2: Pruning

- **Goal**: Prune redundant nodes (twigs) based on validation accuracy
- **Process**:
  - Dataset split into Train (60%), Validation (20%), Test (20%)
  - Iteratively prune the twig with least Information Gain
  - Re-label twig with majority class if validation accuracy doesn't degrade
- **Outcome**:
  - Comparison of pre-pruning and post-pruning performance
  - Analysis of pruned attributes and their impact

---

## 📁 Files

- `assignment2.py`: Core implementation of ID3, discretization, classification, pruning, and metric calculation
- `reportcode.ipynb`: Combined code + report (written using Jupyter Notebook)
- `Assignment2_Fall2022_409.pdf`: Original assignment PDF

---

## 🧪 Results

| Fold | Accuracy | Precision | Recall | F1 Score |
|------|----------|-----------|--------|----------|
| 1    | ...      | ...       | ...    | ...      |
| ...  | ...      | ...       | ...    | ...      |
| Avg  | ...      | ...       | ...    | ...      |

Rules generated from the best tree are printed at the end of the execution.

---

## ⚙️ Technologies

- Python
- Numpy, Pandas
- No external ML libraries were used for modeling (ID3 and pruning implemented from scratch)

---

## 🚀 Run

Make sure `dataset.csv` is available and update the path if needed.

```bash
python assignment2.py
