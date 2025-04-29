 💳 Online Payment Fraud Detection using Machine Learning

This project focuses on detecting fraudulent online payment transactions using machine learning algorithms. It aims to assist financial systems in identifying and preventing fraud in real-time with high accuracy and minimal false positives.

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Screenshots](#screenshots)
- [Limitations](#limitations)
- [Author](#author)
- [License](#license)

---

## 🧠 About the Project

Online payment fraud has become a growing concern with the rise in digital transactions. The goal of this project is to use **machine learning** to detect fraudulent transactions based on historical data. The model learns patterns from past legitimate and fraudulent transactions to classify new ones accurately.

---

## 📂 Dataset

- **Source**: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions
- **Features**: 
  - 30 numerical input features (anonymized via PCA)
  - `Time`, `Amount`, and `Class` (target: 0 = normal, 1 = fraud)
- **Highly imbalanced** dataset (~0.17% fraud)

---

## 🧱 Technologies Used

- Python 3.x
- NumPy, Pandas
- Scikit-learn
- Matplotlib, Seaborn
- Imbalanced-learn (SMOTE)
- Jupyter Notebook

---

## 🧠 Model Architecture

- Preprocessing:
  - Handling class imbalance using **SMOTE**
  - Feature scaling with StandardScaler
- Algorithms used:
  - Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- Evaluation Metrics:
  - Accuracy, Precision, Recall, F1-Score, ROC-AUC

---

## 📁 Project Structure

payment-fraud-detection/ ├── dataset/ │ └── creditcard.csv ├── notebook/ │ └── fraud_detection.ipynb ├── model/ │ └── fraud_model.pkl ├── scripts/ │ └── predict.py ├── images/ │ └── confusion_matrix.png ├── requirements.txt └── README.md

yaml
Copy
Edit

---
## 📈 Results

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC
Logistic Regression	99.2%	87.3%	78.9%	82.9%	0.96
Random Forest	99.6%	92.5%	84.7%	88.4%	0.98
XGBoost	99.7%	94.1%	86.2%	89.9%	0.99


## ⚠️ Limitations
Based on anonymized data — real-world data may vary in feature significance.

Highly imbalanced data can affect performance if not handled carefully.

Could benefit from more real-time processing pipeline integration.


## 📌 Screenshots

![image](https://github.com/user-attachments/assets/d1210249-3e44-4f2b-a451-7aebedcb42de)


## ⚙️ Installation

```bash
git clone https://github.com/yourusername/payment-fraud-detection.git
cd payment-fraud-detection
pip install -r requirements.txt
🚀 Usage
Run the notebook:

bash
Copy
Edit
jupyter notebook notebook/fraud_detection.ipynb
Model prediction script:

bash
Copy
Edit
python scripts/predict.py --input data/sample_transaction.csv
Load model for real-time prediction (via joblib or pickle).

---
