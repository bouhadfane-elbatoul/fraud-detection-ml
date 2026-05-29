# 💳 Fraud Detection System using Machine Learning

## 📌 Project Overview

This project is a **Fraud Detection System** built using Machine Learning and Streamlit.
It aims to detect fraudulent financial transactions based on transaction patterns such as amount, balance changes, and transaction type.

The system includes:

* Data preprocessing and feature engineering
* Handling class imbalance (SMOTE / Undersampling)
* Multiple ML models comparison
* Real-time prediction using a Streamlit web app
* Performance evaluation using ROC-AUC, F1-score, and confusion matrix

---

## 📊 Dataset

The dataset used is the **PaySim synthetic financial dataset**, which simulates mobile money transactions.

🔗 Dataset source:
https://www.kaggle.com/datasets/ealaxi/paysim1

⚠️ Note: The dataset is not included in this repository due to its large size.

---

## ⚙️ Features Engineering

The following features were created to improve model performance:

* Balance differences (origin & destination)
* Transaction type encoding
* Detection of account draining
* Amount ratio analysis
* Destination balance stability flag

---

## 🤖 Machine Learning Models Used

The following models were trained and compared:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* XGBoost (if available)

---

## ⚖️ Handling Class Imbalance

Fraud detection datasets are highly imbalanced. This project uses:

* SMOTE (Synthetic Minority Oversampling Technique)
* Random Undersampling
* Balanced evaluation metrics

---

## 📈 Evaluation Metrics

Models are evaluated using:

* ROC-AUC Score
* F1 Score
* Precision & Recall
* Confusion Matrix
* Sensitivity & Specificity

---

## 🖥️ Web Application (Streamlit)

The project includes an interactive Streamlit dashboard where users can:

* Input transaction details
* Get real-time fraud prediction
* View probability score
* Analyze model performance visually

---

## 🚀 How to Run the Project

### 1. Clone repository

```bash
git clone https://github.com/bouhadfane-elbatoul/fraud-detection-ml.git
cd fraud-detection-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Streamlit app

```bash
streamlit run app/fraud_detection_app.py
```

---

## 📦 Requirements

* Python 3.8+
* pandas
* numpy
* scikit-learn
* imbalanced-learn
* xgboost
* streamlit
* plotly

---

## 🧠 Project Structure

```
fraud-detection-ml/
│
├── app/
│   └── fraud_detection_app.py
│
├── data/ (not included)
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🎯 Results

The best performing model achieves:

* High ROC-AUC score
* Strong fraud recall (important for detection)
* Balanced precision-recall tradeoff

---

## 👩‍💻 Author

**Bouhadfane Elbatoul**

---

## 📌 Key Insight

Fraud detection is a highly imbalanced classification problem where **recall is more important than accuracy**, because missing a fraud case is more critical than a false alarm.

---
