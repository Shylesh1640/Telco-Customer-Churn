# 📊 Telco Customer Churn Prediction

A machine learning project that predicts customer churn for a telecommunications company using classification algorithms and provides an interactive web dashboard for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)

---

## 🎯 Problem Statement

Customer churn is a critical business challenge in the telecom industry. Acquiring new customers costs 5-25x more than retaining existing ones. This project aims to:
- Identify customers likely to churn using historical data
- Enable proactive retention strategies
- Reduce customer attrition and improve revenue

---

## ✨ Features

- **Exploratory Data Analysis (EDA)** - Visual insights into customer behavior and churn patterns
- **Multiple ML Models** - Comparison of Logistic Regression, Random Forest, and Decision Tree classifiers
- **Hyperparameter Tuning** - GridSearchCV optimization for best model performance
- **Interactive Web App** - Streamlit-powered dashboard for real-time churn predictions
- **Model Persistence** - Serialized trained model for deployment

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.8+ |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn |
| **Web Framework** | Streamlit |
| **Model Serialization** | Joblib |

---

## 📁 Dataset

**Source:** IBM Sample Dataset - Telco Customer Churn

| Feature | Description |
|---------|-------------|
| `customerID` | Unique customer identifier |
| `gender` | Customer gender |
| `SeniorCitizen` | Whether customer is a senior citizen |
| `Partner` | Whether customer has a partner |
| `Dependents` | Whether customer has dependents |
| `tenure` | Number of months with the company |
| `PhoneService` | Whether customer has phone service |
| `MultipleLines` | Whether customer has multiple lines |
| `InternetService` | Type of internet service (DSL, Fiber optic, None) |
| `OnlineSecurity` | Whether customer has online security |
| `OnlineBackup` | Whether customer has online backup |
| `DeviceProtection` | Whether customer has device protection |
| `TechSupport` | Whether customer has tech support |
| `StreamingTV` | Whether customer has streaming TV |
| `StreamingMovies` | Whether customer has streaming movies |
| `Contract` | Contract term (Month-to-month, One year, Two year) |
| `PaperlessBilling` | Whether customer uses paperless billing |
| `PaymentMethod` | Payment method |
| `MonthlyCharges` | Monthly charges |
| `TotalCharges` | Total charges |
| `Churn` | Target variable - Whether customer churned |

**Dataset Size:** 7,043 customers with 21 features

---

## 🔬 Methodology

### 1. Data Preprocessing
- Removed irrelevant features (customerID)
- Encoded categorical variables using Label Encoding
- Handled class imbalance awareness

### 2. Model Training & Comparison
```
Models Evaluated:
├── Logistic Regression
├── Random Forest Classifier
└── Decision Tree Classifier
```

### 3. Hyperparameter Optimization
```python
GridSearchCV Parameters:
- n_estimators: [100, 200, 300]
- max_depth: [5, 10, 15]
- min_samples_split: [2, 5, 10]
- Cross-validation: 5-fold
```

### 4. Model Selection
Best performing model (Random Forest) was selected and serialized for deployment.

---

## 🚀 Installation & Usage

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### Clone Repository
```bash
git clone https://github.com/yourusername/Telco-Customer-Churn.git
cd Telco-Customer-Churn
```

### Run Jupyter Notebook (Training)
```bash
jupyter notebook clasification.ipynb
```

### Launch Web Application
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Telco-Customer-Churn/
│
├── app.py                      # Streamlit web application
├── clasification.ipynb         # ML model training notebook
├── Telco-Customer-Churn.csv    # Dataset
├── churn_model.pkl             # Trained model (generated after training)
└── README.md                   # Project documentation
```

---

## 💻 Web Application Preview

The Streamlit app allows users to:
- Input customer tenure, contract type, and payment method
- Get instant churn prediction (Churn / Not Churn)
- Make data-driven retention decisions

---

## 📈 Key Learnings & Skills Demonstrated

- **Data Science Pipeline** - End-to-end ML workflow from EDA to deployment
- **Feature Engineering** - Data preprocessing and encoding techniques
- **Model Evaluation** - Using accuracy, confusion matrix, and classification reports
- **Hyperparameter Tuning** - GridSearchCV for optimal model configuration
- **Web Development** - Building interactive ML applications with Streamlit
- **Version Control** - Project documentation and code organization

---

## 🔮 Future Enhancements

- [ ] Add more input features to the web app for comprehensive predictions
- [ ] Implement SMOTE for handling class imbalance
- [ ] Add feature importance visualization
- [ ] Deploy on cloud platforms (Heroku/AWS/GCP)
- [ ] Add customer segmentation analysis
- [ ] Implement model monitoring and retraining pipeline

---

## 🤝 Connect With Me

I'm actively seeking **Data Science / Machine Learning internship** opportunities where I can apply my skills in:
- Machine Learning & Predictive Modeling
- Data Analysis & Visualization
- Python Programming
- Building End-to-End ML Solutions

**Feel free to reach out for collaboration or opportunities!**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://www.linkedin.com/in/shylesh1640/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black.svg)](https://github.com/Shylesh1640)
[![Email](https://img.shields.io/badge/Email-Contact-red.svg)](mailto:shylesh1640@gmail.com)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

⭐ **If you found this project helpful, please give it a star!**