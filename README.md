<div align="center">

![CordisAI Demo](Heart_disease_look.gif)

# 🫀 CordisAI

### KNN-Powered Heart Disease Risk Prediction

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com)
[![License](https://img.shields.io/badge/License-MIT-00c896?style=for-the-badge)](LICENSE)

> A professional clinical decision-support dashboard for cardiovascular risk assessment — built with Streamlit, powered by K-Nearest Neighbors, trained on 918 real patients.

</div>

---

## ✨ Overview

**CordisAI** transforms raw patient vitals and clinical findings into an interpretable heart disease risk score. Designed with a dark medical-grade UI, it brings together machine learning and interactive data visualization into a single, production-quality Streamlit app.

---
![CordisAI Demo](assets/Heart_disease_look.gif)

## 🚀 Features

| Feature | Description |
|---|---|
| 🔍 **Risk Prediction** | Real-time KNN inference with probability gauge, risk tier badge & class breakdown bar |
| 📊 **Dataset Insights** | Distributions, correlation heatmap, violin plots, sex & chest-pain breakdowns |
| 📈 **Model Performance** | Grouped bar, scatter & radar charts comparing 5 ML algorithms |
| ℹ️ **Feature Reference** | Clinical glossary with normal ranges and preprocessing pipeline docs |
| 🧬 **Patient Radar** | Normalized vitals overlay vs population average |
| 🎨 **Pro Dark UI** | Teal grid background, glassmorphism cards, glowing accents |

---

## 🧠 Model Details

```
Algorithm       →  K-Nearest Neighbors (KNN)
Accuracy        →  86.4%
F1 Score        →  88.2%
Training Size   →  80% of 918 patients
Features        →  15 (after one-hot encoding)
Scaler          →  StandardScaler (fit on full feature set)
```

### Algorithms Benchmarked

| Model | Accuracy | F1 Score |
|---|---|---|
| Logistic Regression | 86.96% | 88.57% |
| **KNN ★** | **86.41%** | **88.15%** |
| Naive Bayes | 85.33% | 86.83% |
| SVM | 84.78% | 86.79% |
| Decision Tree | 76.09% | 78.64% |

---

## 📁 Project Structure

```
cordisai-heart-prediction/
│
├── Heart_disease_app.py        # Main Streamlit application
├── Heart_Disease.ipynb         # EDA + model training notebook
├── heart.xls                   # Dataset (918 patients, CSV format)
│
├── KNN_Heart_Prediction.pkl    # Trained KNN model
├── scaler.pkl                  # Fitted StandardScaler
├── columns.pkl                 # Feature column order
│
└── README.md
```

---
![CordisAI Demo](assets/Heart_disease_look.gif)

## ⚙️ Installation

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/cordisai-heart-prediction.git
cd cordisai-heart-prediction

# 2. Install dependencies
pip install streamlit scikit-learn pandas numpy plotly joblib

# 3. Run the app
streamlit run Heart_disease_app.py
```

---

## 🩺 Input Features

| Feature | Type | Description |
|---|---|---|
| Age | Numeric | Patient age in years |
| Sex | M / F | Biological sex |
| ChestPainType | ASY / ATA / NAP / TA | Type of chest pain |
| RestingBP | Numeric (mmHg) | Resting blood pressure |
| Cholesterol | Numeric (mg/dL) | Serum cholesterol |
| FastingBS | 0 / 1 | Fasting blood sugar > 120 mg/dL |
| RestingECG | Normal / ST / LVH | Resting ECG result |
| MaxHR | Numeric (bpm) | Maximum heart rate achieved |
| ExerciseAngina | Y / N | Exercise-induced angina |
| Oldpeak | Float | ST depression (exercise vs rest) |
| ST_Slope | Up / Flat / Down | Slope of peak exercise ST segment |

---

## 🔬 Preprocessing Pipeline

1. **Zero-value imputation** — `Cholesterol` and `RestingBP` zeros replaced with column mean
2. **One-hot encoding** — Categorical features encoded, first category dropped
3. **Standard scaling** — All 15 model features scaled to μ=0, σ=1 before inference

---

## ⚠️ Disclaimer

> CordisAI is intended for **educational and research purposes only**.
> It is **not** a substitute for professional medical advice, diagnosis, or treatment.
> Always consult a qualified healthcare provider for clinical decisions.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

<div align="center">

Made with 🫀 by **CordisAI**

</div>
