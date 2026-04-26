# 💼 Salary Intelligence Platform
### ML-Powered Job Salary Prediction System · Portfolio Edition

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![Plotly](https://img.shields.io/badge/Plotly-5.20+-purple?logo=plotly)](https://plotly.com)

---

## 📌 Project Overview

A production-quality machine learning system that predicts annual job salaries based on candidate profile and company attributes. Built to demonstrate ML engineering best practices including sklearn Pipelines, proper train/test isolation, cross-validation, hyperparameter tuning, and a polished Streamlit web application.

**Problem Statement:** HR teams and job seekers often lack objective, data-driven salary benchmarks. This system provides instant, explainable salary estimates grounded in 250,000 real compensation records.

---

## 📊 Dataset

| Property | Value |
|---|---|
| **Rows** | 250,000 salary records |
| **Features** | 9 input columns |
| **Target** | `salary` (annual USD) |
| **Task** | Regression |

**Feature descriptions:**

| Feature | Type | Description |
|---|---|---|
| `job_title` | Nominal | Role/position title |
| `experience_years` | Numerical | Total years in the field |
| `education_level` | Ordinal | High School → PhD |
| `skills_count` | Numerical | Number of listed skills |
| `industry` | Nominal | Business sector |
| `company_size` | Ordinal | Small / Medium / Large |
| `location` | Nominal | City or region |
| `remote_work` | Binary | Yes / No |
| `certifications` | Binary | Yes / No |

---

## 🤖 ML Engineering Highlights

### sklearn Pipeline Architecture
```
Input Dict / DataFrame
        ↓
ColumnTransformer
├── Numerical  → SimpleImputer(median)  + StandardScaler
├── Ordinal    → SimpleImputer(mode)   + OrdinalEncoder(order-aware)
└── Nominal    → SimpleImputer(mode)   + OneHotEncoder(handle_unknown='ignore')
        ↓
Regression Model  (auto-selected best)
        ↓
Predicted Salary
```

**Why Pipeline?**
- ✅ Zero data leakage — fit only on training set
- ✅ Unseen categories handled safely (`handle_unknown='ignore'`)
- ✅ One `.pkl` file = complete reproducible system
- ✅ No manual encoding at prediction time

### Models Trained

| Model | Details |
|---|---|
| Linear Regression | Baseline, regularisation-free |
| Decision Tree | depth=12, min_leaf=50 |
| Random Forest | 200 trees, tuned via RandomizedSearchCV |
| Gradient Boosting | 200 estimators, lr=0.08, tuned |
| XGBoost *(optional)* | 300 estimators if installed |

### Validation Strategy
- **80/20 train/test split** (stratified random, random_state=42)
- **5-Fold cross-validation** on training set
- **RandomizedSearchCV** (n_iter=15, 3-fold) for top 2 models

### Metrics

| Metric | Description | Target |
|---|---|---|
| MAE | Mean Absolute Error (avg dollar error) | < $8,000 |
| RMSE | Root Mean Squared Error | < $12,000 |
| R² | Variance explained | > 0.85 |

---

## 🌐 Streamlit Application

**4 pages:**

| Page | Contents |
|---|---|
| 🏠 Home Dashboard | KPI cards, salary distribution, model comparison table |
| 🔍 Salary Prediction | Input form → instant prediction + salary band + confidence range |
| 📊 Analytics & Insights | Feature importance, salary trends, actual vs predicted, data insights |
| ⬇️ Download Reports | Batch predictions (vectorized), Power BI CSV exports |

---

## 🚀 How to Run Locally

### Prerequisites
- Python 3.11+
- Your Kaggle salary CSV

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/salary-intelligence-platform
cd salary-intelligence-platform

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate    # Mac/Linux
venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your dataset
# Copy Kaggle CSV to:  data/salary_data.csv
```

### Train the Model

```bash
# Full pipeline (recommended)
python main.py

# With cross-validation disabled (faster for testing)
python main.py data/salary_data.csv --no-cv
```

### Launch Web App

```bash
streamlit run app.py
```
Opens at: **http://localhost:8501**

### Test a Single Prediction

```python
from src.predict import predict_one, load_pipeline

pipeline = load_pipeline()

result = predict_one({
    "job_title":        "Data Scientist",
    "experience_years": 7,
    "education_level":  "Master's",
    "skills_count":     10,
    "industry":         "Technology",
    "company_size":     "Large",
    "location":         "New York",
    "remote_work":      "Yes",
    "certifications":   "Yes",
}, pipeline)

print(result["formatted"])       # $124,500
print(result["range_formatted"]) # $109,560 – $139,440
print(result["salary_band"])     # Senior Level
```

---

## 🗂️ Project Structure

```
salary_project/
│
├── app.py                    ← Streamlit multi-page web app
├── main.py                   ← Full pipeline entry point
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── salary_data.csv       ← Your Kaggle dataset (not in repo)
│
├── models/                   ← Auto-created after training
│   ├── salary_pipeline.pkl   ← Complete sklearn pipeline
│   ├── model_comparison.csv  ← All 4+ model results
│   ├── model_metrics.csv     ← Best model metrics
│   └── feature_importance.csv
│
├── outputs/
│   ├── predictions.csv       ← Batch predictions for Power BI
│   └── test_predictions.csv  ← Test set actual vs predicted
│
├── notebooks/
│   └── eda.ipynb             ← Exploratory Data Analysis (optional)
│
└── src/
    ├── utils.py              ← Constants, paths, logging
    ├── preprocess.py         ← sklearn Pipeline + ColumnTransformer
    ├── train.py              ← Training, CV, tuning, model selection
    ├── evaluate.py           ← Metrics, comparison table
    ├── predict.py            ← Single + batch vectorized prediction
    └── insights.py           ← Data-driven business insights
```

---

## ☁️ Deployment

### Streamlit Cloud

```bash
# 1. Push to GitHub (public repo)
git push origin main

# 2. Go to https://share.streamlit.io
# 3. Connect your GitHub repo
# 4. Set main file: app.py
# 5. Add dataset to data/ folder or use st.file_uploader
```

### Render

```bash
# render.yaml
services:
  - type: web
    name: salary-predictor
    env: python
    buildCommand: pip install -r requirements.txt && python main.py
    startCommand: streamlit run app.py --server.port $PORT --server.headless true
```

**Important for cloud deployment:**
- Pre-train the model locally, commit `models/*.pkl` to the repo
- Or add a startup script that trains if no model is found

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| scikit-learn | ML pipeline, models, preprocessing |
| XGBoost | Optional gradient boosting |
| pandas / numpy | Data manipulation |
| Streamlit | Web application |
| Plotly | Interactive charts |
| joblib | Model serialisation |
| matplotlib / seaborn | Static charts |

---

## 📈 Future Improvements

- [ ] Add SHAP values for individual prediction explainability
- [ ] Add salary percentile ranking vs. peers
- [ ] API endpoint (FastAPI) for programmatic predictions
- [ ] Scheduled model retraining pipeline
- [ ] Geographic salary maps with Folium
- [ ] A/B test between model versions
- [ ] Docker containerisation
- [ ] CI/CD with GitHub Actions

---

## 👤 About

Built as a portfolio project demonstrating:
- End-to-end ML engineering (not just model fitting)
- Production-ready code architecture
- Data-driven business insights
- Polished, deployable web application


