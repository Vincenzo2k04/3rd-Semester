# 🎯 Resume Matched Score Predictor

A machine learning project that predicts how well a resume matches a job description using text vectorization and regression modeling.

---

## 🧠 Project Summary

This project uses Python to predict a `matched_score` for resumes based on their content. It combines text preprocessing, TF-IDF vectorization, and Ridge Regression to build a predictive model.

---

## ⚙️ Workflow Overview

### 1. 📥 Import Dependencies
Essential libraries used include:
- `pandas`, `numpy`
- `scikit-learn`: `MinMaxScaler`, `TfidfVectorizer`, `Ridge`, `train_test_split`, `metrics`
- `scipy.sparse`

### 2. 📂 Load Dataset
Reads data from `resume_data.csv`.

### 3. 🧹 Data Preprocessing
- Drops columns with >70% missing values.
- Fills remaining nulls:
  - Text: `"Unknown"`
  - Numeric: column mean
- Removes outliers (IQR method).
- Normalizes numerical features using `MinMaxScaler`.

### 4. 🧱 Feature Engineering
- Applies **TF-IDF vectorization** to:
  - `skills`
  - `responsibilities`
  - `skills_required`
  - `related_skills_in_job`
- Combines vectorized features into a sparse matrix.

### 5. 🤖 Model Training
- Splits data into 80/20 train/test sets.
- Trains a **Ridge Regression** model (`alpha=1.0`) using TF-IDF features to predict `matched_score`.

### 6. 📊 Model Evaluation
- Metrics on the test set:
  - **R² Score**: `0.406`
  - **MAE**: `0.139`
  - **RMSE**: `0.175`

### 7. 🧪 Sample Prediction
- Includes an example of predicting a score for a hardcoded sample resume.

---

## 🧰 Key Libraries
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn scipy
   ```
2. Place `resume_data.csv` in the same directory as the script.
3. Run the Python script:
   ```bash
   python script_name.py
   ```
