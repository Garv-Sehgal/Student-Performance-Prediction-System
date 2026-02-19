# 📘 Student Performance Risk Prediction System

### A Machine Learning + Streamlit Based Student Risk Analyzer

This project is an end-to-end **Student Risk Prediction Web Application** that identifies students who are **likely to be at risk of poor academic performance**. It enables faculty to upload student datasets, automatically analyze performance, generate risk categories, and view interactive visual analytics.

The system uses **Machine Learning (Random Forest)** to predict risk and provides **real-time dashboards** using Streamlit.

---

## 🚀 Key Features

### 🔹 1. Upload & Auto Process CSV

* Upload any CSV with academic performance data
* Automatic cleaning & type conversion
* Department normalization (CS, Engineering, Business, Mathematics)
* Yes/No → numeric conversion

### 🔹 2. Automatic Risk Categorization

Students are categorized into:

* **AT RISK** (Total Score < Threshold)
* **WATCHLIST** (Threshold → Threshold + Buffer)
* **GOOD** (Above both ranges)

Thresholds are adjustable from the sidebar.

### 🔹 3. Synthetic Data Generator (Optional)

Generate additional **low-score synthetic students** to test the model or strengthen the dataset.

### 🔹 4. Machine Learning Model

Algorithm used: **RandomForestClassifier**
Model outputs:

* Prediction Accuracy
* Confusion Matrix
* Feature Importance Chart
* Predicted Category Per Student

### 🔹 5. Interactive Visualizations

* Risk Distribution Bars
* Department vs Risk Heatmap
* Feature Importance Graph
* Prediction vs Actual Comparison

### 🔹 6. Downloadable Results

* Full categorized CSV
* AT RISK student list

---

## 🧠 Machine Learning Overview

### ✔ Algorithm Used

**RandomForestClassifier** (Scikit-Learn)

### ✔ Why Random Forest?

* Handles mixed categorical & numeric data
* Works well with small/medium datasets
* Robust to noise
* Provides feature importance
* Low tuning required

### ✔ ML Pipeline Steps

1. Clean & normalize data
2. Encode categorical features
3. Select relevant numeric features
4. Split dataset (80–20) using stratified sampling
5. Train Random Forest
6. Predict risk categories
7. Evaluate with accuracy + confusion matrix

---

## 📂 Project Structure

```
📁 Student-Risk-Prediction
│── app.py                # Main Streamlit app
│── requirements.txt      # Dependencies
│── README.md             # Complete project documentation
└── sample_dataset.csv    # Optional example dataset
```


## 🧪 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/YOURUSERNAME/YOURREPO.git
cd YOURREPO
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📊 Required Columns in CSV

Minimum required:

```
Total_Score
```

Recommended for better ML performance:

```
Attendance (%)
Midterm_Score
Final_Score
Assignments_Avg
Quizzes_Avg
Participation_Score
Projects_Score
Study_Hours_per_Week
Stress_Level (1-10)
Sleep_Hours_per_Night
Extracurricular_Activities
Internet_Access_at_Home
Department
Gender
Age
```

---

## 🎯 Objective of the Project

* Predict students at risk using machine learning
* Provide early-warning alerts for weak learners
* Help faculty make data-driven decisions
* Build a usable, functional, and deployable ML application

---

## 📘 Why This Is a Strong Final Year Project

* Full ML pipeline (data → processing → model → evaluation → predictions)
* End-to-end deployed web app
* Strong real-world use case (education analytics)
* Clean UI + interactive dashboard
* Practical and scalable
* Demonstrates ML + data engineering + frontend integration
* Highly relevant to current EdTech trends

---

## 🔮 Future Enhancements

* Add SHAP explainability (reason for each prediction)
* Add login authentication for faculty
* Connect to a real-time student database
* Multi-semester trend prediction
* Deploy permanently on Render / AWS / DigitalOcean

---

## 👨‍💻 Developer

**Garv Sehgal**
BTech CSE
