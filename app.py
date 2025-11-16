import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

st.set_page_config(page_title="Student Risk Snapshot", layout="wide")
st.title("Student Risk Snapshot — Auto Categorized On Upload")

# Columns expected
ID_COLS = ["Student_ID","First_Name","Last_Name","Email","Gender","Age","Department","Grade"]
NUM_FEATURES = [
    "Attendance (%)","Midterm_Score","Final_Score","Assignments_Avg","Quizzes_Avg",
    "Participation_Score","Projects_Score","Study_Hours_per_Week",
    "Stress_Level (1-10)","Sleep_Hours_per_Night"
]
YESNO = ["Extracurricular_Activities","Internet_Access_at_Home"]

# ---------------- Sidebar ----------------
st.sidebar.header("Upload")
file = st.sidebar.file_uploader("Upload your Students CSV", type=["csv"])

st.sidebar.header("Settings")
thr = st.sidebar.slider("At-Risk threshold (Total_Score < threshold)", 45.0, 80.0, 50.0, step=1.0)
watch_width = st.sidebar.slider("Watchlist band above threshold (points)", 5, 20, 10, step=1)

st.sidebar.header("Synthetic data (optional)")
add_synth = st.sidebar.checkbox("Add synthetic AT RISK students for demo", value=False)
synth_n = st.sidebar.number_input("How many synthetic students", 5, 200, 20, step=5)

if not file:
    st.info("Upload your CSV to proceed.")
    st.stop()

# ---------------- Load & Clean Data ----------------
df = pd.read_csv(file)

if "Total_Score" not in df.columns:
    st.error("CSV must contain the column: Total_Score")
    st.stop()

# Normalize Department names to avoid duplicates like Math / Mathematics / math
if "Department" in df.columns:
    dept = (
        df["Department"].astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.lower()
    )

    dept = dept.replace({
        "math": "mathematics",
        "mathematics": "mathematics",
        "cs": "cs",
        "computer science": "cs",
        "eng": "engineering",
        "engineering": "engineering",
        "biz": "business",
        "business": "business",
    })

    df["Department"] = dept.str.title()

# Yes/No columns → numeric
for c in YESNO:
    if c in df.columns:
        df[c] = df[c].map({"Yes": 1, "No": 0})

# Convert numerics
df["Total_Score"] = pd.to_numeric(df["Total_Score"], errors="coerce")
for c in NUM_FEATURES:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=["Total_Score"]).copy()

# ---------------- Synthetic AT-RISK Generator ----------------
def make_low_rows(n=20, seed=7, max_score=45.0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        total = float(rng.uniform(max_score - 20, max_score - 1))

        row = {
            "Student_ID": f"SYN{1000+i}",
            "First_Name": rng.choice(["Arjun","Aisha","Rohan","Meera","Neeraj","Sara","Dev","Aarav",
                                      "Kiara","Virat","Ananya","Kabir","Riya","Aditya","Myra","Arnav",
                                      "Simran","Ishaan","Sneha","Ira"]),
            "Last_Name": rng.choice(["Mehra","Kapoor","Verma","Patel","Das","Sharma","Khanna","Khurana",
                                     "Chatterjee","Sen","Bhatt","Nair","Gill","Ghose","Menon","Kaur",
                                     "Reddy","Puri","Chawla","Gupta"]),
            "Email": f"syn{1000+i}@university.com",
            "Gender": rng.choice(["Male","Female"]),
            "Age": int(rng.integers(18, 24)),
            "Department": rng.choice(["CS","Business","Engineering","Mathematics"]),
            "Attendance (%)": float(rng.uniform(40, 75)),
            "Midterm_Score": float(rng.uniform(20, 45)),
            "Final_Score": float(rng.uniform(20, 45)),
            "Assignments_Avg": float(rng.uniform(20, 50)),
            "Quizzes_Avg": float(rng.uniform(10, 40)),
            "Participation_Score": float(rng.uniform(10, 40)),
            "Projects_Score": float(rng.uniform(10, 40)),
            "Total_Score": total,
            "Grade": "F",
            "Study_Hours_per_Week": float(rng.uniform(2, 10)),
            "Extracurricular_Activities": float(rng.choice([0, 1])),
            "Internet_Access_at_Home": float(rng.choice([0, 1])),
            "Parent_Education_Level": rng.choice(["None","High School","Bachelor's","Master's"]),
            "Family_Income_Level": rng.choice(["Low","Medium","High"]),
            "Stress_Level (1-10)": int(rng.integers(5, 10)),
            "Sleep_Hours_per_Night": float(rng.uniform(4.5, 7.5)),
        }
        rows.append(row)

    return pd.DataFrame(rows)

if add_synth:
    synth_df = make_low_rows(int(synth_n), max_score=min(thr, 60))
    # ensure same columns
    for col in df.columns:
        if col not in synth_df.columns:
            synth_df[col] = np.nan
    df = pd.concat([df, synth_df[df.columns]], ignore_index=True)

# ---------------- Categorize ----------------
at_risk_mask = df["Total_Score"] < thr
watch_mask = (df["Total_Score"] >= thr) & (df["Total_Score"] < thr + watch_width)
good_mask = ~(at_risk_mask | watch_mask)

df["Risk_Category"] = np.where(at_risk_mask, "AT RISK",
                        np.where(watch_mask, "WATCHLIST", "GOOD"))

# ---------------- Machine Learning Prediction ----------------
# Prepare features
features = NUM_FEATURES + YESNO
if "Gender" in df.columns:
    features.append("Gender")
if "Department" in df.columns:
    features.append("Department")
if "Age" in df.columns:
    features.append("Age")

# Encode categorical
le_gender = LabelEncoder()
le_dept = LabelEncoder()
df_ml = df.copy()
if "Gender" in df_ml.columns:
    df_ml["Gender"] = le_gender.fit_transform(df_ml["Gender"].fillna("Unknown"))
if "Department" in df_ml.columns:
    df_ml["Department"] = le_dept.fit_transform(df_ml["Department"].fillna("Unknown"))

# Select available features
available_features = [f for f in features if f in df_ml.columns]
X = df_ml[available_features].fillna(0)
y = df_ml["Risk_Category"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Predict on full data
df["Predicted_Risk"] = model.predict(X)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': available_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# ---------------- Tabs ----------------
tab1, tab2, tab3 = st.tabs(["Summary", "Predictions", "Data"])

with tab1:
    # ---------------- Summary + Visualization ----------------
    left, right = st.columns([1.6, 1])

    with left:
        st.subheader("Categorization Summary")
        st.caption(f"AT RISK if Total_Score < {thr}, WATCHLIST = {thr} to {thr + watch_width}")

        counts = df["Risk_Category"].value_counts().reindex(["AT RISK","WATCHLIST","GOOD"]).fillna(0)
        st.bar_chart(counts)

    with right:
        st.metric("Total Students", len(df))
        st.metric("AT RISK", int(counts.get("AT RISK", 0)))
        st.metric("WATCHLIST", int(counts.get("WATCHLIST", 0)))

    # Department category chart
    if "Department" in df.columns:
        st.markdown("### Department × Category")
        pivot = (
            df.pivot_table(index="Department", columns="Risk_Category",
                           values="Total_Score", aggfunc="count")
            .reindex(columns=["AT RISK","WATCHLIST","GOOD"])
            .fillna(0)
        )
        st.bar_chart(pivot)

    # ---------------- AT-RISK List ----------------
    st.markdown("---")
    st.subheader("Students Flagged AT RISK")

    show_columns = [c for c in ID_COLS if c in df.columns] + \
                   [c for c in NUM_FEATURES if c in df.columns] + ["Total_Score","Risk_Category"]

    at_risk_df = df[df["Risk_Category"]=="AT RISK"][show_columns].sort_values("Total_Score")

    if at_risk_df.empty:
        st.info("No students flagged AT RISK at this threshold.")
    else:
        st.dataframe(at_risk_df, use_container_width=True)
        st.download_button(
            "Download AT RISK list (CSV)",
            at_risk_df.to_csv(index=False).encode(),
            "at_risk_students.csv",
            "text/csv"
        )

with tab2:
    st.subheader("Machine Learning Predictions")
    st.write(f"Model Accuracy on Test Set: {accuracy:.2%}")

    # Feature Importance
    st.markdown("### Feature Importance")
    st.bar_chart(feature_importance.set_index('feature'))

    # Prediction vs Actual
    st.markdown("### Prediction vs Actual Risk Category")
    pred_counts = df.groupby(['Risk_Category', 'Predicted_Risk']).size().unstack().fillna(0)
    st.bar_chart(pred_counts)

    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=["AT RISK", "WATCHLIST", "GOOD"])
    cm_df = pd.DataFrame(cm, index=["AT RISK", "WATCHLIST", "GOOD"], columns=["AT RISK", "WATCHLIST", "GOOD"])
    st.markdown("### Confusion Matrix")
    st.dataframe(cm_df)

with tab3:
    # ---------------- Full Table ----------------
    st.subheader("Full Categorized Table")
    show_columns_full = show_columns + ["Predicted_Risk"]
    full = df[show_columns_full].sort_values(["Risk_Category","Total_Score"])
    st.dataframe(full, use_container_width=True)
    st.download_button(
        "Download full table (CSV)",
        full.to_csv(index=False).encode(),
        "categorized_students.csv",
        "text/csv"
    )
