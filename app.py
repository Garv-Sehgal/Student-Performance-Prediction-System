import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import altair as alt

st.set_page_config(page_title="Student Risk Snapshot", layout="wide")
st.title("Student Risk Snapshot — Auto Categorized On Upload")

ID_COLS = ["Student_ID","First_Name","Last_Name","Email","Gender","Age","Department","Grade"]
NUM_FEATURES = [
    "Attendance (%)","Midterm_Score","Final_Score","Assignments_Avg","Quizzes_Avg",
    "Participation_Score","Projects_Score","Study_Hours_per_Week",
    "Stress_Level (1-10)","Sleep_Hours_per_Night"
]
YESNO = ["Extracurricular_Activities","Internet_Access_at_Home"]

st.sidebar.header("Upload")
uploaded_file = st.sidebar.file_uploader("Upload Students CSV or XLSX", type=["csv","xlsx"])

st.sidebar.header("Options")
threshold = st.sidebar.slider("AT RISK Threshold (Final Score)", min_value=0.0, max_value=100.0, value=45.0, step=0.1)

AT_RISK_CUTOFF = threshold
WATCHLIST_CUTOFF = 60.0

if not uploaded_file:
    st.info("Upload your CSV/XLSX to proceed. The app uses Final_Score for categorization.")
    st.stop()

try:
    if uploaded_file.name.lower().endswith((".xls", ".xlsx")):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Failed to read file: " + str(e))
    st.stop()

df.columns = [c.strip() for c in df.columns]

if "Department" in df.columns:
    df["Department"] = (
        df["Department"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.lower()
    )
    df["Department"] = df["Department"].replace({
        "math": "mathematics", "mathematics": "mathematics",
        "cs": "cs", "computer science": "cs",
        "eng": "engineering", "engineering": "engineering",
        "biz": "business", "business": "business",
    }).str.title()

for c in YESNO:
    if c in df.columns:
        df[c] = df[c].astype(str).str.strip().map({"Yes": 1, "No": 0, "yes": 1, "no": 0, "True": 1, "False": 0}).fillna(df[c])

for c in NUM_FEATURES + ["Total_Score"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

if "Total_Score" not in df.columns or df["Total_Score"].isna().all():
    weights = {
        "Attendance (%)": 0.05,
        "Midterm_Score": 0.25,
        "Final_Score": 0.35,
        "Assignments_Avg": 0.10,
        "Quizzes_Avg": 0.05,
        "Participation_Score": 0.05,
        "Projects_Score": 0.15
    }
    available = [k for k in weights.keys() if k in df.columns]
    if available:
        w = {k: weights[k] for k in available}
        s = sum(w.values())
        total = pd.Series(np.zeros(len(df)))
        for k in w:
            total += df[k].fillna(0) * (w[k] / s)
        df["Total_Score"] = total.round(2)

df = df.dropna(subset=["Final_Score"]).copy()

df["Risk_Category"] = np.where(df["Final_Score"] < AT_RISK_CUTOFF, "AT RISK",
                       np.where(df["Final_Score"] <= WATCHLIST_CUTOFF, "WATCHLIST", "GOOD"))
df["Risk_Category"] = pd.Categorical(df["Risk_Category"], categories=["AT RISK","WATCHLIST","GOOD"], ordered=True)

features = NUM_FEATURES + YESNO
if "Gender" in df.columns:
    features.append("Gender")
if "Department" in df.columns:
    features.append("Department")
if "Age" in df.columns:
    features.append("Age")

df_ml = df.copy()
if "Gender" in df_ml.columns:
    df_ml["Gender"] = LabelEncoder().fit_transform(df_ml["Gender"].fillna("Unknown"))
if "Department" in df_ml.columns:
    df_ml["Department"] = LabelEncoder().fit_transform(df_ml["Department"].fillna("Unknown"))

available_features = [f for f in features if f in df_ml.columns]
X = df_ml[available_features].fillna(0)
y = df_ml["Risk_Category"]

do_model = y.nunique() > 1 and len(df) > 10
if do_model:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    df["Predicted_Risk"] = model.predict(X)
    feature_importance = pd.DataFrame({'feature': available_features, 'importance': model.feature_importances_}).sort_values('importance', ascending=False)
else:
    accuracy = None
    df["Predicted_Risk"] = df["Risk_Category"]
    feature_importance = pd.DataFrame({'feature': available_features, 'importance': [0]*len(available_features)})

tab1, tab2, tab3 = st.tabs(["Summary", "Predictions", "Data"])

with tab1:
    left, right = st.columns([1.6, 1])
    with left:
        st.subheader("Categorization Summary")
        st.caption(f"AT RISK: Final_Score < {AT_RISK_CUTOFF}  •  WATCHLIST: {AT_RISK_CUTOFF}–{WATCHLIST_CUTOFF}  •  GOOD: > {WATCHLIST_CUTOFF}")
        counts = df["Risk_Category"].value_counts().reindex(["AT RISK","WATCHLIST","GOOD"]).fillna(0)
        st.bar_chart(counts)
    with right:
        st.metric("Total Students", len(df))
        st.metric("AT RISK", int(counts.get("AT RISK", 0)))
        st.metric("WATCHLIST", int(counts.get("WATCHLIST", 0)))

    if "Department" in df.columns:
        st.markdown("### Department × Category")
        pivot = df.pivot_table(index="Department", columns="Risk_Category", values="Final_Score", aggfunc="count").reindex(columns=["AT RISK","WATCHLIST","GOOD"]).fillna(0)
        st.bar_chart(pivot)

    st.markdown("---")
    st.subheader("Students Flagged AT RISK")
    show_columns = [c for c in ID_COLS if c in df.columns] + [c for c in NUM_FEATURES if c in df.columns] + ["Total_Score","Risk_Category"]
    at_risk_df = df[df["Risk_Category"]=="AT RISK"][show_columns].sort_values("Final_Score")
    if at_risk_df.empty:
        st.info("No students flagged AT RISK.")
    else:
        st.dataframe(at_risk_df, width='stretch')
        st.download_button("Download AT RISK list (CSV)", at_risk_df.to_csv(index=False).encode(), "at_risk_students.csv", "text/csv")

    st.markdown("---")
    st.subheader("Machine Learning Predictions")
    if accuracy is not None:
        st.write(f"Model Accuracy on Test Set: {accuracy:.2%}")
        st.markdown("### Feature Importance")
        st.bar_chart(feature_importance.set_index('feature'))
        pred_counts = df.groupby(['Risk_Category', 'Predicted_Risk'], observed=False).size().unstack().fillna(0)
        st.markdown("### Prediction vs Actual Risk Category")
        st.bar_chart(pred_counts)
        cm = confusion_matrix(y_test, y_pred, labels=["AT RISK", "WATCHLIST", "GOOD"])
        cm_df = pd.DataFrame(cm, index=["AT RISK", "WATCHLIST", "GOOD"], columns=["AT RISK", "WATCHLIST", "GOOD"])
        st.markdown("### Confusion Matrix")
        st.dataframe(cm_df, width='stretch')
    else:
        st.info("Not enough classes or data to train ML model. Predictions = rule-based Risk_Category")

    st.markdown("---")
    st.subheader("Full Categorized Table")
    show_columns_full = show_columns + ["Predicted_Risk"]
    full = df[show_columns_full].sort_values(["Risk_Category","Final_Score"])
    st.dataframe(full, width='stretch')
    st.download_button("Download full table (CSV)", full.to_csv(index=False).encode(), "categorized_students.csv", "text/csv")

with tab2:
    st.info("Predictions are now displayed in the Summary tab.")

with tab3:
    st.info("Data table is now displayed in the Summary tab.")
