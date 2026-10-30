"""
Live demo: Sleep Disorder Classification (ML + Genetic Algorithm)
Run:  streamlit run app.py
Needs sleep_disorder_classification.py in the same folder.
Data: place Sleep_health_and_lifestyle_dataset.csv next to this file,
      or upload it in the sidebar, or click "Download from Kaggle".
"""
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from sleep_disorder_classification import (CLASS_NAMES, SEED, build_model,
                                           evaluate, genetic_search, load_data)

MODELS = ["KNN", "SVM", "DT", "RF", "ANN"]
LOCAL_CSV = "Sleep_health_and_lifestyle_dataset.csv"

st.set_page_config(page_title="Sleep Disorder Classifier", page_icon="😴", layout="wide")
st.title("😴 Sleep Disorder Classification")
st.caption("KNN · SVM · Decision Tree · Random Forest · ANN, with Genetic Algorithm tuning. "
           "Screening aid only - not a medical diagnosis.")

# ---------------------------------------------------------------- data source
st.sidebar.header("Dataset")
upload = st.sidebar.file_uploader("Upload the Kaggle CSV", type="csv")
if st.sidebar.button("Download from Kaggle"):
    try:
        import kagglehub
        folder = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
        pd.read_csv(glob.glob(os.path.join(folder, "*.csv"))[0]).to_csv(LOCAL_CSV, index=False)
        st.sidebar.success("Downloaded.")
    except Exception as e:  # network / auth problems
        st.sidebar.error(f"Download failed: {e}")

if upload is not None:
    raw = pd.read_csv(upload)
elif os.path.exists(LOCAL_CSV):
    raw = pd.read_csv(LOCAL_CSV)
else:
    st.info("Upload the dataset CSV in the sidebar (or place "
            f"`{LOCAL_CSV}` next to app.py) to start.")
    st.stop()

if upload is not None:
    upload.seek(0)
X, y, full = load_data(upload if upload is not None else LOCAL_CSV)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
st.sidebar.write(f"Rows: **{len(X)}**  |  Train/Test: **{len(X_tr)}/{len(X_te)}**")


# ---------------------------------------------------------------- training
@st.cache_resource(show_spinner="Training default models...")
def train_default(n_rows):
    out = {}
    for n in MODELS:
        m = build_model(n).fit(X_tr, y_tr)
        pred, met = evaluate(m, X_te, y_te)
        out[n] = (m, pred, met)
    return out


@st.cache_resource(show_spinner="Running genetic algorithm (about 1-2 minutes)...")
def train_ga(n_rows):
    out = {}
    for n in MODELS:
        p, _ = genetic_search(n, X_tr, y_tr, verbose=False)
        m = build_model(n, p).fit(X_tr, y_tr)
        pred, met = evaluate(m, X_te, y_te)
        out[n] = (m, pred, met, p)
    return out


default = train_default(len(X))
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dataset", "🏆 Model comparison", "🔮 Predict", "🔍 Feature importance"])

# ---------------------------------------------------------------- tab 1
with tab1:
    st.subheader("Sample of the data")
    st.dataframe(raw.head(15), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("Class distribution")
        st.bar_chart(y.map(dict(enumerate(CLASS_NAMES))).value_counts())
    with c2:
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(full.corr(), cmap="YlGnBu", ax=ax, annot=False)
        ax.set_title("Correlation heatmap")
        st.pyplot(fig)

# ---------------------------------------------------------------- tab 2
with tab2:
    st.subheader("Default parameters (test set, %)")
    base_df = pd.DataFrame({n: default[n][2] for n in MODELS}).round(2)
    st.dataframe(base_df, use_container_width=True)

    use_ga = st.toggle("Also run Genetic Algorithm tuning")
    ga = train_ga(len(X)) if use_ga else None
    if ga:
        st.subheader("GA-tuned parameters (test set, %)")
        ga_df = pd.DataFrame({n: ga[n][2] for n in MODELS}).round(2)
        st.dataframe(ga_df, use_container_width=True)
        st.bar_chart(pd.concat({"Default": base_df.loc["Accuracy"],
                                "GA-tuned": ga_df.loc["Accuracy"]}, axis=1))
        with st.expander("Best parameters found by the GA"):
            st.json({n: ga[n][3] for n in MODELS})
    else:
        st.bar_chart(base_df.loc["Accuracy"])

    pick = st.selectbox("Confusion matrix for", MODELS, index=4)
    pred = (ga[pick][1] if ga else default[pick][1])
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(confusion_matrix(y_te, pred, labels=[0, 1, 2]), annot=True, fmt="d",
                cmap="cool", cbar=False, xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    st.pyplot(fig)

# ---------------------------------------------------------------- tab 3
with tab3:
    st.subheader("Enter a person's details")
    cats = {c: sorted(raw[c].astype(str).unique()) for c in ["Gender", "Occupation", "BMI Category"]}
    c1, c2, c3 = st.columns(3)
    gender = c1.selectbox("Gender", cats["Gender"])
    age = c1.number_input("Age", 18, 90, 35)
    occ = c1.selectbox("Occupation", cats["Occupation"])
    dur = c2.slider("Sleep duration (hours)", 3.0, 10.0, 7.0, 0.1)
    qual = c2.slider("Quality of sleep (1-10)", 1, 10, 6)
    act = c2.slider("Physical activity (min/day)", 0, 150, 45)
    stress = c3.slider("Stress level (1-10)", 1, 10, 5)
    bmi = c3.selectbox("BMI category", cats["BMI Category"])
    sys_bp = c3.number_input("Systolic BP", 80, 200, 120)
    dia_bp = c3.number_input("Diastolic BP", 50, 130, 80)
    hr = c1.number_input("Heart rate (bpm)", 40, 130, 72)
    steps = c2.number_input("Daily steps", 0, 30000, 6000, step=500)
    model_name = c3.selectbox("Model", MODELS, index=4)

    if st.button("Predict", type="primary"):
        row = {"Gender": cats["Gender"].index(gender), "Age": age,
               "Occupation": cats["Occupation"].index(occ), "Sleep Duration": dur,
               "Quality of Sleep": qual, "Physical Activity Level": act, "Stress Level": stress,
               "BMI Category": cats["BMI Category"].index(bmi), "Heart Rate": hr,
               "Daily Steps": steps, "Blood Pressure1": sys_bp, "Blood Pressure2": dia_bp}
        xrow = pd.DataFrame([row])[list(X.columns)]
        model = (ga[model_name][0] if ga else default[model_name][0])
        label = CLASS_NAMES[int(model.predict(xrow)[0])]
        st.success(f"Predicted: **{label}**")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(xrow)[0]
            st.bar_chart(pd.Series(proba, index=[CLASS_NAMES[i] for i in model.classes_]))
        st.caption("Demo for educational use only; not medical advice.")

# ---------------------------------------------------------------- tab 4
with tab4:
    rf = default["RF"][0].steps[-1][1]
    fi = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
    st.subheader("Random Forest feature importance")
    st.bar_chart(fi)
