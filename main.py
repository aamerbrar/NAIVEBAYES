import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


st.set_page_config(page_title="Naive Bayes Trainer", layout="wide")
st.title("🧠 Naive Bayes — Train/Test Split + Accuracy + Confusion Matrix")

st.sidebar.header("1) Upload dataset")
file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv", "xlsx", "xls"])

if file is None:
    st.info("Upload a dataset to start.")
    st.stop()

# ---------- Load ----------
def load_data(f):
    name = f.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(f)
    return pd.read_excel(f)

df = load_data(file)

st.subheader("Dataset Preview")
st.dataframe(df.head(50), use_container_width=True)
st.write(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")

# ---------- Select Target ----------
st.sidebar.header("2) Choose target")
target_col = st.sidebar.selectbox("Target column (y)", df.columns)

# Drop rows where target missing
df = df.dropna(subset=[target_col]).copy()

X = df.drop(columns=[target_col])
y_raw = df[target_col]

# ---------- Train/Test Split ----------
st.sidebar.header("3) Train/Test Split")
test_size = st.sidebar.slider("Test size", 0.1, 0.5, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)

# ---------- Model ----------
st.sidebar.header("4) Model")
model_type = st.sidebar.selectbox("Naive Bayes type", ["GaussianNB (numeric)", "MultinomialNB (counts/non-negative)"])

st.sidebar.caption(
    "Tip:\n"
    "- GaussianNB works best for continuous numeric features.\n"
    "- MultinomialNB needs non-negative features (like counts)."
)

# ---------- Preprocessing ----------
# Identify column types
num_cols = X.select_dtypes(include=["number"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Build preprocessing:
# - numeric: impute missing with median
# - categorical: impute missing with most_frequent + one-hot encode
numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ],
    remainder="drop"
)

# Encode y if needed (for confusion matrix labels + model)
le = LabelEncoder()
y = le.fit_transform(y_raw.astype(str))

# Split
# Count samples per class
class_counts = pd.Series(y).value_counts()
min_class = int(class_counts.min())

# Only stratify if every class has at least 2 samples
use_stratify = min_class >= 2 and len(class_counts) > 1

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=float(test_size),
    random_state=int(random_state),
    stratify=y if use_stratify else None
)

if not use_stratify:
    st.warning(
        "Stratified split disabled because at least one class has < 2 samples. "
        "Use a dataset with more rows per class for best results."
    )
counts = pd.Series(y_raw.astype(str)).value_counts()
valid_classes = counts[counts >= 2].index
df = df[df[target_col].astype(str).isin(valid_classes)].copy()
# Pick model
if model_type.startswith("Gaussian"):
    model = GaussianNB()
else:
    model = MultinomialNB()

# For MultinomialNB, we should enforce non-negative features
# We'll do a small safe transform after preprocessing: clip negatives to 0
class ClipToNonNegative:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # X can be sparse or dense
        try:
            # sparse matrix: clip via data
            import scipy.sparse as sp
            if sp.issparse(X):
                X = X.tocsr(copy=True)
                X.data = np.clip(X.data, 0, None)
                return X
        except Exception:
            pass
        return np.clip(X, 0, None)

# Build full pipeline
steps = [("preprocess", preprocess)]
if model_type.startswith("Multinomial"):
    steps.append(("clip", ClipToNonNegative()))
steps.append(("model", model))

clf = Pipeline(steps=steps)

# ---------- Run ----------
st.subheader("Run Training")
run = st.button("🚀 Train & Evaluate", type="primary")

if run:
    try:
        clf.fit(X_train, y_train)

        # Predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)

        st.success("Done!")

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Training Accuracy", f"{train_acc*100:.2f}%")
        with c2:
            st.metric("Testing Accuracy", f"{test_acc*100:.2f}%")

        # Confusion Matrix
        st.subheader("Confusion Matrix (Test Set)")
        cm = confusion_matrix(y_test, y_pred_test)

        labels = le.inverse_transform(sorted(np.unique(np.concatenate([y_test, y_pred_test]))))
        cm_df = pd.DataFrame(cm, index=[f"True: {x}" for x in labels], columns=[f"Pred: {x}" for x in labels])
        st.dataframe(cm_df, use_container_width=True)

        # Classification Report
        st.subheader("Classification Report (Test Set)")
        report = classification_report(
            y_test, y_pred_test,
            target_names=le.inverse_transform(np.unique(y_test)),
            zero_division=0,
            output_dict=False
        )
        st.code(report)

        # Optional: show processed feature count
        st.caption("Note: categorical columns are one-hot encoded. Model trains on the processed feature matrix.")

    except Exception as e:
        st.error("Training failed. Most common reasons:")
        st.write("- Target column has only 1 class (needs at least 2).")
        st.write("- MultinomialNB chosen but your features are negative / not suitable (try GaussianNB).")
        st.write("- Dataset contains weird mixed datatypes.")
        st.code(str(e))