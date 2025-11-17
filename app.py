# app.py
import os
import logging
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------- Config ----------------
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)
PIPE_PATH = os.path.join(MODEL_DIR, "pipeline.pkl")
EXPECTED_COLS_PATH = os.path.join(MODEL_DIR, "expected_cols.pkl")
DOCTOR_MAP_PATH = os.path.join(MODEL_DIR, "doctor_map.pkl")
TRAINING_CSV = "training.csv"

# ---------------- Simple data (home remedies & followups) ----------------
home_remedy_dict = {
    "Common Cold": ["Warm fluids and rest", "Steam inhalation", "Stay hydrated"],
    "Fever": ["Stay hydrated", "Paracetamol if needed", "Rest"],
    "Cough": ["Honey & warm water", "Steam inhalation", "Humidifier"],
    "Headache": ["Rest in a quiet/dim room", "Hydration", "Cold or warm compress"],
    "Period Cramps": ["Warm compress on lower abdomen", "NSAIDs (if allowed)", "Light movement/exercise"]
}

followup_questions = {
    "Period Cramps": [
        {"name": "menstrual_now", "q": "Are you currently menstruating (having a period)?", "type": "yesno"},
        {"name": "pain_severity", "q": "Rate belly pain severity (0-10):", "type": "scale", "min": 0, "max": 10}
    ],
    "Appendicitis": [
        {"name": "fever", "q": "Do you have fever?", "type": "yesno"},
        {"name": "pain_location", "q": "Is the pain located on the lower right side of the abdomen?", "type": "yesno"},
        {"name": "pain_severity", "q": "Rate pain severity (0-10):", "type": "scale", "min": 0, "max": 10}
    ],
    "default": [
        {"name": "pain_severity", "q": "Rate your main pain severity (0-10):", "type": "scale", "min": 0, "max": 10},
        {"name": "onset_hours", "q": "How many hours since the problem started?", "type": "number"},
        {"name": "fever", "q": "Do you have fever?", "type": "yesno"}
    ]
}

# ---------------- Globals to hold model & metadata ----------------
pipeline = None
expected_input_columns = None
doctor_map = {}
model_accuracy = None

# ---------------- Utility: Pure-Python fuzzy utilities ----------------
def trapmf(x, a, b, c, d):
    """Trapezoidal membership over scalar x."""
    if x <= a or x >= d:
        return 0.0
    if a < x < b:
        return (x - a) / (b - a) if (b - a) != 0 else 0.0
    if b <= x <= c:
        return 1.0
    if c < x < d:
        return (d - x) / (d - c) if (d - c) != 0 else 0.0
    return 0.0

def trimf(x, a, b, c):
    """Triangular membership."""
    if x <= a or x >= c:
        return 0.0
    if a < x <= b:
        return (x - a) / (b - a) if (b - a) != 0 else 0.0
    if b < x < c:
        return (c - x) / (c - b) if (c - b) != 0 else 0.0
    if x == b:
        return 1.0
    return 0.0

def fuzz_memberships(pain, onset, age):
    """Return dict of membership degrees for fuzzy sets used below."""
    mem = {}
    # pain 0-10 -> low, medium, high
    mem['pain_low'] = trapmf(pain, 0, 0, 2, 4)
    mem['pain_medium'] = trimf(pain, 3, 5, 7)
    mem['pain_high'] = trapmf(pain, 6, 8, 10, 10)
    # onset in hours 0-72 -> recent (0-12), mid (8-48), old (36-72)
    mem['onset_recent'] = trapmf(onset, 0, 0, 6, 12)
    mem['onset_mid'] = trimf(onset, 8, 24, 48)
    mem['onset_old'] = trapmf(onset, 36, 48, 72, 72)
    # age 0-100 -> child/adult/elder
    mem['age_child'] = trapmf(age, 0, 0, 10, 14)
    mem['age_adult'] = trimf(age, 13, 30, 60)
    mem['age_elder'] = trapmf(age, 50, 65, 100, 100)
    return mem

def evaluate_fuzzy_adjustment(pain, onset, age):
    """
    Evaluate a set of fuzzy rules and return a single adjustment multiplier (0.5 - 2.0).
    The function uses membership degrees computed above to weigh rule consequents.
    """
    mem = fuzz_memberships(pain, onset, age)

    # rules: each returns (activation_strength, suggested_multiplier)
    rules = []

    # rule 1: if pain_low and onset_old -> low adjustment (0.6)
    rules.append((min(mem['pain_low'], mem['onset_old']), 0.6))
    # rule 2: if pain_low and onset_recent -> normal (1.0)
    rules.append((min(mem['pain_low'], mem['onset_recent']), 1.0))
    # rule 3: pain_medium & onset_recent -> normal
    rules.append((min(mem['pain_medium'], mem['onset_recent']), 1.0))
    # rule 4: pain_high & onset_recent -> increase (1.6)
    rules.append((min(mem['pain_high'], mem['onset_recent']), 1.6))
    # rule 5: pain_high & onset_mid -> increase (1.4)
    rules.append((min(mem['pain_high'], mem['onset_mid']), 1.4))
    # rule 6: age_elder & pain_high -> increase (1.7)
    rules.append((min(mem['age_elder'], mem['pain_high']), 1.7))
    # rule 7: age_child & pain_medium -> normal (1.0)
    rules.append((min(mem['age_child'], mem['pain_medium']), 1.0))

    # If no rule fired strongly, default is 1.0
    numerator = 0.0
    denominator = 0.0
    for strength, mult in rules:
        numerator += strength * mult
        denominator += strength

    if denominator == 0:
        return 1.0
    return float(numerator / denominator)

# ---------------- Helper: expected columns handling ----------------
def get_expected_columns_from_pipeline(pipe):
    """
    Try to infer expected input column names from the fitted ColumnTransformer (if present).
    If not possible, return a conservative default.
    """
    default_cols = ["Symptoms", "Smoker", "Meds", "Food consumed", "Weather",
                    "Alcohol", "Fruit consumed", "Age", "Weight"]
    try:
        pre = pipe.named_steps.get("pre", None)
        if pre is None:
            return default_cols
        # If ColumnTransformer has attribute _feature_names_in_ or transformers_
        if hasattr(pre, "_feature_names_in"):
            cols = list(pre._feature_names_in)
            if cols:
                return cols
        # transformers_ -> list of tuples (name, transformer, columns)
        cols = []
        for name, transformer, spec in getattr(pre, "transformers_", []):
            if isinstance(spec, (list, tuple)):
                for c in spec:
                    if isinstance(c, str):
                        cols.append(c)
            elif isinstance(spec, str):
                cols.append(spec)
            # if spec is slice or indices, we cannot reconstruct names reliably
        if cols:
            # preserve order & uniqueness
            seen = []
            for c in cols:
                if c not in seen:
                    seen.append(c)
            return seen
    except Exception as e:
        app.logger.exception("Could not infer expected columns from pipeline")
    return default_cols

def ensure_input_has_expected_columns(single_row_df, expected_cols):
    """
    Add missing columns with sensible defaults:
     - numeric (Age/Weight) -> 0
     - else -> 'Unknown'
    Reorder columns to expected_cols if possible.
    """
    df = single_row_df.copy()
    for c in expected_cols:
        if c not in df.columns:
            if c.lower() in ("age", "weight", "onset_hours"):
                df[c] = 0
            else:
                df[c] = "Unknown"
    # Try to reorder
    try:
        return df[expected_cols]
    except Exception:
        return df

# ---------------- Training / Loading ----------------
def train_or_load():
    global pipeline, expected_input_columns, doctor_map, model_accuracy
    # load existing
    if os.path.exists(PIPE_PATH) and os.path.exists(EXPECTED_COLS_PATH):
        app.logger.info("Loading pipeline and expected columns from disk...")
        pipeline = joblib.load(PIPE_PATH)
        expected_input_columns = joblib.load(EXPECTED_COLS_PATH)
        if os.path.exists(DOCTOR_MAP_PATH):
            doctor_map = joblib.load(DOCTOR_MAP_PATH)
        model_accuracy = None  # we don't saved acc separately previously
        return

    # else train if CSV exists
    if not os.path.exists(TRAINING_CSV):
        app.logger.warning("training.csv not found — pipeline not trained.")
        pipeline = None
        expected_input_columns = None
        return

    app.logger.info("Training pipeline from training.csv ...")
    df = pd.read_csv(TRAINING_CSV)
    df.columns = [c.strip() for c in df.columns]

    if "Diesase" not in df.columns:
        app.logger.error("training.csv must contain target column named 'Diesase' (note spelling). Training aborted.")
        pipeline = None
        expected_input_columns = None
        return

    # Basic cleaning
    df = df.replace({pd.NA: None}).copy()
    # numeric filling
    for col in ["Age", "Weight"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)

    y = df["Diesase"]
    # build feature list based on CSV columns
    features = []
    if "Symptoms" in df.columns:
        features.append("Symptoms")
    for c in ["Smoker", "Meds", "Food consumed", "Weather", "Alcohol", "Fruit consumed"]:
        if c in df.columns:
            features.append(c)
    for c in ["Age", "Weight"]:
        if c in df.columns:
            features.append(c)

    if not features:
        app.logger.error("No usable feature columns found in training.csv")
        pipeline = None
        expected_input_columns = None
        return

    X = df[features].copy()

    # Build ColumnTransformer
    transformers = []
    # if Symptoms present -> TF-IDF
    if "Symptoms" in X.columns:
        transformers.append(("text", TfidfVectorizer(stop_words="english", max_features=2000), "Symptoms"))
    cat_cols = [c for c in ["Smoker", "Meds", "Food consumed", "Weather", "Alcohol", "Fruit consumed"] if c in X.columns]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    num_cols = [c for c in ["Age", "Weight"] if c in X.columns]
    if num_cols:
        transformers.append(("num", StandardScaler(), num_cols))

    preproc = ColumnTransformer(transformers, remainder="drop")
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    pipeline_local = Pipeline([("pre", preproc), ("clf", clf)])

    # train/test split (if possible)
    try:
        X_train, X_test, y_train, y_test = train_test_split(X.fillna(""), y, test_size=0.18, random_state=42, stratify=y if y.nunique()>1 else None)
        pipeline_local.fit(X_train, y_train)
        y_pred = pipeline_local.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        model_accuracy = acc
        app.logger.info(f"Trained model accuracy: {acc:.4f}")
    except Exception as e:
        # fallback: train on all data
        app.logger.exception("Train/test split failed; training on all data as fallback")
        pipeline_local.fit(X.fillna(""), y)
        model_accuracy = None

    pipeline = pipeline_local
    expected_input_columns = features  # save the raw feature list as expected columns
    # persist
    joblib.dump(pipeline, PIPE_PATH)
    joblib.dump(expected_input_columns, EXPECTED_COLS_PATH)
    joblib.dump({}, DOCTOR_MAP_PATH)  # placeholder
    app.logger.info("Pipeline and expected columns saved.")

# Train/load at startup
train_or_load()

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Step 1: user provides free-text symptoms and optional age/weight.
    We predict top candidates and return follow-up questions.
    """
    global pipeline, expected_input_columns, model_accuracy
    symptoms_text = request.form.get("symptoms", "").strip()
    age_val = request.form.get("age", "")
    weight_val = request.form.get("weight", "")
    smoker = request.form.get("smoker", "No").strip().title()

    try:
        age = int(age_val) if str(age_val).strip() != "" else None
    except Exception:
        age = None
    try:
        weight = float(weight_val) if str(weight_val).strip() != "" else None
    except Exception:
        weight = None

    # Build raw one-row DataFrame with at least the common fields
    raw = {"Symptoms": symptoms_text, "Smoker": smoker, "Age": age if age is not None else 0, "Weight": weight if weight is not None else 0}
    X = pd.DataFrame([raw])

    if pipeline is None:
        return render_template("error.html", message="Model not available. Place training.csv in project root and restart to train.")

    # Determine expected columns
    if expected_input_columns is None:
        expected_cols = get_expected_columns_from_pipeline(pipeline)
    else:
        expected_cols = expected_input_columns

    X_pre = ensure_input_has_expected_columns(X, expected_cols).fillna("")
    # Now safe to predict
    try:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_pre)[0]
            classes = pipeline.classes_
            idx = np.argsort(probs)[-3:][::-1]
            top_diseases = classes[idx].tolist()
            top_probs = probs[idx].tolist()
        else:
            preds = pipeline.predict(X_pre)
            top_diseases = preds.tolist()
            top_probs = [1.0] * len(top_diseases)
    except Exception as e:
        app.logger.exception("Prediction error")
        return render_template("error.html", message="Prediction failed; check server logs."), 500

    primary = top_diseases[0] if len(top_diseases)>0 else None
    questions = followup_questions.get(primary, followup_questions.get("default"))
    candidates = [{"disease": d, "prob": f"{p:.1%}"} for d, p in zip(top_diseases, top_probs)]
    return render_template("ask.html", symptoms=symptoms_text, age=age or "", weight=weight or "", smoker=smoker,
                           candidates=candidates, questions=questions, primary=primary, accuracy=(f"{model_accuracy:.2%}" if model_accuracy else "N/A"))

@app.route("/finalize", methods=["POST"])
def finalize():
    """
    Step 2: receive follow-up answers and compute final adjusted probabilities using fuzzy adjustment.
    """
    global pipeline, expected_input_columns, model_accuracy, doctor_map
    symptoms_text = request.form.get("symptoms", "")
    age_val = request.form.get("age", "")
    weight_val = request.form.get("weight", "")
    smoker = request.form.get("smoker", "No").strip().title()

    # follow-up values: support pain_severity, onset_hours, menstrual_now etc.
    try:
        pain_severity = float(request.form.get("pain_severity", 0))
    except:
        pain_severity = 0.0
    try:
        onset_hours = float(request.form.get("onset_hours", 24))
    except:
        onset_hours = 24.0
    menstrual_now = request.form.get("menstrual_now", "No").strip().lower()

    try:
        age = int(age_val) if str(age_val).strip() != "" else 30
    except:
        age = 30

    # Build input row as before
    raw = {"Symptoms": symptoms_text, "Smoker": smoker, "Age": age, "Weight": (float(weight_val) if weight_val else 0)}
    X = pd.DataFrame([raw])

    if pipeline is None:
        return render_template("error.html", message="Model not available.")

    expected_cols = expected_input_columns if expected_input_columns is not None else get_expected_columns_from_pipeline(pipeline)
    X_pre = ensure_input_has_expected_columns(X, expected_cols).fillna("")

    try:
        if hasattr(pipeline, "predict_proba"):
            probs = pipeline.predict_proba(X_pre)[0]
            classes = pipeline.classes_
        else:
            classes = pipeline.predict(X_pre)
            probs = np.ones(len(classes))
    except Exception:
        app.logger.exception("Prediction during finalize failed")
        return render_template("error.html", message="Prediction failed during finalize."), 500

    # Compute fuzzy multiplier
    adj = evaluate_fuzzy_adjustment(pain=float(pain_severity), onset=float(onset_hours), age=float(age))
    app.logger.info(f"Fuzzy adjustment factor: {adj:.3f}")

    # Convert arrays
    probs = np.array(probs, dtype=float)

    # Apply a domain-specific rule: if menstruating -> boost period cramps
    if menstrual_now and (menstrual_now.startswith("y") or menstrual_now == "yes"):
        for i, d in enumerate(classes):
            if "period" in str(d).lower() or "cramp" in str(d).lower():
                probs[i] *= max(adj, 1.0)  # boost by at least 1.0

    # apply global adj multiplier
    probs = probs * adj
    if probs.sum() == 0:
        probs = np.ones_like(probs)
    probs = probs / probs.sum()

    # top 3 after adjustment
    top_idx = np.argsort(probs)[-3:][::-1]
    top_diseases = classes[top_idx].tolist()
    top_probs = probs[top_idx].tolist()

    results = []
    for d, p in zip(top_diseases, top_probs):
        doc = doctor_map.get(d, "General Physician")
        remedies = home_remedy_dict.get(d, [])
        results.append({"disease": d, "probability": f"{p:.1%}", "doctor": doc, "remedies": remedies})

    # maps query - decide specialty
    specialty = "General Physician"
    # heuristics:
    if any("period" in str(r['disease']).lower() or "cramp" in str(r['disease']).lower() for r in results):
        specialty = "Gynecologist"
    elif any("append" in str(r['disease']).lower() for r in results):
        specialty = "Surgeon"
    else:
        specialty = "General Physician"

    maps_query = f"https://www.google.com/maps/search/?api=1&query={specialty.replace(' ', '+')}+near+me"

    return render_template("result.html", results=results, accuracy=(f"{model_accuracy:.2%}" if model_accuracy else "N/A"),
                           maps_query=maps_query, specialty=specialty)

@app.route("/health")
def health():
    return {"status": "ok", "model_loaded": pipeline is not None}

# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
