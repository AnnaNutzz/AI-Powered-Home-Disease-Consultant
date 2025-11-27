# AI-Powered Home Disease Consultant (ML + Fuzzy Logic)

## Overview

This project is an **AI-powered medical triage assistant** combining:

* **Machine Learning** (Random Forest + TF-IDF)
* **Soft Computing / Fuzzy Logic** (custom, pure Python)
* **Interactive multi-step diagnosis flow**
* **Google Maps doctor search integration**
* **Home remedy recommendations**

Users enter free-text symptoms, answer follow-up questions, and receive refined diagnoses with probabilities.

---

## Features

### Machine Learning

* Trains automatically using **training.csv**
* Handles:

  * Free-text symptoms via TF-IDF
  * Categorical features (Smoker, Food consumed, Weather, etc.)
  * Numeric features (Age, Weight)
* Random Forest classifier

### Fuzzy Logic (Soft Computing)

* Pure Python implementation
* Membership functions: trapezoidal & triangular
* Inputs: pain severity, onset duration, age
* Rule-based inference adjusts ML probabilities

### Interactive 2-Step Flow

1. User submits free-text symptoms
2. System predicts top diseases
3. System asks disease-specific follow-up questions
4. Fuzzy logic refines final probabilities

### Final Output

* Top 3 diseases with adjusted probabilities
* Recommended doctor type
* Home remedy suggestions
* "Find doctors near me" (Google Maps link)

---

## Project Structure

```
project/
│
├── app.py
├── training.csv
├── requirements.txt
│
├── model/
│   ├── pipeline.pkl
│   ├── expected_cols.pkl
│   └── doctor_map.pkl
│
├── templates/
│   ├── index.html
│   ├── ask.html
│   ├── result.html
│   └── error.html
│
└── static/
    └── style.css
```

---

## Installation & Running

### 1. Create Virtual Environment

```
python -m venv .venv
```

### 2. Activate

**Windows:**

```
.venv\Scripts\activate
```

**Mac/Linux:**

```
source .venv/bin/activate
```

### 3. Install Dependencies

```
pip install -r requirements.txt
```

### 4. Run

```
python app.py
```

Then open: `http://127.0.0.1:5000/`

---

## How It Works

### Training

* Reads `training.csv`
* Builds pipeline: TF-IDF + OneHotEncoder + StandardScaler + RandomForest
* Saves:

  * `pipeline.pkl`
  * `expected_cols.pkl`

### Prediction (Step 1)

* Accepts free-text symptoms
* Normalizes data to match training columns
* Predicts top 3 diseases
* Displays follow-up questions

### Finalization (Step 2)

* Applies fuzzy logic multiplier
* Boosts probabilities (e.g., menstruation → period cramps)
* Normalizes and outputs final ranked diseases

---

## Improving Accuracy

* Add more rows to training.csv
* Clean symptom text labels
* Merge duplicated disease names
* Add features (pain location, cycle info, fever duration)

---

## Fuzzy Logic Details

### Inputs

* Pain severity (0–10)
* Onset duration (hours)
* Age

### Rules (examples)

* If pain is high & onset is recent → urgency increase
* If age is elder & pain is high → higher multiplier
* If menstruating & symptoms match → boost period cramps

### Output

* Adjustment multiplier (0.5–2.0)
* Applied to ML probabilities

---

## Google Maps Integration

Final screen provides:

```
https://www.google.com/maps/search/?api=1&query=<doctor-type>+near+me
```

Autodetects location or asks permission.

---

## Credits

* Built for **AI in Healthcare** and **Soft Computing** coursework
* Machine Learning via scikit-learn
* Fuzzy Logic written manually in Python

---

## License

------------------------------------------------------------------------------
PROJECT: AI-Powered-Home-Disease-Consultant-using-Fuzzy-Logic-and-Machine-Learning
------------------------------------------------------------------------------
AUTHOR: Ahana Kaur
DATE:    November 17, 2025
COURSE:  Soft Computing Lab and AI in Healthcare (End Term Project)
INSTITUTION: Bennett University

COPYRIGHT & LICENSE:
Copyright (c) 2025 by the Authors listed above. All Rights Reserved.

This software was developed exclusively for academic assessment purposes.
Unauthorized commercial use, redistribution, or publication of this code
without the express written consent of the authors is strictly prohibited.
