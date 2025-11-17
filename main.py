'''
AI Disease Symptom Checker 🩺

A command-line based AI consultant that predicts potential medical conditions based on 
user-provided symptoms, age, and smoking status.

Description

This project uses a `RandomForestClassifier` machine learning model to analyze user input 
and suggest the top three most likely medical conditions. For each potential condition, 
it provides a calculated probability and recommends a relevant medical specialist to consult.

The model is trained on a dataset linking symptoms and demographic data to specific diseases. 
It employs a `scikit-learn` pipeline that processes textual symptom data with `TfidfVectorizer` 
and categorical data with `OneHotEncoder`.
'''

# --- Imports ---
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# --- 1. The Dataset ---
# Load the dataset
csv_data = "train.csv"
df = pd.read_csv(csv_data)

# Create a mapping from disease to doctor for later use
doctor_map = df.drop_duplicates(subset=['Disease/Condition']).set_index('Disease/Condition')['Doctor to Consult'].to_dict()


# --- 2. Define Features (X) and Target (y) ---
# Predict the 'Disease/Condition' based on other columns.
# Simplify and use the most important features for this model.
features = ['Symptoms', 'Age Category', 'Smoker']
target = 'Disease/Condition'

X = df[features]
y = df[target]


# --- 3. Preprocessing and Model Pipeline ---
# Process text and categorical data differently.

# Define the preprocessing steps
# 'TfidfVectorizer' for the 'Symptoms' text data
# 'OneHotEncoder' for the categorical data ('Age Category', 'Smoker')
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(), 'Symptoms'),
        ('category', OneHotEncoder(handle_unknown='ignore'), ['Age Category', 'Smoker'])
    ])

# Create the full pipeline with the preprocessor and the classifier
# RandomForestClassifier is a good and robust choice for this task.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])


# --- 4. Train the Model ---
# Split the data to train the model and test it on unseen data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
pipeline.fit(X_train, y_train)

print("✅ Model training complete.")
accuracy = pipeline.score(X_test, y_test)
print(f"📊 Model Accuracy on Test Data: {accuracy:.2%}\n")


# --- 5. The Interactive Consultant Function ---
def get_consultation():
    """
    Takes user input for symptoms and provides a potential diagnosis.
    """
    print("🩺 Welcome to the AI Disease Consultant!")
    print("Please describe your symptoms (e.g., 'i have a headache and tummy hurts').")
    symptoms = input("Your symptoms: ")

    # The AI asks follow-up questions
    print("\nPlease answer a few follow-up questions:")
    age_category = input("Are you a 'Child', 'Teenager', or 'Adult'? ").strip().title()
    is_smoker = input("Do you smoke ('Yes' or 'No')? ").strip().title()

    # Create a DataFrame from the user's input
    user_data = pd.DataFrame({
        'Symptoms': [symptoms],
        'Age Category': [age_category],
        'Smoker': [is_smoker]
    })
    
    # Get prediction probabilities for the top 3 conditions
    probabilities = pipeline.predict_proba(user_data)[0]
    top_3_indices = np.argsort(probabilities)[-3:][::-1]
    top_3_diseases = pipeline.classes_[top_3_indices]
    top_3_probabilities = probabilities[top_3_indices]

    print("\n--- AI Analysis ---")
    print("Based on your symptoms, here are the most likely possibilities:")
    
    for i, (disease, prob) in enumerate(zip(top_3_diseases, top_3_probabilities)):
        doctor = doctor_map.get(disease, 'General Physician')
        print(f"{i+1}. {disease} ({prob:.1%} chance)")
        print(f"   - Recommended Doctor: {doctor}\n")
        
    print("="*30)
    print("⚠️ IMPORTANT: This is an AI-powered suggestion and not a medical diagnosis.")
    print("Please consult with a real medical professional for accurate advice.")
    print("="*30)


# --- Run the consultant ---
if __name__ == "__main__":
    get_consultation()