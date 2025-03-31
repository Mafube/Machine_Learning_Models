import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load models and encoder
models_folder = '/Users/tshmacm1172/Desktop/Machine_Learning_Models/Health_Care/Disease_Prediction_Using_Machine_Learning/models'

svm_model = joblib.load(os.path.join(models_folder, 'svm_model.pkl'))
nb_model = joblib.load(os.path.join(models_folder, 'nb_model.pkl'))
rf_model = joblib.load(os.path.join(models_folder, 'rf_model.pkl'))
encoder = joblib.load(os.path.join(models_folder, 'encoder.pkl'))  # Encoder for label decoding

# Function to combine predictions using majority vote
def combine_predictions(svm_preds, nb_preds, rf_preds, encoder):
    # Compute mode (majority vote)
    final_preds = [stats.mode([i, j, k], keepdims=True).mode.item() for i, j, k in zip(svm_preds, nb_preds, rf_preds)]
    
    # Convert encoded labels back to disease names
    svm_prognosis = encoder.inverse_transform(svm_preds)
    nb_prognosis = encoder.inverse_transform(nb_preds)
    rf_prognosis = encoder.inverse_transform(rf_preds)
    final_prognosis = encoder.inverse_transform(final_preds)

    # Return DataFrame with readable disease names
    return pd.DataFrame({
        "SVM Prediction": svm_prognosis,
        "Naive Bayes Prediction": nb_prognosis,
        "Random Forest Prediction": rf_prognosis,
        "Final Prediction": final_prognosis
    })

# Streamlit UI
st.title("Disease Prediction Using Machine Learning")
st.subheader("Upload Your Test Data (CSV)")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file:
    test_data = pd.read_csv(uploaded_file)

    if test_data.isnull().values.any():
        st.error("The uploaded data contains missing values. Please clean the data before uploading.")
    else:
        # Extract features and true labels
        test_X = test_data.iloc[:, :-1]
        true_labels = encoder.transform(test_data.iloc[:, -1])  # Convert true prognosis to encoded form

        # Model predictions
        svm_preds = svm_model.predict(test_X)
        nb_preds = nb_model.predict(test_X)
        rf_preds = rf_model.predict(test_X)

        # Get final predictions with readable disease names
        final_preds_df = combine_predictions(svm_preds, nb_preds, rf_preds, encoder)

        # Display predictions in a DataFrame
        st.subheader("Predictions")
        st.dataframe(final_preds_df)

        # Compute accuracy
        final_preds_labels = encoder.transform(final_preds_df["Final Prediction"])
        accuracy = accuracy_score(true_labels, final_preds_labels)
        st.write(f"**Accuracy:** {accuracy * 100:.2f}%")

      