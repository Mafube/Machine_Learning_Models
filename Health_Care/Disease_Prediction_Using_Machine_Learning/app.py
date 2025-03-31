import streamlit as st
import joblib
import pandas as pd
import numpy as np
from scipy import stats


svm_model = joblib.load('./models/svm_model.pkl')
nb_model = joblib.load('./models/nb_model.pkl')
rf_model = joblib.load('./models/rf_model.pkl')
encoder = joblib.load('./models/encoder.pkl')
combine_predictions = joblib.load('./models/combine_predictions.pkl')


st.title("Model Prediction App")


st.subheader("Upload Test Data")
uploaded_file = st.file_uploader("Choose a CSV file", type='csv')

if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)
    
    if test_data.isnull().values.any():
        st.error("The uploaded data contains missing values. Please clean the data before uploading.")
    else:
      
        test_X = test_data.iloc[:, :-1]
        
        
        svm_preds = svm_model.predict(test_X)
        nb_preds = nb_model.predict(test_X)
        rf_preds = rf_model.predict(test_X)


        final_preds = combine_predictions(svm_preds, nb_preds, rf_preds)
        
       
        st.subheader("Predictions")
        st.write(final_preds)
        
        
        if test_data.shape[1] > 1:
            true_labels = encoder.transform(test_data.iloc[:, -1])
            accuracy = np.mean(final_preds == true_labels)
            st.write(f"Accuracy: {accuracy * 100:.2f}%")
        
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        cf_matrix = confusion_matrix(true_labels, final_preds)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cf_matrix, annot=True, fmt='g', cmap="Blues")
        plt.title("Confusion Matrix")
        st.pyplot()
