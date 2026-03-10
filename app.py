import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gender_guesser.detector as gender_detector
import pandas as pd

# Load trained model 
model = tf.keras.models.load_model('scholarship_text_classifier.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the dataset 
scholarship_data = pd.read_csv('tn_india_scholarships_2025.csv')

# Initialize gender detector
detector = gender_detector.Detector()

def predict_gender(name):
    first_name = name.strip().split(' ')[0]
    gender = detector.get_gender(first_name)
    if gender in ['male', 'mostly_male']:
        return 'Male'
    elif gender in ['female', 'mostly_female']:
        return 'Female'
    else:
        return 'Unknown'

def predict_scholarship(user_text):
    seq = tokenizer.texts_to_sequences([user_text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    pred = model.predict(padded)
    pred_label = pred.argmax(axis=1)[0]
    scholarship_name = label_encoder.inverse_transform([pred_label])[0]

    # Get official website link
    scholarship_row = scholarship_data[scholarship_data['name'] == scholarship_name].iloc[0]
    official_website = scholarship_row['official_website']

    return scholarship_name, official_website

# Streamlit App Layout
st.title("🎓 Scholarship Prediction App")

user_name = st.text_input("Enter your full name:")
user_description = st.text_area(
    "Describe your education, income, caste, and other details:")

if st.button("Predict"):
    if user_name and user_description:
        # Predict gender
        predicted_gender = predict_gender(user_name)
        st.success(f"✅ Predicted Gender: {predicted_gender}")

        # Predict scholarship
        predicted_scholarship, scholarship_link = predict_scholarship(user_description)
        st.success(f"🎯 Predicted Scholarship: {predicted_scholarship}")
        st.write(f"[🔗 Official Scholarship Link]({scholarship_link})")
    else:
        st.error("⚠️ Please provide both name and description to proceed.")
