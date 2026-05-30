import os
import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import gender_guesser.detector as gender_detector
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
# Enable CORS for all routes (important so Vercel can communicate with this API)
CORS(app)

# Load trained model
print("Loading TensorFlow Keras model...")
model = tf.keras.models.load_model('scholarship_text_classifier.h5')

print("Loading tokenizer and label encoder...")
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Load the dataset
print("Loading scholarship dataset...")
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

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "scholarship-prediction-api"}), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Missing JSON request body"}), 400

        name = data.get('name', '').strip()
        description = data.get('description', '').strip()

        if not name or not description:
            return jsonify({"error": "Both 'name' and 'description' are required"}), 400

        # Predict gender and scholarship
        predicted_gender = predict_gender(name)
        predicted_scholarship, scholarship_link = predict_scholarship(description)

        return jsonify({
            "success": True,
            "predicted_gender": predicted_gender,
            "predicted_scholarship": predicted_scholarship,
            "official_website": scholarship_link
        }), 200

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable (Render sets this dynamically)
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
