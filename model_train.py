import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = 'tn_india_scholarships_2025.csv'
scholarship_data = pd.read_csv(file_path)

# Preprocess income column
scholarship_data['income_limit_inr'] = pd.to_numeric(
    scholarship_data['income_limit_inr'], errors='coerce')
scholarship_data = scholarship_data.dropna(subset=['income_limit_inr']).reset_index(drop=True)

# Prepare dataset for training
scholarship_data['combined_features'] = (
    scholarship_data['education_level_min'] + ' ' +
    scholarship_data['caste_eligibility'] + ' ' +
    scholarship_data['gender_eligibility'] + ' ' +
    scholarship_data['benefits_summary']
)

# Encode target labels (scholarship name)
label_encoder = LabelEncoder()
scholarship_data['target'] = label_encoder.fit_transform(scholarship_data['name'])

# Tokenize text data
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(scholarship_data['combined_features'])
X = tokenizer.texts_to_sequences(scholarship_data['combined_features'])
X = pad_sequences(X, maxlen=100, padding='post')

y = scholarship_data['target'].values

# Build the text classification model
model = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=100),
    GlobalAveragePooling1D(),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')
])

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=2)

# Save the model, tokenizer, and label encoder
model.save('scholarship_text_classifier.h5')
import pickle
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

print("Model, tokenizer, and label encoder saved successfully.")
