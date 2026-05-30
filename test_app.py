import pickle
import tensorflow as tf

# Load original model in the original working environment
model = tf.keras.models.load_model("scholarship_text_classifier.h5")

# Save model again in TensorFlow SavedModel format
model.save("new_model")

# Reload tokenizer and save again
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("tokenizer_new.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Reload label encoder and save again
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("label_encoder_new.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

print("Conversion complete")