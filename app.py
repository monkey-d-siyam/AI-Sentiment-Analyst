
import os
import numpy as np
from flask import Flask, request, render_template, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from text_pipeline import preprocess_text, load_artifact

# Flexible pathing for cloud deployment
base_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Handle Templates: Use 'templates/' if it exists, else use root '.'
template_dir = os.path.join(base_dir, 'templates')
if not os.path.exists(template_dir):
    template_dir = base_dir

app = Flask(__name__, template_folder=template_dir)

# 2. Handle Models: Use 'models/' if it exists, else use root '.'
MODELS_DIR = os.path.join(base_dir, 'models')
if not os.path.exists(MODELS_DIR):
    MODELS_DIR = base_dir

BEST_ML_PATH = os.path.join(MODELS_DIR, 'best_ml_model.pkl')
BEST_NN_PATH = os.path.join(MODELS_DIR, 'best_nn_model.h5')
TOKENIZER_PATH = os.path.join(MODELS_DIR, 'tokenizer.pkl')
TFIDF_PATH = os.path.join(MODELS_DIR, 'tfidf_vect.pkl')
LE_PATH = os.path.join(MODELS_DIR, 'label_encoder.pkl')

# Global variables for models and utilities
model_nn = None
model_ml = None
tokenizer = None
tfidf = None
label_encoder = None

def load_resources():
    global model_nn, model_ml, tokenizer, tfidf, label_encoder
    
    # Load Label Encoder
    if os.path.exists(LE_PATH):
        label_encoder = load_artifact(LE_PATH)
        print("Label Encoder loaded.")
    
    # Load Tokenizer & NN Model
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = load_artifact(TOKENIZER_PATH)
        print("Tokenizer loaded.")
    
    if os.path.exists(BEST_NN_PATH):
        try:
            model_nn = load_model(BEST_NN_PATH)
            print("NN Model loaded successfully.")
        except Exception as e:
            print(f"Error loading NN model: {e}")

    # Load TF-IDF & ML Model
    if os.path.exists(TFIDF_PATH):
        tfidf = load_artifact(TFIDF_PATH)
        print("TF-IDF Vectorizer loaded.")
        
    if os.path.exists(BEST_ML_PATH):
        model_ml = load_artifact(BEST_ML_PATH)
        print("ML Model loaded successfully.")

# Initialize resources
load_resources()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() if request.is_json else request.form
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    # Prediction logic (Prefer NN, fallback to ML)
    prediction = "N/A"
    confidence = 0.0
    method = "None"

    try:
        if model_nn and tokenizer and label_encoder:
            seq = tokenizer.texts_to_sequences([cleaned_text])
            # FIXED: maxlen must match the training script (50)
            padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
            probs = model_nn.predict(padded)[0]
            idx = np.argmax(probs)
            prediction = label_encoder.inverse_transform([idx])[0]
            confidence = float(probs[idx])
            method = "Neural Network"
        elif model_ml and tfidf:
            features = tfidf.transform([cleaned_text])
            prediction = model_ml.predict(features)[0]
            method = "Machine Learning (TF-IDF)"
            try:
                probs = model_ml.predict_proba(features)[0]
                confidence = float(np.max(probs))
            except:
                confidence = 1.0
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'original_text': text,
        'cleaned_text': cleaned_text,
        'prediction': prediction,
        'confidence': f"{confidence:.2%}",
        'method': method
    })

if __name__ == '__main__':
    # Ensure models dir exists for first run
    os.makedirs(MODELS_DIR, exist_ok=True)
    # Use 7860 for Hugging Face or 5000 as fallback
    port = int(os.environ.get("PORT", 7860))
    app.run(host='0.0.0.0', port=port)
