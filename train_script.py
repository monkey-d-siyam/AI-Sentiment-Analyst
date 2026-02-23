
import os
import glob
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

from text_pipeline import (
    load_and_merge_data, preprocess_dataframe, build_tokenizer, 
    save_artifact
)

# Paths
TRAIN_GLOB = 'Datasets/*[Training]*.csv'
TEST_PATH = 'Datasets/[Updated] Question Answer Classification Dataset[Test] (1).csv'
MODELS_DIR = 'models'

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 1. Load Data
    print("--- Phase 1: Loading Data ---")
    train_df, test_df = load_and_merge_data(TRAIN_GLOB, TEST_PATH)
    
    print(f"Found columns: {train_df.columns.tolist()}")
    
    # Ensure columns exist
    if 'label' not in train_df.columns:
        print("FORCING RENAME")
        train_df.columns = ['text', 'label'] # Fallback
            
    # IMPROVED: Stratified sampling
    if len(train_df) > 50000:
        print("Sampling 50,000 rows...")
        # Use a more robust sampling
        unique_labels = train_df['label'].unique()
        samples = []
        per_class = 50000 // len(unique_labels)
        for label in unique_labels:
            label_df = train_df[train_df['label'] == label]
            samples.append(label_df.sample(min(len(label_df), per_class), random_state=42))
        train_df = pd.concat(samples).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Final Train size: {len(train_df)}")

    # 2. Preprocess
    print("\n--- Phase 2: Preprocessing ---")
    train_df = preprocess_dataframe(train_df)

    # 3. Label Encoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(train_df['label'])
    save_artifact(le, os.path.join(MODELS_DIR, 'label_encoder.pkl'))
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_.tolist()}")

    # 4. ML Features & Training
    print("\n--- Phase 3: Training ML Model ---")
    tfidf_vect = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf_vect.fit_transform(train_df['clean_text'])
    
    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(X_train_tfidf, y_train_enc)
    
    save_artifact(tfidf_vect, os.path.join(MODELS_DIR, 'tfidf_vect.pkl'))
    save_artifact(lr, os.path.join(MODELS_DIR, 'best_ml_model.pkl'))

    # 5. NN Training
    print("\n--- Phase 4: Training Neural Network ---")
    tokenizer, X_train_seq = build_tokenizer(train_df['clean_text'], num_words=10000, maxlen=50)
    save_artifact(tokenizer, os.path.join(MODELS_DIR, 'tokenizer.pkl'))
    
    vocab_size = len(tokenizer.word_index) + 1
    model = Sequential([
        Embedding(vocab_size, 64, input_length=50),
        LSTM(32),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_seq, y_train_enc, epochs=3, batch_size=64, verbose=1)
    
    model.save(os.path.join(MODELS_DIR, 'best_nn_model.h5'))
    print("\n🎉 Training Complete!")

if __name__ == "__main__":
    main()
