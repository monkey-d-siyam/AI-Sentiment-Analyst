
import os
import re
import glob
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet

# Download necessary NLTK resources
REQUIRED_NLTK_RESOURCES = ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger', 'punkt_tab']
for res in REQUIRED_NLTK_RESOURCES:
    try:
        nltk.data.find(res)
    except Exception:
        nltk.download(res, quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEM = WordNetLemmatizer()

def _get_wordnet_pos(treebank_tag):
    """Map POS tag to wordnet tags."""
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def preprocess_text(text, lemmatize=True):
    """
    Modular text preprocessing function.
    Steps: Lowercase, remove non-alphabetic chars, tokenize, remove stopwords, lemmatize.
    """
    if not isinstance(text, str):
        return ""
    
    # Lowercase and remove punctuation/numbers using regex
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and short tokens
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    
    if lemmatize:
        # Lemmatize with POS tagging for better accuracy
        try:
            tagged = pos_tag(tokens)
            tokens = [LEM.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in tagged]
        except Exception:
            # Fallback to default lemmatization
            tokens = [LEM.lemmatize(word) for word in tokens]
            
    return " ".join(tokens)

def preprocess_dataframe(df, text_col='text', out_col='clean_text', lemmatize=True):
    """
    Apply preprocessing to a dataframe using vectorized string operations 
    for initial cleaning and efficient apply for tokenization/lemmatization.
    """
    # Vectorized initial cleaning
    df[out_col] = df[text_col].astype(str).str.lower()
    df[out_col] = df[out_col].str.replace(r'[^a-z\s]', ' ', regex=True)
    df[out_col] = df[out_col].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # Use regular apply instead of progress_apply to avoid tqdm issues in background
    df[out_col] = df[out_col].apply(lambda x: preprocess_text(x, lemmatize=lemmatize))
    
    return df

def load_and_merge_data(train_glob, test_path):
    """
    Load 5 training CSVs and 1 test CSV.
    Handle duplicates and missing values.
    """
    train_files = glob.glob(train_glob)
    train_dfs = []
    
    for f in train_files:
        print(f"Loading {f}...")
        df = pd.read_csv(f)
        
        # Aggressive renaming
        new_cols = {}
        for col in df.columns:
            c_low = col.lower().strip()
            if 'text' in c_low or 'qa' in c_low:
                new_cols[col] = 'text'
            elif 'class' in c_low or 'label' in c_low:
                new_cols[col] = 'label'
        
        df = df.rename(columns=new_cols)
        train_dfs.append(df)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    
    # Same for test
    test_df = pd.read_csv(test_path)
    test_new_cols = {}
    for col in test_df.columns:
        c_low = col.lower().strip()
        if 'text' in c_low or 'qa' in c_low:
            test_new_cols[col] = 'text'
        elif 'class' in c_low or 'label' in c_low:
            test_new_cols[col] = 'label'
    test_df = test_df.rename(columns=test_new_cols)
    
    # Cleanup
    for df in [train_df, test_df]:
        if 'text' not in df.columns or 'label' not in df.columns:
            print(f"Warning: Missing required columns. Available: {df.columns.tolist()}")
            continue
            
        df.dropna(subset=['text', 'label'], inplace=True)
        df.drop_duplicates(subset=['text'] if 'text' in df.columns else None, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    return train_df, test_df

# Neural Network Utilities
def build_tokenizer(texts, num_words=20000, maxlen=100):
    """Fit Keras tokenizer and return sequences."""
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')
    
    return tokenizer, padded

def get_glove_embeddings(word_index, embedding_dim=100, glove_path=None):
    """Load GloVe vectors and creates an embedding matrix."""
    if glove_path is None or not os.path.exists(glove_path):
        print("GloVe path not provided or does not exist. Skipping.")
        return None

    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
            
    return embedding_matrix

def train_word2vec_skipgram(texts, embedding_dim=100):
    """Train Word2Vec Skip-gram model using Gensim."""
    from gensim.models import Word2Vec
    tokenized_texts = [text.split() for text in texts]
    
    print("Training Word2Vec Skip-gram model...")
    model = Word2Vec(sentences=tokenized_texts, 
                     vector_size=embedding_dim, 
                     window=5, 
                     min_count=2, 
                     workers=4, 
                     sg=1) # sg=1 for Skip-gram
    return model

def build_w2v_matrix(word_index, w2v_model, embedding_dim=100):
    """Create embedding matrix from trained Word2Vec model."""
    vocab_size = len(word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in word_index.items():
        if word in w2v_model.wv:
            embedding_matrix[i] = w2v_model.wv[word]
    return embedding_matrix

def save_artifact(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)

def load_artifact(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)