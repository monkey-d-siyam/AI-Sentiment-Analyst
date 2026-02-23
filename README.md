# 🧠 AI-Powered Multi-Class Text Classifier
### *Production-Ready NLP Pipeline & Deployment*

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A sophisticated, end-to-end Natural Language Processing (NLP) system designed to categorize unstructured text into meaningful classes. This project bridges the gap between raw data and actionable intelligence using a combination of classical Machine Learning and state-of-the-art Deep Learning.

---

## 🚀 Key Features

- **Multi-Model Support**: Compare results between **Logistic Regression (TF-IDF)** and **Deep Learning (LSTM)** architectures.
- **Advanced Preprocessing**: Implements POS-aware lemmatization, tokenization, and stopword removal via a modular vectorized pipeline.
- **Smart Embeddings**: Supports **Word2Vec Skip-gram** and **GloVe** representations for deep contextual understanding.
- **Premium UI**: A modern, responsive web interface built with **Glassmorphism** and high-end typography for instant narrative analysis.
- **Scalable Architecture**: Designed to handle datasets with millions of rows while maintaining high performance.

---

## 🛠️ Tech Stack

- **Core**: Python 3.8+
- **NLP**: NLTK, Gensim
- **Machine Learning**: Scikit-Learn (TF-IDF, Logistic Regression, Random Forest)
- **Deep Learning**: TensorFlow/Keras (LSTM, GRU, Word2Vec)
- **Deployment**: Flask (Backend), Modern Vanilla CSS (Premium Frontend)

---

## 📂 Project Structure

```bash
├── Datasets/           # Training & Test CSV files (Standard: 'QA Text' & 'Class')
├── models/             # Exported model "brains" (pkl, h5, tokenizer)
├── templates/          # HTML templates for the Flask web application
├── text_pipeline.py    # The core NLP engine (preprocessing & utilities)
├── train_script.py     # High-speed training script with Stratified Sampling
├── train.ipynb         # Interactive research & experimentation notebook
├── app.py              # Flask deployment server
└── requirements.txt    # Project dependencies
```

---

## 🏁 Quick Start

### 1. Installation
Clone the repository and install the necessary libraries:
```bash
pip install -r requirements.txt
```

### 2. Training the "Brain"
You can train the models either interactively via the notebook or quickly via the terminal:
```bash
# High-speed stratified training
python train_script.py
```

### 3. Launching the Dashboard
Once the training is complete and artifacts are in the `models/` folder:
```bash
python app.py
```
Visit `http://127.0.0.1:5000` to start analyzing text!

---

## 🧪 The "Chaotic Text" Test
The system is built to handle noise. Try pasting a "chaotic" sample like this to see how the model ignores noise and finds the core intent:

> *"MARKET ALERT! 📉 Stocks are tumbling as investors react... Should I diversify into gold?? #WallStreet #Investing"*

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

---
*Built with ❤️ for High-Performance NLP.*
