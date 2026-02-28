# 🧠 AI Context Analyst: Multi-Class Text Classification
### *A Production-Ready Hybrid NLP Pipeline & Live Deployment*

[![Live Demo](https://img.shields.io/badge/demo-HuggingFace-yellow.svg)](YOUR_HUGGINGFACE_SPACE_URL_HERE)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web%20App-lightgrey.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

This repository contains a comprehensive, end-to-end NLP system designed to categorize unstructured text into 9+ semantic classes (Business, Science, Sports, etc.). Developed as a senior laboratory project for **CSE440: Natural Language Processing II**, this system bridges the gap between raw data and actionable intelligence using a combination of classical Machine Learning and state-of-the-art Recurrent Neural Networks.

---

## 🚀 Key Features

- **Hybrid Inference Engine**: An intelligent backend that prefers **Deep Learning (LSTM)** for high-confidence predictions but fallbacks to **Logistic Regression (TF-IDF)** for efficiency.
- **Advanced Preprocessing Stack**: Implements a modular, vectorized pipeline featuring **POS-aware lemmatization**, tokenization, and noise reduction.
- **Diverse Word Representations**: Comparison study across **BoW, TF-IDF, GloVe embeddings,** and locally trained **Word2Vec Skip-gram** models.
- **Stratified Data Balancing**: Engineered to handle class imbalance in a 1.45M+ row dataset through sophisticated sampling techniques.
- **Premium Web UI**: A modern, responsive interface built with **Glassmorphism** and instant narrative analysis capabilities.
- **Cloud-Ready Deployment**: Fully Dockerized and live on Hugging Face Spaces.

---

## 🔬 Experimental Results

We conducted **22+ distinct experiments** to identify the most robust architecture.

| Model Architecture | Word Representation | Accuracy | Macro F1-Score |
| :--- | :--- | :--- | :--- |
| **LSTM (Final)** | **Balanced Word2Vec** | **72.4%** | **0.71** |
| Logistic Regression | TF-IDF | 64.2% | 0.62 |
| SimpleRNN | GloVe | 58.5% | 0.54 |
| Naive Bayes | Bag-of-Words | 61.8% | 0.59 |

---

## 🛠️ Tech Stack

- **Core Engine**: Python 3.11
- **NLP Libraries**: NLTK (WordNet, POS Tagging), Gensim
- **ML Frameworks**: Scikit-Learn, TensorFlow/Keras
- **Deployment**: Flask (Inference API), Docker, Hugging Face Spaces

---

## 📂 Project Structure

```bash
├── models/             # Exported "Brains" (h5 models, pkl vectorizers)
├── templates/          # Flask HTML templates (Premium Glassmorphism UI)
├── text_pipeline.py    # The core NLP engine (Preprocessing logic)
├── train_script.py     # High-speed Stratified Training script
├── app.py              # Cloud-ready Flask Inference Server
├── Dockerfile          # Container configuration for deployment
├── requirements.txt    # Project dependencies
└── paper.tex           # Full IEEE Research Paper (LaTeX)
```

---

## 🏁 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. High-Speed Training
To train the balanced models locally:
```bash
python train_script.py
```

### 3. Launch the App
```bash
python app.py
```
Visit `http://localhost:5000` to interact with the local AI analyst.

---
