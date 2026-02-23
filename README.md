# Multi-class Text Classification Project

This repository contains a training notebook and a small Flask app for deploying a multi-class text classifier.

Files created:

- `train.ipynb` - Training notebook (data merge, preprocessing, ML & NN training, evaluation)
- `text_pipeline.py` - Reusable preprocessing, tokenizer, embedding helpers
- `app.py` - Flask app for loading saved models and predicting classes
- `requirements.txt` - Python dependencies

Quick start:

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the training notebook (`train.ipynb`) in Jupyter to produce artifacts in `models/`.

3. Start the Flask app:

```bash
python app.py
```

Then open http://localhost:5000 and input text to classify.

Notes:
- `text_pipeline.preprocess_text` is the canonical preprocessing used by both training and Flask app.
- The notebook saves vectorizers, tokenizer, embedding matrices, and example models under `models/`.
