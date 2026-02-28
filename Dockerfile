# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy everything from local to the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (ensures they are available in the container)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('punkt_tab')"

# Expose the port Flask is running on
EXPOSE 7860

# Run the app
CMD ["python", "app.py"]
