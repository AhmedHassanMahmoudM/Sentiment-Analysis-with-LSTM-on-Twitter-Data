# app.py

import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import re
import nltk
import contractions
import emoji
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Download required nltk data
nltk.download('stopwords')

# Load the pre-trained model
model = tf.keras.models.load_model('sentiment_analysis_model.h5')  # Replace with your model path

# Load tokenizer (use the tokenizer that was used during training)
# Assuming you saved it as a pickle file
import pickle
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Preprocessing function
def preprocess_text(text):
    # Expand contractions
    text = contractions.fix(text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#','', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    
    # Convert emojis to text
    text = emoji.demojize(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    text = ' '.join([word for word in text.split() if word not in stop_words])
    
    return text

# Function to predict sentiment
def predict_sentiment(text):
    # Preprocess the text
    processed_text = preprocess_text(text)
    
    # Convert text to sequences using the tokenizer
    sequences = tokenizer.texts_to_sequences([processed_text])
    
    # Pad the sequences to match the input length used during training
    padded_sequences = pad_sequences(sequences, maxlen=100)  # Use the maxlen from training
    
    # Make predictions using the loaded model
    prediction = model.predict(padded_sequences)
    
    # Convert the output to a binary sentiment
    sentiment = 'Positive' if prediction >= 0.5 else 'Negative'
    
    return sentiment

# Streamlit app
st.title("Sentiment Analysis on Twitter Data")
st.write("This app predicts the sentiment (Positive/Negative) of a tweet using an LSTM model.")

# User input
user_input = st.text_area("Enter a tweet:", "")

# Predict button
if st.button("Predict"):
    if user_input:
        # Make prediction
        sentiment = predict_sentiment(user_input)
        
        # Display the result
        st.write(f"Sentiment: **{sentiment}**")
        
        # Optional: Display Word Cloud of the input text
        if sentiment == "Positive":
            st.write("Word Cloud for positive tweet:")
        else:
            st.write("Word Cloud for negative tweet:")
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(preprocess_text(user_input))
        
        # Display the word cloud using matplotlib
        plt.figure(figsize=(10,5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    else:
        st.write("Please enter a tweet for prediction.")

