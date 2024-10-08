import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import load_model

# Load the text file containing book titles to fit the tokenizer
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().lower().splitlines()
    return data

# Load the tokenizer (you may need to re-instantiate it)
def load_tokenizer(file_path):
    titles = load_data(file_path)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(titles)
    return tokenizer

# Sampling function to introduce temperature into word generation
def sample_with_temperature(predictions, temperature=1.0):
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions + 1e-10) / temperature
    exp_preds = np.exp(predictions)
    predictions = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, predictions, 1)
    return np.argmax(probas)

# Generate a new book title based on input text with temperature sampling
def generate_title(seed_text, model, tokenizer, max_len, temperature=1.0):
    for _ in range(10):  # Generate up to 10 words
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_len - 1, padding='pre')
        predicted = model.predict(token_list, verbose=0)[0]
        
        predicted_word_index = sample_with_temperature(predicted, temperature)
        
        word = ''
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                break
        
        if word == '' or len(seed_text.split()) > 5:  # Limit the length to avoid verbosity
            break
        
        seed_text += ' ' + word
    
    return seed_text

# Constants
file_path = 'book_titles.txt'  # Path to the text file containing book titles
model_file_path = 'book_title_generator_model.h5'  # Path to the saved model
max_len = 20  # This should match the max_len used during training

# Load the trained model
model = load_model(model_file_path)

# Load the tokenizer
tokenizer = load_tokenizer(file_path)

# User input
user_input = input("Enter a phrase to generate a book title: ")

# Adjust the temperature value for randomness: Try values between 0.7 (less random) and 1.5 (more random)
temperature = 0.8

generated_title = generate_title(user_input, model, tokenizer, max_len, temperature)
print("Generated title:", generated_title)