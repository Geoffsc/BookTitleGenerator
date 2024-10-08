import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Load the text file
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = file.read().lower().splitlines()
    return data

# Preprocess the data
def preprocess_data(titles, tokenizer, max_len):
    sequences = []
    
    for title in titles:
        token_list = tokenizer.texts_to_sequences([title])[0]
        for i in range(1, len(token_list)):
            sequence = token_list[:i+1]
            sequences.append(sequence)
    
    sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
    return sequences

# Define constants
file_path = 'book_titles.txt'  # Path to the text file containing book titles
max_len = 20  # Maximum length of the sequence

# Load and prepare data
titles = load_data(file_path)

# Tokenizer to convert text into sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(titles)
total_words = len(tokenizer.word_index) + 1  # +1 for padding

# Prepare sequences for training
sequences = preprocess_data(titles, tokenizer, max_len)

# Input and output sequences
X = sequences[:, :-1]  # Features
y = tf.keras.utils.to_categorical(sequences[:, -1], num_classes=total_words)  # Labels

# Build the model
def create_model(total_words, max_len):
    model = Sequential()
    model.add(Embedding(input_dim=total_words, output_dim=128, input_length=max_len - 1))
    model.add(LSTM(100))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model(total_words, max_len)
model.summary()  # Optional: Print model summary

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Save the model
model.save('book_title_generator_model.h5')
print("Model saved as 'book_title_generator_model.h5'")