"""
Training module for AI text detection

Author: Keratin
"""
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def load_data():
    """Load dataset from CSV file and split into training and testing sets."""
    df = pd.read_csv('./datasets/labeled_data.csv')
    train_text, test_text, train_labels, test_labels = train_test_split(df['message'], df['class'], test_size=0.2)
    return train_text, test_text, train_labels, test_labels


def tokenize_data(train_text, test_text):
    """Tokenize text data using Tokenizer from Keras and pad sequences to a fixed length."""
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(train_text)
    train_sequences = pad_sequences(tokenizer.texts_to_sequences(train_text), maxlen=100)
    test_sequences = pad_sequences(tokenizer.texts_to_sequences(test_text), maxlen=100)
    return tokenizer, train_sequences, test_sequences


def define_model():
    """Define a convolutional neural network model using Keras."""
    input_layer = Input(shape=(100,))
    embedding_layer = Embedding(input_dim=5000, output_dim=50)(input_layer)
    conv_layer = Conv1D(filters=128, kernel_size=5, activation='relu')(embedding_layer)
    pooling_layer = MaxPooling1D(pool_size=5)(conv_layer)
    flatten_layer = Flatten()(pooling_layer)
    output_layer = Dense(units=1, activation='sigmoid')(flatten_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, train_sequences, train_labels, test_sequences, test_labels):
    """Train the convolutional neural network model."""
    model.fit(
        train_sequences, 
        train_labels, 
        epochs=1, 
        batch_size=1, # Works better, dont be a puss
        validation_data=(test_sequences, test_labels)
    )


def save_model(model):
    """Save the trained convolutional neural network model to a file."""
    model.save('Models/ai_detection_model.h5')


def classify_input(model, tokenizer):
    """Classify a text string input as AI or human using the trained model and tokenizer."""

    while True:
        user_input = input('Enter a text string to classify (or "exit" to quit): ')

        if user_input.lower() == 'exit':
            break

        if user_input.lower() == '':
            continue

        sequence = pad_sequences(tokenizer.texts_to_sequences([user_input]), maxlen=100)
        prediction = model.predict(sequence)[0][0]

        if prediction > 0.5:
            print(f"Message classified as AI.")
        else:
            print(f"Message is classified as human.")

if __name__ == '__main__':
    train_text, test_text, train_labels, test_labels = load_data()
    tokenizer, train_sequences, test_sequences = tokenize_data(train_text, test_text)
    model = define_model()
    train_model(model, train_sequences, train_labels, test_sequences, test_labels)
    save_model(model)
    classify_input(model, tokenizer)
