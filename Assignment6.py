import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GRU, Dense
import matplotlib.pyplot as plt


vocab_size = 10000  # top 10k words

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocab_size)

print("Training samples:", len(X_train))
print("Testing  samples:", len(X_test))


max_len = 200  # all sequences will be padded to 200 tokens

X_train = pad_sequences(X_train, maxlen=max_len)
X_test  = pad_sequences(X_test,  maxlen=max_len)



lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    LSTM(128),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

lstm_model.summary()




history_lstm = lstm_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)


loss, acc = lstm_model.evaluate(X_test, y_test)
print("LSTM Test Accuracy:", acc)



gru_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
    GRU(128),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

gru_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])




history_gru = gru_model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)
gru_model.summary()




loss, acc = gru_model.evaluate(X_test, y_test)
print("GRU Test Accuracy:", acc)



plt.figure(figsize=(14,5))

# LSTM
plt.subplot(1,2,1)
plt.plot(history_lstm.history['accuracy'], label='Train Accuracy')
plt.plot(history_lstm.history['val_accuracy'], label='Val Accuracy')
plt.title("LSTM Accuracy")
plt.legend()

# GRU
plt.subplot(1,2,2)
plt.plot(history_gru.history['accuracy'], label='Train Accuracy')
plt.plot(history_gru.history['val_accuracy'], label='Val Accuracy')
plt.title("GRU Accuracy")
plt.legend()

plt.show()




word_index = imdb.get_word_index()
reverse_index = {value: key for key, value in word_index.items()}

def encode_review(text):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) for w in words]  # 2 = unknown word
    return pad_sequences([encoded], maxlen=max_len)




text = "This movie was absolutely amazing with great acting"
encoded = encode_review(text)

pred = lstm_model.predict(encoded)[0][0]
print("LSTM Sentiment:", "Positive" if pred < 0.5 else "Negative", "| Score:", pred)
