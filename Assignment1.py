import pandas as pd
import numpy as np
!pip install tensorflow
!pip install matplotlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

train_df = pd.read_csv("fashion-mnist_train.csv")
test_df = pd.read_csv("fashion-mnist_test.csv")

print(train_df.shape)
print(test_df.shape)

y_train = train_df["label"].values
y_test = test_df["label"].values

X_train = train_df.drop("label",axis=1).values
X_test = test_df.drop("label",axis = 1).values

print(X_train.shape, X_test.shape)

X_train = X_train/255.0
X_test = X_test / 255.0

# one-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

from tensorflow.keras.layers import Input

model = Sequential([
    Input(shape=(784,)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.summary()

model.compile(
    optimizer = "adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train,y_train_cat,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

test_loss,test_acc = model.evaluate(X_test,y_test_cat)
print("Test Accuracy:",test_acc)
print("Test Loss:",test_loss)

plt.figure(figsize=(15,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"])

plt.subplot(1,2,2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"])

plt.show()

index = 20  # choose any row

sample_image = X_train[index].reshape(28, 28)

plt.imshow(sample_image, cmap="gray")
plt.title("True Label: " + str(y_train[index]))
plt.show()

prediction = model.predict(X_train[index].reshape(1, 784))
print("Predicted Label:", np.argmax(prediction))
