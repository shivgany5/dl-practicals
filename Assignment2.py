# First, install the missing OpenCV package
!pip install opencv-python

# Then import the libraries
import os
import cv2  # This will work after installation
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

data = []
labels = []

classes = {"modi":0,"trump":1}

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
for class_name, label in classes.items():
    folder_path = class_name  # because folders are at same level

    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)

        img = cv2.imread(img_path)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # If face detected, take the first one
        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            
            data.append(face)
            labels.append(label)
            break   # use only first detected face


X = np.array(data)
y = np.array(labels)

X = X.reshape(X.shape[0], 100, 100, 1)  # Add channel dimension
X = X / 255.0  # Normalize

y = to_categorical(y, 2)  # 2 classes

print("Dataset shape:", X.shape)
print("Labels shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100,100,1)),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),

    Dense(2, activation='softmax')
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

model.summary()


history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1
)


loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)


def predict_image(img_path):
    img = cv2.imread(img_path)

    if img is None:
        print("❌ Image not found. Check the path.")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    print("Faces detected:", len(faces))  # DEBUG

    if len(faces) == 0:
        print("❌ No face detected. Try another image or adjust parameters.")
        plt.imshow(gray, cmap='gray')
        plt.title("No Face Detected")
        plt.show()
        return

    for (x,y,w,h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100,100))
        face = face.reshape(1,100,100,1) / 255.0

        pred = model.predict(face)
        cls = np.argmax(pred)

        print("Raw prediction:", pred)

        if cls == 0:
            print("Prediction: Modi")
        else:
            print("Prediction: Trump")

        plt.imshow(gray[y:y+h, x:x+w], cmap='gray')
        plt.title("Detected Face")
        plt.show()

        break

predict_image("modi/modi4.jpg")
