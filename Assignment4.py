import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("rain_forecasting assign4.csv")

print(df.head())
print(df.info())



df['RainToday'] = df['RainToday'].map({'Yes':1, 'No':0})
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes':1, 'No':0})
df = df.dropna()


features = [
    'MinTemp', 'MaxTemp', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'WindSpeed9am', 'WindSpeed3pm',
    'RainToday'
]

X = df[features].values
y = df['RainTomorrow'].values


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)


def create_sequences(X, y, time_steps=7):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 7
X_seq, y_seq = create_sequences(X_scaled, y, time_steps)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

model = Sequential([
    LSTM(64, activation='tanh', return_sequences=False, input_shape=(time_steps, X.shape[1])),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')   # binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)


loss, acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", acc)
y_pred = model.predict(X_test)
y_pred_label = (y_pred > 0.5).astype(int)
plt.figure(figsize=(12,6))
plt.plot(y_test[:200], label="Actual Rain Tomorrow")
plt.plot(y_pred_label[:200], label="Predicted")
plt.legend()
plt.title("Rain Forecasting (LSTM)")
plt.show()
