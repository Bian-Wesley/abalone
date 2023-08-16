import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from keras import layers

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("abalone.csv")
df["Age"] = df["Rings"] + 1.5
df = df.drop(["Rings"], axis = 1)

x = df.drop(["Age", "Sex"], axis = 1)
y = df["Age"]
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size = 0.2, random_state = 22)

model = keras.Sequential([
    layers.Dense(256, activation = "relu", input_shape = [7]),
    layers.BatchNormalization(),
    layers.Dense(256, activation = "relu"),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1, activation = "relu")
])
model.compile(
    loss = "mae",
    optimizer = "adam", 
    metrics = ["mape"]
)
#model.summary()

history = model.fit(x_train, y_train, epochs = 20, verbose = 1, batch_size = 64, validation_data = (x_val, y_val))
'''
hist_df = pd.DataFrame(history.history)
hist_df["loss"].plot()
hist_df["val_loss"].plot()
plt.title("loss with validation loss over epochs")
plt.legend()
plt.show()
'''
# Convert the model.
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)

print("abalones with these ranodm characteristics characteristics has an age of ")
#print(model.predict([[1, 2, 3, 4, 5, 6, 7]]))
print(
    model.predict(
        [
            [0.455,0.365,0.095,0.514,0.2245,0.101,0.15],
            [0.53,0.42,0.135,0.677,0.2565,0.1415,0.21]
        ]
    )
)
