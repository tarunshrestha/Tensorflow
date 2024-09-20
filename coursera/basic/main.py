import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense,  Flatten # type: ignore
from tensorflow.keras.models import Sequential # type: ignore

from core import myCallback

os.system("clear")

print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

# X = Images, y= Labels
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Normalization (0-1)
X_train = X_train / 255.0
X_test = X_test / 255.0

# Design Model
model = Sequential([
    Flatten(input_shape= (28, 28)), # spacifying expected data to be 28*28, error will occur if not flatten due to size diff in other datas.
    Dense(128, activation=tf.nn.relu), # Neurons number 
    Dense(10, activation=tf.nn.softmax) # 10 neurans = 10 classes of clothes, if kept less or more it will send error unexpected value
])

# Optimization
model.compile(optimizer=tf.optimizers.Adam(), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy']
              )

# Train model
callbacks = myCallback() # used to stop training after reaching 60% or more to stop all the time consuming runs
model.fit(X_train, y_train, epochs=5, callbacks=[callbacks]) # perfect epochs is required if more it will be over fitting

# Evaluate
model.evaluate(X_test, y_test)

print("Check")
