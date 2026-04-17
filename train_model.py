import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

tf.config.set_visible_devices([], 'GPU')

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train[:10000]
y_train = y_train[:10000]

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

model = keras.Sequential([
    layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x_train,
    y_train,
    epochs=3,
    validation_data=(x_test, y_test)
)

loss, acc = model.evaluate(x_test, y_test)
print(f"Acurácia final: {acc:.4f}")

os.makedirs("models", exist_ok=True)
model.save("models/model.h5")