import tensorflow as tf
import os

model = tf.keras.models.load_model("model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

os.makedirs("models", exist_ok=True)

with open("model.tflite", "wb") as f:
    f.write(tflite_model)