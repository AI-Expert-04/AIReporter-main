import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 4)
])
model.summary()
