import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 4),
    tf.keras.layers.SimpleRNN(5),
    tf.keras.layers.Dense(6)
])
model.summary()
