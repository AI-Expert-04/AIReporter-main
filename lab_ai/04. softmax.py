import tensorflow as tf

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5),
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(6),
    tf.keras.layers.Softmax()
])
model.summary()
