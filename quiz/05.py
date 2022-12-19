import tensorflow as tf
import os

if not os.path.exists('../models'):
    os.mkdir('../models')

embedding_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5),
])
embedding_model.save('../models/embedding.h5')

rnn_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5),
    tf.keras.layers.SimpleRNN(3),
])
rnn_model.save('../models/rnn.h5')

dense_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5),
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(4),
])
dense_model.save('../models/dense.h5')

softmax_model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(3, 5),
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Softmax()
])
softmax_model.save('../models/softmax.h5')
