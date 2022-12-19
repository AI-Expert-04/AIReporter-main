import tensorflow as tf
import numpy as np
import os

index_to_word = ['가', '나', '다', '라']

# 가나다라, 다나가가
x = [[0, 1, 2], [2, 1, 0]]
y = [[0, 0, 0, 1], [1, 0, 0, 0]]

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(4, 5),
    tf.keras.layers.SimpleRNN(3),
    tf.keras.layers.Dense(4),
    tf.keras.layers.Softmax()
])

loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

if not os.path.exists('../logs'):
    os.mkdir('../logs')

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs')

model.fit(x, y, epochs=100, callbacks=[tensorboard])

if not os.path.exists('../models'):
    os.mkdir('../models')

model.save('../models/test.h5')
