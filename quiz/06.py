import tensorflow as tf

embedding_model = tf.keras.models.load_model('../models/embedding.h5')
rnn_model = tf.keras.models.load_model('../models/rnn.h5')
dense_model = tf.keras.models.load_model('../models/dense.h5')
softmax_model = tf.keras.models.load_model('../models/softmax.h5')

print(embedding_model.predict([[0, 1, 2]]))
print(rnn_model.predict([[0, 1, 2]]))
print(dense_model.predict([[0, 1, 2]]))
print(softmax_model.predict([[0, 1, 2]]))
