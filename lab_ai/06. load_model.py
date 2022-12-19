import tensorflow as tf

model = tf.keras.models.load_model('../models/softmax.h5')
model.summary()
