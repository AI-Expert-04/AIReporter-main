import tensorflow as tf

data = [0, 1, 2, 3]

categorical_data = tf.keras.utils.to_categorical(data, num_classes=4)
print(categorical_data)
