import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('../models/softmax.h5')

predict = model.predict([[0, 1, 2]])
print(predict)

index_word = ['가', '나', '다', '라', '마', '바']

index = np.argmax(predict[0])

print(index)
print(index_word[index])
