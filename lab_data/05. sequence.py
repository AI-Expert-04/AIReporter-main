import tensorflow as tf
import pandas as pd

data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)
print(tokenizer.word_index)

sequence = tokenizer.texts_to_sequences([titles[0]])[0]
print(titles[0])
print(sequence)
