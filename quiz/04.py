import tensorflow as tf
import pandas as pd

data = pd.read_csv('../data/titles.csv')
titles = data['title'].values

tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(titles)
word_count = len(tokenizer.word_index) + 1

x = []
y = []

for i in range(len(titles)):
    sequence = tokenizer.texts_to_sequences([titles[i]])[0]
    for i in range(1, len(sequence)):
        x.append(sequence[:i])
        y.append(sequence[i])

max_len = max(len(i) for i in x)
x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len)
y = tf.keras.utils.to_categorical(y, num_classes=word_count)

print(x)
print(y)
