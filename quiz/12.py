import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
import pandas as pd
import numpy as np
import os
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import ssl

menu = st.sidebar.radio("메뉴", ('사전 학습 모델', '모델 학습'))

if menu == '사전 학습 모델':
    model_file_names = os.listdir('../models')
    model_file_name = st.sidebar.selectbox('모델 선택', [''] + model_file_names)

    if model_file_name != '':
        data = pd.read_csv(f'../data/{model_file_name.split(".")[-2]}.csv')
        titles = data['title'].values

        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        tokenizer.fit_on_texts(titles)
        word_count = len(tokenizer.word_index) + 1

        index_word = ['' for _ in range(word_count)]
        for k, v in tokenizer.word_index.items():
            index_word[v] = k

        model = tf.keras.models.load_model(f'../models/{model_file_name}')

        st.title('AI Reporter')
        title = st.selectbox('키워드', [''] + list(tokenizer.word_index.keys()))
        length = st.slider('문장 길이', 1, 30, 5)

        if title != '':
            for i in range(length):
                sequence = tokenizer.texts_to_sequences([title])[0]
                predict = model.predict([sequence])
                index = np.argmax(predict[0])
                title += ' ' + index_word[index]
            st.text(title)
else:
    model_name = st.sidebar.text_input('모델 이름')
    if model_name != '':
        news_class = st.sidebar.selectbox('분야', ['정치', '경제', '사회', '생활/문화', 'IT/과학', '세계'])

        if news_class == '정치':
            link = '[네이버 뉴스 - 정치](https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=100)'
            headline_class = 'nclicks(cls_pol.clsart1)'
        elif news_class == '경제':
            link = '[네이버 뉴스 - 경제](https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=101)'
            headline_class = 'nclicks(cls_eco.clsart1)'
        elif news_class == '사회':
            link = '[네이버 뉴스 - 사회](https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=102)'
            headline_class = 'nclicks(cls_nav.clsart1)'
        elif news_class == '생활/문화':
            link = '[네이버 뉴스 - 생활/문화](https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=103)'
            headline_class = 'nclicks(cls_lif.clsart1)'
        elif news_class == 'IT/과학':
            link = '[네이버 뉴스 - IT/과학](https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=105)'
            headline_class = 'nclicks(cls_sci.clsart1)'
        elif news_class == '세계':
            link = '[네이버 뉴스 - 세계](https://news.naver.com/main/main.naver?mode=LSD&mid=shm&sid1=104)'
            headline_class = 'nclicks(cls_wor.clsart1)'

        st.sidebar.markdown(link, unsafe_allow_html=True)

        data_url = st.sidebar.text_input('데이터 URL')

        if data_url != '':
            context = ssl._create_unverified_context()
            hds = {'User-Agent': 'Mozilla/5.0'}

            request = Request(data_url, headers=hds)
            response = urlopen(request, context=context)
            html = response.read()
            soup = BeautifulSoup(html, 'html.parser')

            headlines = soup.find_all("a", {'class', headline_class})
            titles = []

            for i in range(len(headlines)):
                title = headlines[i].text
                if len(title) < 10:
                    continue
                titles.append(title)

            data = pd.DataFrame({'title': titles})

            st.title('데이터')
            st.dataframe(data)

            learning_rate = st.sidebar.number_input('Learning Rate', format="%.6f", step=0.000001, value=0.01)
            epochs = st.sidebar.slider('epochs', 100, 10000, 100)
            save_log = st.sidebar.checkbox('학습 로그 저장')

            if st.sidebar.button('학습 & 저장'):
                tokenizer = tf.keras.preprocessing.text.Tokenizer()
                tokenizer.fit_on_texts(titles)
                word_count = len(tokenizer.word_index) + 1

                index_word = ['' for _ in range(word_count)]
                for k, v in tokenizer.word_index.items():
                    index_word[v] = k

                x = []
                y = []

                for i in range(len(titles)):
                    sequence = tokenizer.texts_to_sequences([titles[i]])[0]
                    for i in range(1, len(sequence)):
                        x.append(sequence[:i])
                        y.append(sequence[i])

                max_len = max(len(i) for i in x)
                x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_len, padding='pre')
                y = tf.keras.utils.to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

                model = tf.keras.models.Sequential([
                    tf.keras.layers.Embedding(word_count, 10),
                    tf.keras.layers.SimpleRNN(32),
                    tf.keras.layers.Dense(word_count),
                    tf.keras.layers.Softmax()
                ])

                loss = tf.keras.losses.CategoricalCrossentropy()
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

                model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

                if save_log:
                    if not os.path.exists('../logs'):
                        os.mkdir('../logs')

                    tensorboard = tf.keras.callbacks.TensorBoard(log_dir='../logs')

                    history = model.fit(x, y, epochs=epochs, callbacks=[tensorboard])
                else:
                    history = model.fit(x, y, epochs=epochs)

                st.title('통계')
                st.dataframe(pd.DataFrame(history.history))

                fig = plt.figure()
                cnt = 1
                for k, v in history.history.items():
                    ax = fig.add_subplot(len(history.history), 1, cnt)
                    ax.title.set_text(k)
                    ax.plot(v)
                    cnt += 1

                st.title('그래프')
                st.pyplot(fig)

                data.to_csv(f'../data/{model_name}.csv', encoding='utf-8')
                model.save(f'../models/{model_name}.h5')
