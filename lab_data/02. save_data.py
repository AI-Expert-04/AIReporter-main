import pandas as pd
import os

if not os.path.exists('../data'):
    os.mkdir('../data')

titles = ['제목 1', '제목 2', '제목 3']

data = pd.DataFrame({'title': titles})
data.to_csv('../data/titles.csv', encoding='utf-8')
