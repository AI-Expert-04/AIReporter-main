import pandas as pd

data = pd.read_csv('../data/titles.csv')

titles = data['title'].values
print(titles)
