import pandas as pd

data = pd.read_csv('../data/titles_test.csv')

titles = data['title'].values
print(titles)
