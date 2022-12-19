from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
import ssl
import pandas as pd

context = ssl._create_unverified_context()

hds = {'User-Agent': 'Mozilla/5.0'}

web_url = input()
request = Request(web_url, headers=hds)
response = urlopen(request, context=context)
html = response.read()
soup = BeautifulSoup(html, 'html.parser')
headlines = soup.find_all("a", {'class', 'nclicks(cls_nav.clsart1)'})
titles = []

for i in range(len(headlines)):
    title = headlines[i].text
    if len(title) < 10:
        continue
    titles.append(title)

data = pd.DataFrame({'title': titles})
data.to_csv('../data/titles.csv', encoding='utf-8')
