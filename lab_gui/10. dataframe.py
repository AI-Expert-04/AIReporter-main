import streamlit as st
import pandas as pd

data = pd.read_csv('../data/titles_test.csv')
st.dataframe(data)
