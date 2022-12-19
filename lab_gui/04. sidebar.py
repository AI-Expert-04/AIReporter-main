import streamlit as st

length = st.sidebar.slider('슬라이더', 1, 100, 50)
st.sidebar.text(length)
