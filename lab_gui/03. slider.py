import streamlit as st

length = st.slider('슬라이더', 1, 100, 50)
st.text(length)
