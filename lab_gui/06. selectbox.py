import streamlit as st

select = st.selectbox('선택 박스', ['A', 'B', 'C'])
st.text(select)
