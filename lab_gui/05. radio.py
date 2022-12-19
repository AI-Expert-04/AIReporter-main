import streamlit as st

select = st.sidebar.radio('라디오', ('A', 'B', 'C'))
st.text(select)
