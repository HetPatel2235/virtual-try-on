import streamlit as st
from streamlit_javascript import st_javascript

token = st_javascript("sessionStorage.getItem('vton_auth');")
st.write(f"Token: {token}")

if st.button("Login"):
    st_javascript("sessionStorage.setItem('vton_auth', 'my_token');")
    st.rerun()

if st.button("Logout"):
    st_javascript("sessionStorage.removeItem('vton_auth');")
    st.rerun()
