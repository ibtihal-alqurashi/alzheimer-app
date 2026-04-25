import streamlit as st

def login():
    st.sidebar.title("🔐 Login")

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login"):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
        else:
            st.sidebar.error("Invalid credentials")

def check_login():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    return st.session_state["logged_in"]