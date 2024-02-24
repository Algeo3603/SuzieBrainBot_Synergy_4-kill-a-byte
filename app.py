import streamlit as st


st.set_page_config(page_title="Suicide Help", layout="wide")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a psychiatrist. Give the user specififc steps and suggestions to improve their condition",
    },
    # {"role": "user", "content": "I am feeling lonely, whom should I approach?"},
]


nav_pages = ['Regi', 'Converse']
selected_page = st.sidebar.selectbox("Navigate: ", nav_pages)

if selected_page == 'Regi':
    st.title("Regi page")
    st.write("Enter your details")
elif selected_page == 'Converse':
    st.title("Chatbot")
    st.write("You suck, you should suicide")

    chat_ui = st.container(height=400)

    # for message in messages[1:]:
        # if message['role']:
