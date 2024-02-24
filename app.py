import streamlit as st

selected_options = []

st.set_page_config(page_title="Suicide Help", layout="wide")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a psychiatrist. Give the user specific steps and suggestions to improve their condition",
    },
    {"role": "user", "content": "I am feeling lonely, whom should I approach?"},
]

nav_pages = ['Regi', 'Converse']
selected_page = st.sidebar.selectbox("Navigate: ", nav_pages)

if selected_page == 'Regi':
    st.title("Regi page")
    st.write("Answer the following questions to help us understand your situation better.")

    # Form with questions and checkbox options
    history_of_attempts = st.checkbox("1. History of suicide attempts (Yes)")
    substance_abuse = st.checkbox("2. Substance abuse (Yes)")
    trauma_and_abuse = st.checkbox("3. Trauma and abuse (Yes)")
    chronic_pain = st.checkbox("4. Chronic pain (Yes)")
    loss_and_grief = st.checkbox("5. Loss and grief (Yes)")
    social_isolation = st.checkbox("6. Social isolation (Yes)")
    financial_trouble = st.checkbox("7. Financial trouble ")
    unemployment = st.checkbox("8. Unemployment ")
    physical_movement = st.checkbox("9. Any physical movement ")

    # Generate response based on selected checkboxes
    if st.button("Submit"):
        if history_of_attempts:
            selected_options.append("History of suicide attempts")
        if substance_abuse:
            selected_options.append("Substance abuse")
        if trauma_and_abuse:
            selected_options.append("Trauma and abuse")
        if chronic_pain:
            selected_options.append("Chronic pain")
        if loss_and_grief:
            selected_options.append("Loss and grief")
        if social_isolation:
            selected_options.append("Social isolation")
        if financial_trouble:
            selected_options.append("Financial trouble")
        if unemployment:
            selected_options.append("Unemployment")
        if physical_movement:
            selected_options.append("Any physical movement")

    # Display selected options
    st.write("Selected options:", ", ".join(selected_options))

elif selected_page == 'Converse':
    st.title("Echo Bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("What is up?"):
        st.chat_message("user")
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = f"Echo: {prompt}"
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})