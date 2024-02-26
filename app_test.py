import streamlit as st
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
from io import BytesIO

selected_options = []
selected_option_string = ""

st.set_page_config(page_title="Suicide Help", layout="wide")
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a psychiatrist. Give the user specific steps and suggestions to improve their condition",
    },
]

nav_pages = ['Regi', 'Converse','Classification']
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
    selected_options_string = "I am experiencing" + ", ".join(selected_options)
    messages.append({"role": "user", "content": selected_options_string})

elif selected_page == 'Converse':
    st.title("Echo Bot")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user")
        st.write(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        response = f"Echo: {prompt}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

elif selected_page == 'Classification':

    def func(file_content):
        if file_content is None:
            return None  # Handle the case where the file upload failed

        # Continue with your existing logic
        df = pd.read_csv(file_content
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         
                         )
        df1 = df.filter(regex='^(?!.*COH)')
        columns_to_remove = ['education', 'date', 'Unnamed: 122', 'no.', 'eeg.date']
        df2 = df1.drop(columns=columns_to_remove, errors='ignore')
        df3 = df2.dropna()
        df3['sex'] = df3['sex'].replace({'M': 0, 'F': 1})
        df3 = df3.drop(['specific.disorder'], axis=1)
        df3['sex'] = df3['sex'].replace({'M': 0, 'F': 1})

        with open('pca_model(1).pkl', 'rb') as file:
            pca = pickle.load(file)

        X = df3.drop(columns=['main.disorder'])  # Features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        pca_data = pca.transform(X)

        with open('rf_model(1).pkl', 'rb') as file:
            rf1_model = pickle.load(file)

        predicted_labels = rf1_model.predict(pca_data)

        return predicted_labels

    def main():
        st.title("Classification Page")

        with st.expander("Upload CSV File"):
            uploaded_file = st.file_uploader("Choose a file", type=["csv"])
            if uploaded_file is not None:
                #file_content = uploaded_file.read()
                # predicted_labels = func(uploaded_file.read())
                predicted_labels = func(uploaded_file)
                st.write("Predicted labels:", predicted_labels)

    if __name__ == "__main__":  
        main()