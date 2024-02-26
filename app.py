import streamlit as st
import torch
from transformers import pipeline
import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")
selected_options = []
base_condition_statement = "I am experiencing the following: "

st.session_state.messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot who always responds in the style of a psychiatrist. Give the user specififc steps and suggestions to improve their condition",
    },
    # {"role": "user", "content": "How can I suicide?"},
]

def func(path):
    dfr=pd.read_csv(path)
    df = pd.read_csv("EEG.machinelearing_data_BRMH.csv")
    df1 = df.filter(regex='^(?!.*COH)')
    dfr = dfr.filter(regex='^(?!.*COH)')
    columns_to_remove = ['education', 'date', 'Unnamed: 122','no.','eeg.date']
    df2 = df1.drop(columns=columns_to_remove, errors='ignore')
    dfr = dfr.drop(columns=columns_to_remove, errors='ignore')
    df3 = df2.dropna()
    df3['sex'] = df3['sex'].replace({'M': 0, 'F': 1})
    dfr['sex'] = dfr['sex'].replace({'M': 0, 'F': 1})
    df3=df3.drop(['specific.disorder'],axis=1)
    dfr=dfr.drop(['specific.disorder'],axis=1)


    # Assuming df is your DataFrame containing the data
    X = df3.drop(columns=['main.disorder'])  # Features

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    pca = PCA(n_components=20)
    X_pca = pca.fit_transform(X_scaled)
    y=df3['main.disorder']
    X_train=X_pca
    y_train=y
    # X_test=X_pca
    # y_test=y
    X_train, X_test1, y_train, y_test1 = train_test_split(X_pca, y, test_size=0.1, random_state=42)
    X_test, X_test2, y_test, y_test1 = train_test_split(X_pca, y, test_size=0.1, random_state=45)

    # Train a Random Forest classifier
    rf_classifier = RandomForestClassifier()
    rf_classifier.fit(X_train, y_train)
    rf_accuracy = rf_classifier.score(X_test, y_test)


    X = dfr.drop(columns=['main.disorder'])  # Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca_data = pca.transform(X)

    predicted_labels = rf_classifier.predict(pca_data) 
    return predicted_labels

st.set_page_config(page_title="Suzie :)", layout="wide")
# suzie_logo = "suzie.png"
# # st.write(
# #     f'<img src="{suzie_logo}" alt="Suzie!" style="position:absolute; top:10px; right:10px;" />',
# #     unsafe_allow_html=True
# # )
# st.image(suzie_logo, width=100, caption="Suzie!", use_container_width=False, output_format="PNG", clamp=True, channels="RGB", format="PNG")

nav_pages = ['Suzie wants to know', 'Converse with Suzie', 'EEG analysis']
selected_page = st.sidebar.selectbox("Navigate: ", nav_pages)

if selected_page == 'Suzie wants to know':
    st.title("Suzie wants to know")
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
    selected_options_string = "I am experiencing " + ", ".join(selected_options)
    st.session_state.messages.append({"role": "user", "content": selected_options_string})


elif selected_page == 'Converse with Suzie':
    st.title("Converse with Suzie")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    

    if prompt := st.chat_input("What is up?"):
        st.chat_message("user")
        st.write(prompt)
    # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        llm_prompt = pipe.tokenizer.apply_chat_template(st.session_state.messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(llm_prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        response = outputs[0]["generated_text"].split("<|assistant|>")[-1]
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.write(response)
        # Add assistant response to chat history    
        st.session_state.messages.append({"role": "assistant", "content": response})

elif selected_page == 'EEG analysis':
    st.title("EEG analysis")

    path = st.file_uploader("Upload a CSV", type=['csv'])

    if path is not None:
        predicted_labels = func(path)
        st.write("You maybe at risk for the following condition: ", predicted_labels[0])
        if predicted_labels[0] == 'Mood disorder':
            st.write(r"Approximately 15-20% of individuals with mood disorder are observed to be at elevated risk for suicide.")
        elif predicted_labels[0] == 'Schizophrenia':
            st.write(r"Approximately 5-10% of individuals with Schizophrenia are observed to be at elevated risk for suicide.")
        elif predicted_labels[0] == 'Anxiety disorder':
            st.write(r"Approximately 5-15% of individuals with Anxiety disorder are observed to be at elevated risk for suicide.")
        elif predicted_labels[0] == 'Trauma and stress related disorder':
            st.write(r"Approximately 10-20% of individuals with Trauma and stress related disorder are observed to be at elevated risk for suicide.")
        elif predicted_labels[0] == 'Addictive disorder':
            st.write(r"Approximately 10-25% of individuals with Addictive disorder (especially substance abuse) are observed to be at elevated risk for suicide.")
        elif predicted_labels[0] == 'Obsessive compulsive disorder':
            st.write(r"Approximately 5-10% of individuals with Obsessive compulsive disorder are observed to be at elevated risk for suicide.")
