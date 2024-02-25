import streamlit as st
import pandas as pd
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Pickles", layout="wide")


uploaded_file = st.file_uploader("Upload a CSV", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    df1 = df.filter(regex='^(?!.*COH)')
    columns_to_remove = ['education', 'date', 'Unnamed: 122','no.','eeg.date']
    df2 = df1.drop(columns=columns_to_remove, errors='ignore')
    df3 = df2.dropna()
    df3['sex'] = df3['sex'].replace({'M': 0, 'F': 1})
    df3=df3.drop(['specific.disorder'],axis=1)
    df3['sex'] = df3['sex'].replace({'M': 0, 'F': 1})

    with open('pca_model(1).pkl', 'rb') as file:
        pca = pickle.load(file)

    X = df3.drop(columns=['main.disorder'])  # Features
    # scaler = StandardScaler()
    # X = scaler.fit_transform(X)
    pca_data = pca.transform(X)
    with open('rf_model(1).pkl', 'rb') as file:
        rf1_model = pickle.load(file)

    predicted_labels = rf1_model.predict(pca_data)
    # return predicted_labels
    st.write(predicted_labels)

