import pickle
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier


with open('rf_model(1).pkl', 'rb') as file:
    rf1_model = pickle.load(file)