import streamlit as st
import pandas as pd
import numpy as np
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
#import SessionState

st.set_page_config(layout="wide")
st.title("Nutritional Partner to Health!")
st.cache()

data = pd.read_csv("dataset.csv")
option = st.selectbox(label="Select a Recipe", options = data.title)
if option == 'None':
    st.write("Please choose the Recipe Title from the list below")
else:
    st.write('You selected:', option)
    features = data.iloc[:, 1:7]
    sc = StandardScaler()
    scaled_data = sc.fit_transform(features)
    knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    knn.fit(features)
    index_of_recipe = data[data['title'] == option].index.values
    distance, indices = knn.kneighbors(features.iloc[index_of_recipe,:], n_neighbors = 5)
    for i in indices:
        st.write(data.iloc[i])










