import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Heart Data Clustering using K-Means")

df = pd.read_csv("heart.csv")

if 'target' in df.columns:
    X = df.drop('target', axis=1)
else:
    X = df.copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = st.slider("Select number of clusters", 2, 10, 3)

kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df['Cluster'] = clusters

st.write(df.head())

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
ax.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters)
ax.set_title("Cluster Visualization")
st.pyplot(fig)