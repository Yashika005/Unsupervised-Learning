import streamlit as st
import pandas as pd
import numpy as np
import base64  
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

def set_background(image_file):
    with open(image_file, "rb") as image:
        image_data = image.read()
    b64_image = base64.b64encode(image_data).decode()

    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        opacity: 0.9; 
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

@st.cache_data
def load_data():
    data = pd.read_csv("wine.csv")  
    return data

def preprocess_data(data):
    data.dropna(inplace=True)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def perform_clustering(method, data_scaled, clusters=3, eps=0.5, min_samples=5):
    if method == "K-Means":
        model = KMeans(n_clusters=clusters, random_state=42)
    elif method == "Hierarchical":
        model = AgglomerativeClustering(n_clusters=clusters)
    elif method == "DBSCAN":
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        return None
    
    labels = model.fit_predict(data_scaled)
    return labels

set_background("wine.png")  

st.title("🍷 Wine Dataset Clustering App 🍷")
st.write("## Explore Clustering on Wine Dataset Using Different Algorithms")

data = load_data()
st.write("### Preview of the Dataset:")
st.dataframe(data.head())

clustering_method = st.sidebar.selectbox("Choose Clustering Method:", ["K-Means", "Hierarchical", "DBSCAN"])

if clustering_method in ["K-Means", "Hierarchical"]:
    clusters = st.sidebar.slider("Number of Clusters:", min_value=2, max_value=10, value=3)
else:
    clusters = None

if clustering_method == "DBSCAN":
    eps = st.sidebar.slider("Epsilon (eps):", min_value=0.1, max_value=2.0, value=0.5)
    min_samples = st.sidebar.slider("Min Samples:", min_value=1, max_value=20, value=5)
else:
    eps, min_samples = None, None

data_scaled, scaler = preprocess_data(data)

labels = perform_clustering(clustering_method, data_scaled, clusters, eps, min_samples)
data["Cluster"] = labels

st.write("### Clustering Results:")
st.dataframe(data[["Alcohol", "Color_Intensity", "Cluster"]])

st.write("### Cluster Distribution")
st.bar_chart(data["Cluster"].value_counts())

st.write("### Cluster Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=data["Alcohol"], y=data["Color_Intensity"], hue=data["Cluster"], palette="viridis", s=100, alpha=0.8)
plt.xlabel("Alcohol")
plt.ylabel("Color Intensity")
plt.title(f"{clustering_method} Clustering")
plt.legend(title="Cluster")
st.pyplot(fig)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

data_pca = pd.DataFrame(data_pca, columns=['PC1', 'PC2'])
data_pca['Cluster'] = labels

st.write("### PCA Visualization")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=data_pca['PC1'], y=data_pca['PC2'], hue=data_pca['Cluster'], palette='coolwarm', s=100, alpha=0.8)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title(f"PCA Visualization of {clustering_method} Clusters")
st.pyplot(fig)
