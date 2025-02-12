import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering

def load_data(file_path):
    
    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print("Error loading dataset:", e)
        return None

def preprocess_data(data):
    
    if data is None:
        return None, None

    data.dropna(inplace=True)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled, scaler

def hierarchical_clustering(data_scaled, clusters):
    
    if data_scaled is None:
        return None

    clustering_model = AgglomerativeClustering(n_clusters=clusters)
    labels = clustering_model.fit_predict(data_scaled)
    return labels

def main():
    file_path = "wine.csv"
    wine_data = load_data(file_path)
    user_clusters = input("Enter number of clusters (default is 3): ").strip()
    clusters = int(user_clusters) if user_clusters else 3

    preprocessed_data, _ = preprocess_data(wine_data)
    labels = hierarchical_clustering(preprocessed_data, clusters)

    if labels is not None:
        wine_data["Cluster"] = labels
        print(wine_data[["Alcohol", "Color_Intensity", "Cluster"]].head(10))

if __name__ == "__main__":
    main()
