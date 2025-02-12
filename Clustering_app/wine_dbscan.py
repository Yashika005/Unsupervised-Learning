import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

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

def dbscan_clustering(data_scaled, eps, min_samples):
    
    if data_scaled is None:
        return None

    clustering_model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = clustering_model.fit_predict(data_scaled)
    return labels

def main():
    file_path = "wine.csv"
    wine_data = load_data(file_path)
    
    user_eps = input("Enter the value for eps (default is 0.5): ").strip()
    eps = float(user_eps) if user_eps else 0.5

    user_min_samples = input("Enter min_samples (default is 5): ").strip()
    min_samples = int(user_min_samples) if user_min_samples else 5

    preprocessed_data, _ = preprocess_data(wine_data)
    labels = dbscan_clustering(preprocessed_data, eps, min_samples)

    if labels is not None:
        wine_data["Cluster"] = labels
        print(wine_data[["Alcohol", "Color_Intensity", "Cluster"]].head(10))

if __name__ == "__main__":
    main()
