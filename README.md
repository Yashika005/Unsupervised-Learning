# Wine Clustering Project

## Overview
This project explores different clustering algorithms on a wine dataset to group wines based on their characteristics. It includes implementations of K-Means, Hierarchical Clustering, and DBSCAN, along with a Streamlit web application for interactive visualization.

## Dataset
The dataset used is `wine.csv`, which contains various chemical properties of wines. Key attributes used for clustering include:
- Alcohol
- Color Intensity
- Other chemical properties

## Clustering Methods
### 1. K-Means Clustering (`wine_kmeans.py`)
- Uses the K-Means algorithm to group wines into clusters.
- Requires specifying the number of clusters.
- Standardizes the data before clustering.

### 2. Hierarchical Clustering (`wine_hierarchical.py`)
- Uses Agglomerative Clustering to form hierarchical clusters.
- Number of clusters can be adjusted by the user.

### 3. DBSCAN Clustering (`wine_dbscan.py`)
- Uses the DBSCAN algorithm, which does not require predefining the number of clusters.
- Requires setting `eps` (epsilon) and `min_samples` parameters.

## Streamlit Application (`wine_clustering_app.py`)
- Provides an interactive interface for clustering wines using different methods.
- Allows users to select clustering algorithms and tweak parameters.
- Displays clustering results using tables and visualizations.

## Installation & Dependencies
Ensure you have Python installed, then install the required packages using:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn streamlit
```

## Running the Project
1. **Running individual clustering scripts**:
   ```bash
   python wine_kmeans.py
   python wine_hierarchical.py
   python wine_dbscan.py
   ```
2. **Launching the Streamlit app**:
   ```bash
   streamlit run wine_clustering_app.py
   ```

## Outputs
- Console outputs for clustering scripts display cluster assignments.
- Streamlit app provides a user-friendly interface with tables and visualizations.

## File Structure
```
|-- wine.csv                # Wine dataset
|-- wine_kmeans.py          # K-Means clustering script
|-- wine_hierarchical.py    # Hierarchical clustering script
|-- wine_dbscan.py          # DBSCAN clustering script
|-- wine_clustering_app.py  # Streamlit web application
|-- wine.png                # Watermark image for app
|-- wine_image.jpg          # Additional visualization image
```

