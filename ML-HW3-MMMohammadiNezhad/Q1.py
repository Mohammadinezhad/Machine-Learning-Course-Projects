import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv(r'C:\Users\mhd76\Desktop\master\courses\M L\projects\t 3\Book1.csv')

# Assuming your data has features, replace this with your feature matrix
df = pd.DataFrame(data, columns=['CF', 'K', 'peak', 'RMS'])

# Select relevant features for clustering
selected_features = df[['CF', 'K', 'peak', 'RMS']]

# Scale the features (important for k-means)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_features)

# Choose the number of clusters (k) - you may need to tune this parameter
n_clusters = 2

# Apply k-means clustering
kmeans = KMeans(n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)

data['cluster'] = df['cluster']

# Visualize the clusters
plt.scatter(df['CF'], df['K'], c=df['cluster'], cmap='viridis',s=1)
plt.title(f'K-Means Clustering (k={n_clusters})')
plt.xlabel('CF')
plt.ylabel('K')
plt.show()

