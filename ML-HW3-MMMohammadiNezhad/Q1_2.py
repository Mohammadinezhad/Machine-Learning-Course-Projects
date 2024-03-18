import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv(r'C:\Users\mhd76\Desktop\master\courses\M L\projects\t 3\Book1.csv')

# Assuming your data has features, replace this with your feature matrix
df = pd.DataFrame(data, columns=['CF', 'K', 'peak', 'RMS'])

# Select relevant features for clustering
selected_features = df[['CF', 'K', 'peak', 'RMS']]

# Scale the features (important for GMM)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(selected_features)

# Choose the number of components (clusters) - you may need to tune this parameter
n_components = 2

# Apply GMM clustering
gmm = GaussianMixture(n_components, random_state=42)
df['cluster'] = gmm.fit_predict(scaled_data)

# Add the 'cluster' column to the original dataset
data['cluster'] = df['cluster']


# Create scatter plot with controlled point size
plt.scatter(data['CF'], data['K'], c=data['cluster'], cmap='viridis', s=2)
plt.title(f'GMM Clustering (n_components={n_components})')
plt.xlabel('CF')
plt.ylabel('K')
plt.show()
