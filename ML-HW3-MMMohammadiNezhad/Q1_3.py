import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load your dataset
data = pd.read_csv(r'C:\Users\mhd76\Desktop\master\courses\M L\projects\t 3\Book1.csv')

# Assuming your data has features, replace this with your feature matrix
features = data[['CF', 'K', 'peak', 'RMS']]

# Scale the features for PCA
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Choose the number of principal components
n_components = min(features.shape[0], features.shape[1])  # Use min for safety
pca = PCA(n_components=n_components)

# Fit the PCA model and transform the data
principal_components = pca.fit_transform(scaled_features)

# Get the explained variance ratio for each principal component
explained_variance_ratio = pca.explained_variance_ratio_

# Print the percentage of variance explained by each principal component
for i in range(len(explained_variance_ratio)):
    print(f"Principal Component {i+1}: {explained_variance_ratio[i]*100:.2f}%")

# Get the Principal Component Indices (PCI)
pci = np.argsort(np.abs(pca.components_))[::-1][:, :n_components]

# Print the Principal Component Indices
print("Principal Component Indices (PCI):")
print(pci)

# Get the correlation of each parameter with the first principal component (PC1)
correlation_with_pc1 = pca.components_[0, :]

# Create a DataFrame for better visualization
correlation_df = pd.DataFrame({'Parameter': features.columns, 'Correlation with PC1': correlation_with_pc1})

# Print the correlation DataFrame
print("\nCorrelation with PC1:")
print(correlation_df)
