import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset
dataset = pd.read_csv(r'C:\Users\mhd76\Desktop\master\courses\M L\projects\project\data\dataR2.csv')

# Extract the features (assuming all columns are features except the target column)
X = dataset.drop('Classification', axis=1) 

print(X)

# standardized the data set
scaler_standard = StandardScaler()
X_standardized = scaler_standard.fit_transform(X)

# Normalize the standardized features
scaler_minmax = MinMaxScaler()
X_normalized = scaler_minmax.fit_transform(X_standardized)

# Create a new DataFrame with the normalized features
normalized_dataset = pd.DataFrame(X_normalized, columns=X.columns)

# Instantiate Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the Random Forest model
rf_classifier.fit(X_normalized, dataset['Classification'])

# Access feature importance scores
feature_importances = rf_classifier.feature_importances_

# Sort features by importance
sorted_indices = feature_importances.argsort()[::-1]

# Display feature importance scores
print("\nFeature Importance Scores:")
print(pd.DataFrame({'Feature': X.columns[sorted_indices], 'Importance': feature_importances[sorted_indices]}))
