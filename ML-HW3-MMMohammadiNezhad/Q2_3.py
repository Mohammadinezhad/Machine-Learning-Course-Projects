from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the Breast Cancer dataset
b_cancer = load_breast_cancer()

# Use all features
X = b_cancer.data[:, :9]
y = b_cancer.target

# Standardize the data set
scaler_standard = StandardScaler()
X_standardized = scaler_standard.fit_transform(X)

# Normalize the standardized features
scaler_minmax = MinMaxScaler()
X_normalized = scaler_minmax.fit_transform(X_standardized)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Create an MLP classifier with 2 hidden layers and 5 perceptrons in each layer
mlp_classifier = MLPClassifier(hidden_layer_sizes=(5, 5), max_iter=1000, random_state=42)

# Train the classifier on the training data
mlp_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = mlp_classifier.predict(X_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Display other metrics if needed
classification_report_str = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_report_str)

# Now, classify new instances
new_instances = [
    [0.2, 0.1, 0.4, 0.3, 0.6, 0.8, 0.5, 0.6, 0.6],
    [0.5, 0.3, 0.4, 0.2, 0.5, 0.7, 0.7, 0.2, 0.3],
    [0.7, 0.6, 0.3, 0.7, 0.9, 0.4, 0.9, 0.3, 0.2]
]

# Standardize and normalize the new instances
new_instances_standardized = scaler_standard.transform(new_instances)
new_instances_normalized = scaler_minmax.transform(new_instances_standardized)

# Predict the class labels for the new instances
predicted_labels = mlp_classifier.predict(new_instances_normalized)

# Display the predicted labels for the new instances
print("\nPredicted Labels for New Instances:")
print(predicted_labels)
