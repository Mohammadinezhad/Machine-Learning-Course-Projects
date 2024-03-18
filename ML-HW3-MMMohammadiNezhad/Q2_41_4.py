from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.datasets import load_breast_cancer
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Breast Cancer dataset
b_cancer = load_breast_cancer()

X_train = b_cancer.data[:469, :9]
X_test = b_cancer.data[470:568, :9]
y_train = b_cancer.target[:469]
y_test = b_cancer.target[470:568]

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create an MLP classifier with 1 hidden layer and 4 perceptrons
mlp_classifier = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42, learning_rate_init=0.01, solver='sgd', momentum=0.9)

# Train the classifier on the scaled training data
mlp_classifier.fit(X_train_scaled, y_train)

# Make predictions on the scaled test data
y_pred = mlp_classifier.predict(X_test_scaled)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the confusion matrix
print("\nConfusion Matrix:")
print(conf_matrix)

# Display other metrics if needed
classification_report_str = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_report_str)

# Display the confusion matrix using seaborn and matplotlib
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d',
            xticklabels=b_cancer.target_names, yticklabels=b_cancer.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
