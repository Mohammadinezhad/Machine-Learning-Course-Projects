# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Load Iris dataset
iris = load_iris()
X = iris.data[51:151]  # Use only samples with index 51 to 150
y = iris.target[51:151]  # Corresponding labels

# Map labels to binary values (considering only two labels)
y_binary = [1 if label == 2 else 0 for label in y]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)

# Decision Tree model
dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, y_train)

# Predictions and metrics for Decision Tree
dt_predictions = dt_model.predict(X_test)

dt_conf_matrix = confusion_matrix(y_test, dt_predictions)
dt_precision = precision_score(y_test, dt_predictions)
dt_recall = recall_score(y_test, dt_predictions)

print("Decision Tree Confusion Matrix:")
print(dt_conf_matrix)
print("Decision Tree Precision:", dt_precision)
print("Decision Tree Recall:", dt_recall)
print("\n")

# k-Nearest Neighbors model
knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train, y_train)

# Predictions and metrics for k-Nearest Neighbors
knn_predictions = knn_model.predict(X_test)

knn_conf_matrix = confusion_matrix(y_test, knn_predictions)
knn_precision = precision_score(y_test, knn_predictions)
knn_recall = recall_score(y_test, knn_predictions)

print("k-Nearest Neighbors Confusion Matrix:")
print(knn_conf_matrix)
print("k-Nearest Neighbors Precision:", knn_precision)
print("k-Nearest Neighbors Recall:", knn_recall)
