from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.datasets import load_iris

def plot_line_and_points_sklearn(clf, data, feature_x, feature_y, ax):
    X = data[[feature_x, feature_y]].values
    y = data['Species'].values

    # Plot data points for both classes
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], marker='o', label='Class 1')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x', label='Class 2')

    # Calculate the decision boundary (classification line)
    slope = -clf.coef_[0][0] / clf.coef_[0][1]
    intercept = -clf.intercept_ / clf.coef_[0][1]

    x_vals = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
    decision_boundary = slope * x_vals + intercept

    ax.plot(x_vals, decision_boundary, label='Classification Line', color='red')

    ax.set_xlabel(feature_x)
    ax.set_ylabel(feature_y)
    ax.set_title(f'Perceptron Classification Line: {feature_x} vs {feature_y}')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(color='gray', linestyle='--', linewidth=0.5)
    ax.legend()

# Load Iris dataset and create a binary classification problem
iris = load_iris()

data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['Species'] = iris.target
filtered_data = data[(data.index >= 0) & (data.index < 100) & (data['Species'].isin([0, 1]))]

filtered_data.loc[filtered_data['Species'] == 0, 'Species'] = -1
filtered_data.loc[filtered_data['Species'] == 1, 'Species'] = 1

labels = filtered_data['Species']
features = filtered_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
features_list = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create and train the Perceptron model
perceptron_model = Perceptron(max_iter=100000, tol=1e-3)
perceptron_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = perceptron_model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Create a single figure for all pairs of features
num_plots = len(list(combinations(features_list, 2)))
fig, axes = plt.subplots(nrows=num_plots // 2, ncols=2, figsize=(10, 10))

# Flatten the 2D array of subplots for easy indexing
axes = axes.flatten()

# Plot decision boundary and data points for each pair of features
for i, feature_pair in enumerate(combinations(features_list, 2)):
    plot_line_and_points_sklearn(perceptron_model, filtered_data, feature_pair[0], feature_pair[1], axes[i])

plt.tight_layout()
plt.show()
