import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.datasets import load_iris

def perceptron(X, y):
    # Add bias term to the features
    learning_rate = 0.0001
    n_iterations = 100
    X['bias'] = 1
    X = X.values
    y = y.values
    n = 0 

    # Initialize weights
    weights = np.zeros(X.shape[1])

    for j in range(n_iterations):
        for i in range(X.shape[0]):
            # Compute activation
            xop = X[i]
            yop = y[i]
            activation = np.dot(xop, weights) 
            activation = yop * activation

            # Apply step function (threshold at 0)
            if activation <= 0:
                for k in range(len(weights)):
                    weights[k] = weights[k] + learning_rate * y[i] * X[i, k]
                n = 1 + n

        # Update weights after going through all data points
        j += 1
    
    print(weights)
    print(n)
    return weights[:-1], weights[-1]  # Return weights and bias

def plot_line_and_points(weights, bias, data, feature_x, feature_y, ax):
    X = data[[feature_x, feature_y]].values
    y = data['Species'].values

    # Plot data points for both classes
    ax.scatter(X[y == -1][:, 0], X[y == -1][:, 1], marker='o', label='Class 1')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], marker='x', label='Class 2')

    # Calculate the decision boundary (classification line)
    feature_idx = [data.columns.get_loc(feature_x), data.columns.get_loc(feature_y)]
    weights_for_features = weights[feature_idx]
    slope = (-1 * weights_for_features[0]) / weights_for_features[1]
    intercept = (-1 *  bias) / weights_for_features[1]

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

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['Species'] = iris.target
filtered_data = data[(data.index >= 0) & (data.index < 100) & (data['Species'].isin([0, 1]))]

filtered_data.loc[filtered_data['Species'] == 0, 'Species'] = -1
filtered_data.loc[filtered_data['Species'] == 1, 'Species'] = 1

labels = filtered_data['Species']
features = filtered_data[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
features_list = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

weights, bias = perceptron(features, labels)

print("Weights:", weights)
print("Bias:", bias)

# Create a single figure for all pairs of features
num_plots = len(list(combinations(features_list, 2)))
fig, axes = plt.subplots(nrows=num_plots // 2, ncols=2, figsize=(10, 10))

# Flatten the 2D array of subplots for easy indexing
axes = axes.flatten()

# Create separate figures for each pair of features
for i, feature_pair in enumerate(combinations(features_list, 2)):
    plot_line_and_points(weights, bias, filtered_data, feature_pair[0], feature_pair[1], axes[i])

plt.tight_layout()
plt.show()
