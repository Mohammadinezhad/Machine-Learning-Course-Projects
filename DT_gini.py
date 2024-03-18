import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load the dataset
dataset = pd.read_csv(r'C:\Users\mhd76\Desktop\master\courses\M L\projects\project\data\dataR2.csv')

# Extract the features and target variable
columns_to_keep = ['Glucose', 'Resistin', 'BMI', 'HOMA']

# Extract only the specified columns
X = dataset[columns_to_keep]
y = dataset['Classification']

# Standardize the data set
scaler_standard = StandardScaler()
X_standardized = scaler_standard.fit_transform(X)

# Normalize the standardized features
scaler_minmax = MinMaxScaler()
X_normalized = scaler_minmax.fit_transform(X_standardized)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

# Create a Decision Tree classifier with Gini impurity as criterion
dt_classifier = DecisionTreeClassifier(criterion='gini', random_state=42)

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report_str)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, 
            xticklabels=['1', '2'], yticklabels=['1', '2'])
plt.title('Confusion Matrix DT_gini')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
