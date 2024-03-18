import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
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

# Create a Decision Tree classifier with information gain (entropy) as criterion
dt_classifier = DecisionTreeClassifier(criterion='entropy', random_state=42)

# Train the classifier on the training data
dt_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = dt_classifier.predict(X_test)

# Evaluate the performance of the classifier
accuracy = accuracy_score(y_test, y_pred)
classification_report_str = classification_report(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report_str)
