import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


# decision tree
def entropy(data):
    total_samples = len(data)
    classes = data['Species'].value_counts()
    ans=0
    entropy_val = 0
    for cls in classes:
        entropy_val -= (cls / total_samples) * (np.log2(cls / total_samples))

    if entropy_val == 0 and 1 in classes.index:
        ans = 1
    if entropy_val == 0 and 2 in classes.index:
        ans = 2
    return entropy_val, ans

def find_best_attribute(data):
    best_attribute = None
    best_threshold = None
    best_info_gain = 0
    max_info_gain = {}
    best_attri_threshold = {}
    best_split_left = None
    best_split_right = None
    decision = 0  # Initialize decision variable

    total_entropy, ans = entropy(data)
    if total_entropy == 0:
        decision = ans
        return best_attribute, max_info_gain, best_threshold, best_attri_threshold, best_split_left, best_split_right, decision

    for attribute in data.columns[:-1]:
        min_value = data[attribute].min()
        max_value = data[attribute].max()
        max_info_gain[attribute] = 0
        best_attri_threshold[attribute] = 0
        for threshold in np.arange(min_value, max_value, 0.1):
            left_data = data[data[attribute] <= threshold]
            right_data = data[data[attribute] > threshold]

            left_entropy, _ = entropy(left_data)
            right_entropy, _ = entropy(right_data)

            info_gain = total_entropy - (len(left_data) / len(data) * left_entropy) - (
                    len(right_data) / len(data) * right_entropy)

            if info_gain > max_info_gain[attribute]:
                max_info_gain[attribute] = info_gain
                best_attri_threshold[attribute] = threshold
                best_split_left = left_data
                best_split_right = right_data

        if max_info_gain[attribute] > best_info_gain:
            best_info_gain = max_info_gain[attribute]
            best_threshold = best_attri_threshold[attribute]
            best_attribute = attribute

    return best_attribute, max_info_gain, best_threshold, best_attri_threshold, best_split_left, best_split_right, decision


def build_tree(data):
    best_attribute, _, best_threshold, _,_,_, decision = find_best_attribute(data)
    root = {
        'decision': None,
        'decision_attribute': None,
        'best_threshold': None,
        'childs': []
    }

    new_data_right = 0
    new_data_left = 0

    if decision != 0 :
        if decision == 1:
            root['decision_attribute'] = 'I. versicolor'

        else:
            root['decision_attribute'] = 'I. virginica'
        return root
    
    root['decision_attribute'] = best_attribute
    root['best_threshold'] = best_threshold  
    new_data_left = data[data[best_attribute] <= best_threshold]
    new_data_right = data[data[best_attribute] > best_threshold]
    temp = build_tree(new_data_left)
    temp['decision'] = f" <= {best_threshold}"
    root['childs'].append(temp)
    temp = build_tree(new_data_right)
    temp['decision'] = f" > {best_threshold}"
    root['childs'].append(temp)
    return root
def traverse(root, depth=0):
    indent = "  " * depth
    print(indent + str(root['decision']))
    print(indent + str(root['decision_attribute']))

    n = len(root['childs'])
    if n > 0:
        for i in range(0, n):
            traverse(root['childs'][i], depth + 1)
def calculate(data):
    root = build_tree(data)
    root['decision'] = 'Start'
    traverse(root)
    return root
def classify_instance(instance, tree):
    while 'childs' in tree:
        attribute = tree['decision_attribute']
        threshold = tree['best_threshold']
        if attribute == 'I. versicolor' or attribute == 'I. virginica':
              return attribute
        value = instance[attribute]
        if value <= threshold:
            tree = tree['childs'][0]
        else:
            tree = tree['childs'][1]
            
def find_con_matrix(raw_data,data, root):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    columns =   ['sepal length (cm)','sepal width (cm)','petal length (cm)', 'petal width (cm)']
    final_X = [[0, 0], [0, 0]]  # Initialize the confusion matrix

    for i in range(len(data)):
        new_instance = {'sepal length (cm)': 0, 'sepal width (cm)': 0, 'petal length (cm)': 0, 'petal width (cm)': 0}  # Use a dictionary to represent the instance
        for j in columns:  
           new_instance[j] = data.iloc[i][j]
           classification_result = classify_instance(new_instance, root) 
           if raw_data.iloc[i]['Species'] == 1:
               actual_result = 'I. versicolor'
           if raw_data.iloc[i]['Species'] == 2:
               actual_result = 'I. virginica'

        if classification_result == actual_result == 'I. versicolor':
            tp += 1
        elif classification_result == actual_result == 'I. virginica':
            tn += 1
        elif classification_result != actual_result and classification_result == 'I. versicolor':
            fp += 1
        elif classification_result != actual_result and classification_result == 'I. virginica':
            fn += 1

    final_X = [[tp, fn], 
               [fp, tn]]  # Update confusion matrix with counts
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # Calculate precision
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0  # Calculate recall
    return final_X,precision,recall

def findconmat(data, root):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    columns =   ['sepal length (cm)','sepal width (cm)','petal length (cm)', 'petal width (cm)']
    final_X = [[0, 0], [0, 0]]  # Initialize the confusion matrix

    for i in range(len(data)):
        new_instance = {'sepal length (cm)': 0, 'sepal width (cm)': 0, 'petal length (cm)': 0, 'petal width (cm)': 0}  # Use a dictionary to represent the instance
        for j in columns:  
           new_instance[j] = data.loc[filtered_data.index[i], j]
           classification_result = classify_instance(new_instance, root) 
           if data.loc[filtered_data.index[i], 'Species'] == 1:
               actual_result = 'I. versicolor'
           if data.loc[filtered_data.index[i], 'Species'] == 2:
               actual_result = 'I. virginica'

        if classification_result == actual_result == 'I. versicolor':
            tp += 1
        elif classification_result == actual_result == 'I. virginica':
            tn += 1
        elif classification_result != actual_result and classification_result == 'I. versicolor':
            fp += 1
        elif classification_result != actual_result and classification_result == 'I. virginica':
            fn += 1

    final_X = [[tp, fn], 
               [fp, tn]]  # Update confusion matrix with counts
    print(final_X)
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # Calculate precision
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0  # Calculate recall
    print('precision:', precision)
    print('recall:', recall)

# KNN
def euclidean_dist(new_instance, instance):
    distance = 0
    for i in range(len(new_instance)):
        distance += np.sqrt((new_instance[i] - instance[i]) ** 2)
    return distance

def knn(data, new_instance, k):
    distances = []
    for instance in data:
        distance = euclidean_dist(new_instance, instance[:-1])  # Assuming last column is the class label
        distances.append(distance)
    
    # Get indices of k nearest neighbors
    nearest_indices = np.argsort(distances)[:k]
    
    # Retrieve class labels of k nearest neighbors
    classes = [data[i][-1] for i in nearest_indices]
    
    # Choose the most common class among the nearest neighbors
    prediction = max(set(classes), key=classes.count)
    return prediction

def calculate_KNN(train_data, test_data, k):
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    for i in range(len(test_data)):
        new_instance = test_data.iloc[i, :-1]  # Assuming last column is the class label
        prediction = knn(train_data.values, new_instance, k)
        actual_result = test_data.iloc[i, -1]  # Assuming last column is the class label

        if prediction == actual_result == 1:
            tp += 1
        elif prediction == actual_result == 2:
            tn += 1
        elif prediction != actual_result and prediction == 1:
            fp += 1
        elif prediction != actual_result and prediction == 2:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) != 0 else 0  # Calculate precision
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0  # Calculate recall
    print('Confusion Matrix for KNN:')
    print([[tp, fn], [fp, tn]])
    print('Precision:', precision)
    print('Recall:', recall)
    return precision, recall , tp, fn, fp, tn



# data visualization 
def plot_distributions_with_thresholds(data):
    _, max_info_gain, _, best_attri_threshold,best_split_left,best_split_right, _= find_best_attribute(data)
    
    for attribute in data.columns[:-1]:
     plt.figure(figsize=(8, 6))
    
     # Plot distribution lines using KDE for both classes
     sns.kdeplot(best_split_left[attribute], label='Iris-versicolor', color='blue')
     sns.kdeplot(best_split_right[attribute], label='Iris-virginica', color='orange')
     plt.xlabel(attribute)
     plt.ylabel('Density')
     plt.legend()
     plt.title(f'Distribution of {attribute} for both classes')

     # Plot the best threshold as a vertical line on the distribution plot
     plt.axvline(x=best_attri_threshold[attribute], color='r', linestyle='--', label=f'Best Threshold: {best_attri_threshold[attribute]}')
     plt.legend()
     plt.show()
     # Calculate information gain and best threshold for the attribute
     print(f'Best threshold for {attribute}: {best_attri_threshold[attribute]} (Information Gain: {max_info_gain[attribute]})')


#k fold tree
def kfold_decision_tree(data):
    precision_max = 0
    recall_max = 0
    final_X_max = 0
    choice3 = input("how mant fold do you want? ")
    kf = int(choice3)
    # Creating 4 subsets
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    subset_size = len(shuffled_data) // kf
    # Creating 4 subsets
    subsets = []
    start_index = 0
    for i in range(kf - 1):
        subset = shuffled_data.iloc[start_index:start_index + subset_size]
        subsets.append(subset)
        start_index += subset_size
    subsets.append(shuffled_data.iloc[start_index:])

    for j in range(kf):
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        test_data = pd.concat([test_data, subsets[j]], ignore_index=True)
        for i in range(kf):
            if i!=j:
                train_data = pd.concat([train_data, subsets[i]], ignore_index=True)
        root=build_tree(train_data)
        final_X,precision,recall = find_con_matrix(shuffled_data,test_data, root)
        if precision > precision_max and recall > recall_max : 
             precision_max=precision
             recall_max=recall
             final_X_max = final_X   
    return precision_max,recall_max,final_X_max

def kfold_KNN(data, k):
    precision_max = 0
    recall_max = 0
    final_X_max = 0
    choice3 = input("how many folds do you want? ")
    kf = int(choice3)
    
    # Creating k subsets
    shuffled_data = data.sample(frac=1).reset_index(drop=True)
    subset_size = len(shuffled_data) // kf
    
    # Creating k subsets
    subsets = []
    start_index = 0
    for i in range(kf - 1):
        subset = shuffled_data.iloc[start_index:start_index + subset_size]
        subsets.append(subset)
        start_index += subset_size
    subsets.append(shuffled_data.iloc[start_index:])
    
    for j in range(kf):
        maxpre = 0 
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        test_data = pd.concat([test_data, subsets[j]], ignore_index=True)
        for i in range(kf):
            if i != j:
                train_data = pd.concat([train_data, subsets[i]], ignore_index=True)
        
        precision, recall , tp, fn, fp, tn = calculate_KNN(train_data, test_data, k)
        if precision >= maxpre :
            maxpre = precision
            rec2 = recall
            tp2 = tp
            fn2 = fn
            fp2 = fp
            tn2 = tn
    print('*****************************')
    print('max Confusion Matrix for KNN:')
    print([[tp2, fn2], [fp2, tn2]])
    print('max Precision:', maxpre)
    print('Recall:', rec2)       




# Choose the analysis type
def choose_analysis():
    print("Select the analysis type:")
    print("1. Plot data distribution based on output classes and draw thresholds")
    print("2. Build a Decision Tree")
    print("3. Build a KNN model with custom K value")
    print("4. Build a K-fold DT model with custom K value")
    print("5. Build a K-fold KNN model with custom K value")
    choice = input("Enter your choice (1, 2, 3, 4 or 5): ")

    return int(choice)
def perform_chosen_analysis(choice, data):
    if choice == 1:
        plot_distributions_with_thresholds(data)
    elif choice == 2:
        root = build_tree(data)
        findconmat(data, root)
        choice2 = input("Do you want to print the tree (1 for yes , 2 for no): ")
        if int(choice2) == 1 :
            root['decision'] = 'Start'
            traverse(root)
        elif int(choice2) == 2 :
            print( "baba maa darim zahmat mikeshim :( ")
        else : 
            print ("maskhare kardi maroo ?   .   :[")
    elif choice == 3:
        k = int(input("Enter the value of K for KNN: "))
        calculate_KNN(data,data, k)
    elif choice == 4:
        precision_max,recall_max,final_X_max = kfold_decision_tree(data)
        print("***************")
        print('confusion matrix :', final_X_max)
        print('precision:', precision_max)
        print('recall:', recall_max)
    elif choice == 5:
        k = int(input("Enter the value of K for KNN: "))
        kfold_KNN(data, k)

    else:
        print("invalide baba")


# Main function
def main():
    choice = choose_analysis()
    perform_chosen_analysis(choice, filtered_data)

iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['Species'] = iris.target
filtered_data = data[(data.index >= 50) & (data.index < 150) & (data['Species'].isin([1, 2]))]
# Run the main function
main()




