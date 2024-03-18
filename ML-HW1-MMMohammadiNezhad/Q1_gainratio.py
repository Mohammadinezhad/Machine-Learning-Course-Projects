import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import copy

dataset = pd.read_csv(r'C:\Users\mhd76\Desktop\master\courses\M L\projects\t 1\data 1.csv')
X = dataset.iloc[:, 1:].values
#print(X)
attribute = ['Colour','Toughness','Single-branch','Appearance']

def findEntropy(data, rows):
    yes = 0
    no = 0
    ans = -1
    idx = len(data[0]) - 1
    entropy = 0
    for i in rows:
        if data[i][idx] == 'Yes':
            yes = yes + 1
        else:
            no = no + 1

    x = yes/(yes+no)
    y = no/(yes+no)
    if x != 0 and y != 0:
        entropy = -1 * (x*math.log2(x) + y*math.log2(y))
    if x == 1:
        ans = 1
    if y == 1:
        ans = 0
    return entropy, ans

def findMaxGainRatio(data, rows, columns):
    maxGainRatio = 0
    retidx = -1
    entropy, ans = findEntropy(data, rows)
    if entropy == 0:
        return maxGainRatio, retidx, ans

    for j in columns:
        mydict = {}
        idx = j
        for i in rows:
            key = data[i][idx]
            if key not in mydict:
                mydict[key] = 1
            else:
                mydict[key] = mydict[key] + 1

        information_gain = entropy
        split_information = 0

        for key in mydict:
            yes = 0
            no = 0
            for k in rows:
                if data[k][j] == key:
                    if data[k][-1] == 'Yes':
                        yes = yes + 1
                    else:
                        no = no + 1
            x = yes / (yes + no)
            y = no / (yes + no)

            if x != 0 and y != 0:
                information_gain += (mydict[key] * (x * math.log2(x) + y * math.log2(y))) / 7
                split_information += -1*(mydict[key] / 7) * math.log2(mydict[key] / 7)

        if split_information != 0:
            gain_ratio = information_gain / split_information
        else:
            gain_ratio = 0

        if gain_ratio > maxGainRatio:
            maxGainRatio = gain_ratio
            retidx = j

    return maxGainRatio, retidx, ans


def buildTree(data, rows, columns):
    maxGainRatio, idx, ans = findMaxGainRatio(X, rows, columns)
    root = {
        'decision': None,
        'decision_condition': None,
        'childs': []
    }
    
    if maxGainRatio == 0:
        if ans == 1:
            root['decision_condition'] = 'Yes'
        else:
            root['decision_condition'] = 'No'
        return root

    root['decision_condition'] = attribute[idx]
    mydict = {}
    for i in rows:
        key = data[i][idx]
        if key not in mydict:
            mydict[key] = 1
        else:
            mydict[key] += 1

    newcolumns = copy.deepcopy(columns)
    newcolumns.remove(idx)
    for key in mydict:
        newrows = [i for i in rows if data[i][idx] == key]
        temp = buildTree(data, newrows, newcolumns)
        temp['decision'] = key
        root['childs'].append(temp)
    return root

def traverse(root, depth=0):
    indent = "  " * depth
    print(indent + str(root['decision']))
    print(indent + str(root['decision_condition']))

    n = len(root['childs'])
    if n > 0:
        for i in range(0, n):
            traverse(root['childs'][i], depth + 1)

def classify(instance, root):
    while root['childs']:
        attribute_idx = attribute.index(root['decision_condition'])
        value = instance[attribute_idx]
        child = next((child for child in root['childs'] if child['decision'] == value), None)
        if child:
            root = child
        else:
            break
    return root['decision_condition']

def calculate():
    rows = [i for i in range(0, 7)]
    columns = [i for i in range(0, 4)]
    root = buildTree(X, rows, columns)
    root['decision'] = 'Start'
    traverse(root)
    return root

root = calculate()
print( root)

# Example classification for a new instance
new_instance = ['Brown', 'soft', 'No', 'Wrinkled']
classification_result = classify(new_instance, root)
print(f"The new instance is classified as: {classification_result}")

