# Aditya Vasantharao, Pd. 6
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from pprint import pprint
import math
import matplotlib.pyplot as plt
df = pd.read_csv('iris.csv')
x_full = df.drop(columns=["class"])
y_full = df["class"]
x_train_full, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=38, stratify=y_full)
pd.set_option("display.max_rows", None, "display.max_columns", None)
def get_entropy(count, total):
    return (count / total) * math.log2(count / total)

class_entropy = 0
classes = {}

for i in y_train:
    if i not in classes:
        classes[i] = 0
    classes[i] += 1
    
for i in classes:
    count = classes[i]
    class_entropy += get_entropy(count, len(y_train))

class_entropy *= -1
curr_entropy = class_entropy

def generate_tree(x_train, curr_entropy): 
    categories = {}

    for i in x_train:
        splits = pd.cut(x_train[i], 3)

        if i not in categories:
            categories[i] = []

        for split in splits:
            tup = (split.left, split.right)

            if tup not in categories[i]:
                categories[i].append(tup)
            
    info_gains = []
    count = 0

    # {feature : {attribute : next feature to split : {...} OR output_class}}
    tree = {}
    
    # {feature : {category : {entropy}}}
    feature_value_entropies = {}

    # 1. get best feature (there's a better way to do this using selectors like df[df[col_name] == ___]) but I realized this too late 
    # and I don't want to change my whole algorithm now)

    for feature in x_train:
        # { feature_value : {class1, class2, class3} }
        classes_per_feature_val = {}
        feature_value_entropies[feature] = {}

        # { feature_value : 0 }
        feature_value_counts = {}

        for cat in categories[feature]:
            classes_per_feature_val[cat] = {}
            feature_value_counts[cat] = 0

        for row_idx in range(len(x_train[feature])):
            row_class = y_train.iloc[row_idx] # gets class value for specified row
            row_value = x_train[feature].iloc[row_idx]
            feature_value = None

            # find the category that this row is in
            for cat in categories[feature]:
                if cat[0] <= row_value <= cat[1]:
                    feature_value = cat
                    break

            feature_value_counts[feature_value] += 1
            
            if row_class not in classes_per_feature_val[feature_value]:
                classes_per_feature_val[feature_value][row_class] = 0
            
            classes_per_feature_val[feature_value][row_class] += 1

        feature_entropy = 0

        for feature_value in classes_per_feature_val:
            # (number of feature values / total num of values in this feature) * entropy of feature value vector
            feature_value_entropy = 0
            
            for feature_value_class in classes_per_feature_val[feature_value]:
                num_class = classes_per_feature_val[feature_value][feature_value_class]
                total_class = feature_value_counts[feature_value]

                feature_value_entropy += get_entropy(num_class, total_class)

                feature_value_entropies[feature][feature_value] = -feature_value_entropy

            feature_value_entropy *= -1
            feature_entropy += (feature_value_counts[feature_value] / len(x_train[feature])) * feature_value_entropy
        
        info_gains.append((curr_entropy - feature_entropy, feature))

    # best feature has highest info gain
    best_feature = max(info_gains)[1]
    tree = {best_feature : {}}

    # 2. split on feature

    best_feature_values = categories[best_feature]

    # find the number of each class for each feature value in the feature
    classes_per_feature_val = {}

    # { feature_value : count }
    feature_value_counts = {}

    for cat in categories[best_feature]:
        classes_per_feature_val[cat] = {}
        feature_value_counts[cat] = 0

    for row_idx in range(len(x_train[best_feature])):
        row_class = y_train.iloc[row_idx] # gets class value for specified row
        row_value = x_train[best_feature].iloc[row_idx]
        feature_value = None

        # find the category that this row is in
        for cat in categories[best_feature]:
            if cat[0] <= row_value <= cat[1]:
                feature_value = cat
                break

        feature_value_counts[feature_value] += 1
        
        if row_class not in classes_per_feature_val[feature_value]:
            classes_per_feature_val[feature_value][row_class] = 0
        
        classes_per_feature_val[feature_value][row_class] += 1

    for feature_value in best_feature_values:
        new_dataset = pd.DataFrame(columns=x_train.columns)
        for idx in range(len(x_train)):
            obj = x_train.iloc[idx]
            if feature_value[0] <= obj[best_feature] <= feature_value[1]:
                new_dataset = new_dataset.append(obj, ignore_index=True)

        new_dataset = x_train[(x_train[best_feature] >= feature_value[0]) & (x_train[best_feature] <= feature_value[1])]

        if feature_value not in feature_value_entropies[best_feature]:
            continue # rare issue where one quantile does not get populated, so you can just throw it away

        new_curr_entropy = feature_value_entropies[best_feature][feature_value]
        
        if len(x_train.columns) == 1:
            tree[best_feature][feature_value] = list(classes_per_feature_val[feature_value].keys())[0]
        elif len(classes_per_feature_val[feature_value]) == 1: # leaf node
            tree[best_feature][feature_value] = list(classes_per_feature_val[feature_value].keys())[0]
        else: # not a leaf node, split further
            tree[best_feature][feature_value] = generate_tree(new_dataset, new_curr_entropy)
    return tree


tree = generate_tree(x_train_full, class_entropy)

num_correct = 0
total_num = 0

# {predicted : {actual : {0, 0, 0}}}
confusion_matrix = {
    "Iris-setosa": {
        "Iris-setosa": 0, 
        "Iris-versicolor": 0, 
        "Iris-virginica": 0
    }, 
    "Iris-versicolor": {
        "Iris-setosa": 0, 
        "Iris-versicolor": 0, 
        "Iris-virginica": 0
    }, 
    "Iris-virginica": {
        "Iris-setosa": 0, 
        "Iris-versicolor": 0, 
        "Iris-virginica": 0
    }
}

for i in range(len(x_test)):
    row = x_test.iloc[i]
    # print(row)
    actual = y_test.iloc[i]
    root_key = list(tree.keys())[0] # guaranteed to only have one node in root
    root = tree[root_key]
    predicted = None

    while True:
        curr_val = row[root_key]
        min_cat = min(root)
        max_cat = max(root)
        actual_cat = None

        if curr_val <= min_cat[0]:
            actual_cat = min_cat
        elif curr_val >= max_cat[1]:
            actual_cat = max_cat
        else:
            for category in root:
                if category[0] <= curr_val <= category[1]:
                    actual_cat = category
                    break

            if actual_cat is None:
                actual_cat = category
            
        next_node = root[actual_cat]
        if isinstance(next_node, str): # we've reached a leaf node in the tree
            predicted = next_node
            break
        else:
            root_key = list(next_node.keys())[0]
            root = next_node[root_key]

    if predicted == actual:
        num_correct += 1

    total_num += 1

    confusion_matrix[predicted][actual] += 1

print("Accuracy:", str(round(num_correct / total_num * 100)) + "%")
print()
print("Confusion matrix:")
print("predicted\t\tactual")

print("\t\tIris-setosa Iris-versicolor Iris-virginica")

for i in confusion_matrix:
    if i != "Iris-versicolor":
        print(i, "\t", end="")
    else:
        print(i, end=" ")
    for j in confusion_matrix[i]:
        print(confusion_matrix[i][j], end="\t\t")
    print()

# Test using sklearn's DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(x_train_full, y_train)
predicted = model.predict(x_test)

print("Accuracy", metrics.accuracy_score(y_test, predicted))
sklearn_confusion_matrix = metrics.confusion_matrix(y_test, predicted)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=sklearn_confusion_matrix, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

disp = disp.plot()

plt.show()