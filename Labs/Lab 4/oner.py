# Aditya Vasantharao, Pd. 6
import pandas as pd
from sklearn.model_selection import train_test_split
from pprint import pprint
df = pd.read_csv('iris.csv')
x_full = df.drop(columns=["class"])
y_full = df["class"]
x, x_test, y, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=95, stratify=y_full)
pd.set_option("display.max_rows", None, "display.max_columns", None)

# {attribute : [split, in order]}
categories = {}
model = {}

for i in x:
    splits = pd.qcut(x[i], q=4)

    if i not in categories:
        categories[i] = []

    for split in splits:
        tup = (split.left, split.right)

        if tup not in categories[i]:
            categories[i].append(tup)

# {attribute : {category_tup : { "final": (error, class), "count" : {setosa: 0, virginica: 0, other: 0} } } }
category_accuracies = {}
category_values = {}
# count up class frequencies
for i in x:
    count = 0

    if i not in category_accuracies:
        category_accuracies[i] = {}
        category_values[i] = {}

    for value in x[i]:
        for category in categories[i]:
            if category[0] <= value <= category[1]:
                if category not in category_accuracies[i]:
                    category_accuracies[i][category] = {}
                    category_accuracies[i][category]["final"] = None
                    category_accuracies[i][category]["count"] = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}
                    category_values[i][category] = []
                
                category_accuracies[i][category]["count"][y.iloc[count]] += 1
                category_values[i][category].append((value, y.iloc[count]))
        count += 1
        
# find the final class per category and error rate
error_rates = []
for i in x:
    for category in categories[i]:
        final_class_tup = max([(category_accuracies[i][category]["count"][j], j) for j in category_accuracies[i][category]["count"]])
        final_class = final_class_tup[1]
        final_class_freq = final_class_tup[0]
        class_count = sum(category_accuracies[i][category]["count"].values())
        category_accuracies[i][category]["final"] = ((class_count - final_class_freq) / class_count, final_class)

    error_rates.append((sum([category_accuracies[i][category]["final"][0] for category in categories[i]]), i))

attribute_order = [i[1] for i in sorted(error_rates)]

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
    best_attr_value = x_test.iloc[i][attribute_order[0]]

    min_cat = None
    min_cat_idx = 0
    max_cat = None
    max_cat_idx = 0

    for c in range(len(categories[attribute_order[0]])):
        if min_cat is None or categories[attribute_order[0]][c][0] < min_cat[0]:
            min_cat = categories[attribute_order[0]][c]
            min_cat_idx = c

        if max_cat is None or categories[attribute_order[0]][c][1] > max_cat[1]:
            max_cat = categories[attribute_order[0]][c]
            max_cat_idx = c

    if best_attr_value < min_cat[0]:
        idx = min_cat_idx
    elif best_attr_value > max_cat[1]:
        idx = max_cat_idx
    else:
        count1 = 0

        for category in categories[attribute_order[0]]:
            
            if category[0] <= best_attr_value <= category[1]:
                idx = count1
                break
            count1 += 1
            
    predicted = category_accuracies[attribute_order[0]][categories[attribute_order[0]][idx]]["final"][1]

    actual = y_test.iloc[i]

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