import pandas as pd
from sklearn.model_selection import train_test_split
from pprint import pprint

df = pd.read_csv('iris.csv')
train, test = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)

y = train['class']
x = train.drop(columns=['class'])
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

# count up class frequencies
for i in x:
    count = 0

    if i not in category_accuracies:
        category_accuracies[i] = {}

    for value in x[i]:
        for category in categories[i]:
            if category[0] <= value <= category[1]:
                if category not in category_accuracies[i]:
                    category_accuracies[i][category] = {}
                    category_accuracies[i][category]["final"] = None
                    category_accuracies[i][category]["count"] = {"Iris-setosa": 0, "Iris-versicolor": 0, "Iris-virginica": 0}
                
                category_accuracies[i][category]["count"][y.iloc[count]] += 1
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

test_y = test['class']
test_x = test.drop(columns=['class'])

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

for i in range(len(test_x)):
    # print(test_x.iloc[i]["petalwidth"],)
    best_attr_value = test_x.iloc[i][attribute_order[0]]

    if best_attr_value < categories[attribute_order[0]][0][0]:
        idx = 0
    elif best_attr_value > categories[attribute_order[0]][-1][1]:
        idx = -1
    else:
        count1 = 0

        for category in categories[attribute_order[0]]:
            
            if category[0] <= best_attr_value <= category[1]:
                # print(category)
                # print(best_attr_value)
                idx = count1
                break
            count1 += 1

    predicted = category_accuracies[attribute_order[0]][categories[attribute_order[0]][idx]]["final"][1]
    actual = test_y.iloc[i]

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