# Aditya Vasantharao, Pd. 6
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from pprint import pprint
import math
import matplotlib.pyplot as plt
df = pd.read_csv('iris.csv')
x_full = df.drop(columns=["class"])
y_full = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=38, stratify=y_full)
pd.set_option("display.max_rows", None, "display.max_columns", None)


# discretize the dataset

# { feature : [category list]}
categories = {}

for i in x_train:
    splits = pd.qcut(x_train[i], q=3)

    if i not in categories:
        categories[i] = []

    for split in splits:
        tup = (split.left, split.right - 1e-12)

        if tup not in categories[i]:
            categories[i].append(tup)

# {class : count}
class_counts = {}

# {class : probability}
class_probabilities = {}

for i in y_train:
    if i not in class_counts:
        class_counts[i] = 0
    class_counts[i] += 1

class_total = len(y_train)

for i in class_counts:
    class_count = class_counts[i]
    class_probabilities[i] = class_count / class_total

# {class value : {feature : {category : probability}}}
probabilities = {}


# check probability of row given each class, find the greatest probability
for output_class in class_probabilities:
    if output_class not in probabilities:
        probabilities[output_class] = {}

    for feature in x_train:
        if feature not in probabilities[output_class]:
            probabilities[output_class][feature] = {}

        number_of_y = 0
        for k in range(len(x_train)):
            if(y_train.iloc[k] == output_class):
                number_of_y += 1

        for category in categories[feature]:
            number_of_x = 0
            for k in range(len(x_train)):
                if(category[0] <= x_train.iloc[k][feature] <= category[1]) and (y_train.iloc[k] == output_class):
                    number_of_x += 1
                    
            probabilities[output_class][feature][category] = number_of_x / number_of_y
            
# fit data using Naive Bayes

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
    max_probability = None
    best_class = None

    # check probability of row given each class, find the greatest probability
    for output_class in class_probabilities:
        prob_x_given_class = 1

        row = x_test.iloc[i]
        
        for idx in range(len(row.index)):
            col = row.index[idx]
            val = row[idx]

            potential_categories = categories[col]

            min_cat = min(potential_categories)
            max_cat = max(potential_categories)
            actual_cat = None

            if val <= min_cat[0]:
                actual_cat = min_cat
            elif val >= max_cat[1]:
                actual_cat = max_cat
            else:
                for category in potential_categories:
                    if category[0] <= val <= category[1]:
                        actual_cat = category
                        break

            prob_x_given_class *= probabilities[output_class][col][actual_cat]

        prob_class_given_x = prob_x_given_class * class_probabilities[output_class]
        
        if (max_probability is None) or (prob_class_given_x > max_probability):
            max_probability = prob_class_given_x
            best_class = output_class

    predicted = best_class
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


# Test using sklearn's MultinomialNB

model = MultinomialNB()
model.fit(x_train, y_train)
predicted = model.predict(x_test)
print("Accuracy", metrics.accuracy_score(y_test, predicted))
sklearn_confusion_matrix = metrics.confusion_matrix(y_test, predicted)

disp = metrics.ConfusionMatrixDisplay(confusion_matrix=sklearn_confusion_matrix, display_labels=["Iris-setosa", "Iris-versicolor", "Iris-virginica"])
disp = disp.plot()

plt.show()