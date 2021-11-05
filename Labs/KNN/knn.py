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

our_model = []
k_vals = list(range(1,11))

for k_val in range(1, 11):
    k = k_val

    x_train, x_test, y_train, y_test = train_test_split(x_full, y_full, test_size=0.2, random_state=5, stratify=y_full)
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    num_correct = 0
    total_num = 0

    # Create K-NearestNeighbors Classifier to classify all values in x_test
    for idx in range(len(x_test)):
        # print(idx)
        row = x_test.iloc[idx]
        distances = []
        for idx2 in range(len(x_train)):
            instance = x_train.iloc[idx2]
            distance = 0

            for attr_idx in range(len(instance)):
                distance += math.pow(row[attr_idx] - instance[attr_idx], 2)

            distances.append((math.sqrt(distance), y_train.iloc[idx2]))
            # print(distances)
        
        distances = sorted(distances)
        predicted = max(set([i[1] for i in distances[:k]]), key=distances.count)
        # print(predicted)

        actual = y_test.iloc[idx]

        if predicted == actual:
            num_correct += 1

        total_num += 1
    our_model.append(num_correct/total_num * 100)
    print("Accuracy with", k, "neighbors:", str(round(num_correct / total_num * 100)) + "%")

plt.plot(k_vals, our_model, label="Our Implementation")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("K vs. Accuracy")
# classify x_train using sklearn KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

sk_acc = []
for k_val in range(1, 11):
    k = k_val
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)

    # print the accuracy of the classifier
    print("SKLearn Accuracy with", k, "neighbors: ", knn.score(x_test, y_test) * 100, "%")
    sk_acc.append(knn.score(x_test, y_test) * 100)

plt.plot(k_vals, sk_acc, label="Scikit-Learn Implementation")
plt.show()