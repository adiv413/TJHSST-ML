import numpy as np
import csv

class DecisionTree:

	def __init__(self, training_data): 
		self.training_data = training_data
		self.decisionList = []

	def calculateImpurity(self, li):
		count0 = 0
		count1 = 0

		for i in li:
			if i == 0:
				count0 += 1
			else:
				count1 += 1

		return 1 - (((count0 / len(li)) ** 2) + ((count1 / len(li)) ** 2))

	def splitList(self, dataset, feature, threshold):
		total = []
		left = []
		right = []

		for i in dataset:
			if i[feature] < threshold:
				left.append(i)
			else:
				right.append(i)

		total.append(left)	
		total.append(right)

		return total

	def isPerfect(self, dataset):
		x = True
		y = dataset[0][len(dataset[0]) - 1]
		for i in dataset:
			if not (i[len(dataset[0]) - 1] == y):
				x = False
		return x

	def calculateIG(self, dataset):

		if not dataset[0] or not dataset[1]:
			return 0

		left = [i[len(dataset[0][0]) - 1] for i in dataset[0]]
		right = [i[len(dataset[0][0]) - 1] for i in dataset[1]]

		total = list(map(float, (' '.join(map(str, left)) + ' ' + ' '.join(map(str, right))).split(' ')))

		return self.calculateImpurity(total) - len(left) / len(total) * self.calculateImpurity(left) - len(right) / len(total) * self.calculateImpurity(right)


	def getBestSplit(self, dataset, depth):
		ig = 0
		bestFeature = -1
		bestThreshold = -1

		total = []
		quartiles = [25, 50, 75]

		for i in range(len(dataset[0]) - 1):
			featureList = [x[i] for x in dataset]
			thresholds = np.percentile(np.array(featureList), quartiles).tolist()

			if thresholds[0] == thresholds[1] == thresholds[2] == -1:
				continue

			for j in thresholds:
				temp_total = self.splitList(dataset, i, j)
				temp_ig = self.calculateIG(temp_total)

				if temp_ig > ig: 
					bestFeature = i
					bestThreshold = j
					total = temp_total
					ig = temp_ig
		
		if len(total) == 0:
			return

		left = [i[len(total[0][0]) - 1] for i in total[0]]
		right = [i[len(total[0][0]) - 1] for i in total[1]]

		temp = []

		temp.append(bestFeature)
		temp.append(bestThreshold)
		temp.append(self.zero_or_one(right))
		temp.append(ig)
		temp.append((len(left) + len(right)) / 2)

		self.decisionList.append(temp) 

		if not self.isPerfect(total[0]) and left != [] and depth < 4:
			self.getBestSplit(total[0], depth + 1)
		if not self.isPerfect(total[1]) and right != [] and depth < 4:
			self.getBestSplit(total[1], depth + 1)

	def zero_or_one(self, data):
		total = 0
		x = len(data)
		for i in data:
			if i == 1:
				total += 1
		y = total / x
		if total > 0.5:
			return 1
		return 0

	def sort_by_IG(self, dataset):
  		
		for i in range(len(dataset)): 
		    x = i 

		    for j in range(i + 1, len(dataset)): 
		        if dataset[x][3] < dataset[j][3]: 
		            x = j 
		                   
		    dataset[i], dataset[x] = dataset[x], dataset[i] 

		temp = []

		for i in dataset:
			if i[3] > 0.01: #0.06 and i[4] > len(self.training_data) / 30 or i[3] > 0.04 and i[4] > len(self.training_data) / 8:
				temp.append(i)

		return temp



	def fit(self, dataset):
		yVals = []
		self.decisionList = self.sort_by_IG(self.decisionList)
		
		yVals = []
		found = False

		
		for i in dataset:
			for j in self.decisionList:
				if i[j[0]] > j[1]:
					yVals.append(j[2]) 
					found = True
					break
			if not found:
				yVals.append(0)

			found = False

		return yVals

	def getAccuracy(self, dataset, fittedData):
		total = 0
		x = len(dataset)

		if x != len(fittedData):
			return -1

		for i in range(x):
			if dataset[i][len(dataset[0]) - 1] == fittedData[i]:
				total += 1
		return total / x

	def train(self):
		self.getBestSplit(self.training_data, 0)


f = open("iris.csv", "r")
reader = csv.reader(f)
class_encoder = {
    "Iris-setosa" : 1,
    "Iris-versicolor" : 2,
    "Iris-virginica" : 3
}
lists_from_csv = []
count = 0
for row in reader:
    # skip the column names
    if count == 0:
        count += 1
        continue
    row = [int(float(i) * 100) if not i[0].isalpha() else class_encoder[i] for i in row]
    lists_from_csv.append(row)

dt = DecisionTree(lists_from_csv[:int(len(lists_from_csv) * .8)])

dt.train()
output = dt.fit([lists_from_csv[int(len(lists_from_csv) * .8):]])
print(output)
print(dt.getAccuracy(lists_from_csv[int(len(lists_from_csv) * .8):], output))