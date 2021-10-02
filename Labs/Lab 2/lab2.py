from scipy.stats import chisquare

data = [[592, 300, 204, 24, 80], [410, 335, 180, 20, 55]]
newdata = [[0 for j in i] for i in data]

# sum row x sum i / total
total = 0
for i in data: 
    for j in i:
        total += j

for i in range(len(data)):
    for j in range(len(data[i])):
        rowsum = sum(data[i])
        colsum = data[0][j] + data[1][j]

        # print(rowsum, colsum, total)

        newdata[i][j] = rowsum * colsum / total

print(newdata)

print(chisquare(data))