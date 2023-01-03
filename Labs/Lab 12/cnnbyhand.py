from pprint import pprint

image = [[-1, -1, -1, -1, -1, -1, -1, -1, -1], 
         [-1, 1, -1, -1, -1, -1, -1, 1, -1], 
         [-1, -1, 1, -1, -1, -1, 1, -1, -1], 
         [-1, -1, -1, 1, -1, 1, -1, -1, -1], 
         [-1, -1, -1, -1, 1, -1, -1, -1, -1], 
         [-1, -1, -1, 1, -1, 1, -1, -1, -1], 
         [-1, -1, 1, -1, -1, -1, 1, -1, -1], 
         [-1, 1, -1, -1, -1, -1, -1, 1, -1], 
         [-1, -1, -1, -1, -1, -1, -1, -1, -1]]
filter = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
filter2 = [[1, -1, 1], [-1, 1, -1], [1, -1, 1]]
filter3 = [[-1, -1, 1], [-1, 1, -1], [1, -1, -1]]


# create method to calculate filter convolution on image
def convolve(image, filter):
    # create empty list to store the convolution
    convolved = []
    # loop through each row in the image
    for i in range(len(image) - 2):
        # create empty list to store the convolution of each row
        convolved_row = []
        # loop through each column in the image
        for j in range(len(image[0]) - 2):
            # create empty list to store the convolution of each column
            convolved_column = []
            # loop through each row in the filter
            for k in range(len(filter)):
                # loop through each column in the filter
                for l in range(len(filter[0])):
                    # if the filter is out of bounds, set the convolution to 0
                    if i + k < len(image) and j + l < len(image[0]):
                        # add the convolution of each pixel to the convolved list
                        convolved_column.append(image[i + k][j + l] * filter[k][l])
                    else:
                        convolved_column.append(0)
            # add the convolved column to the convolved row
            convolved_row.append(sum(convolved_column))
        # add the convolved row to the convolved list
        convolved.append(convolved_row)
    # return the convolved list
    return convolved

out = convolve(image, filter)
out2 = convolve(image, filter2)
out3 = convolve(image, filter3)

# divide all values in out by 9
for i in range(len(out)):
    for j in range(len(out[0])):
        out[i][j] = round(out[i][j] / 9, 2)

for i in range(len(out)):
    for j in range(len(out[0])):
        out2[i][j] = round(out2[i][j] / 9, 2)

for i in range(len(out)):
    for j in range(len(out[0])):
        out3[i][j] = round(out3[i][j] / 9, 2)

relu1 = []
for i in range(len(out)):
    a = []
    for j  in range(len(out[0])):
        if out[i][j] > 0:
            a.append(out[i][j])
        else:
            a.append(0)
    relu1.append(a)

relu2 = []
for i in range(len(out)):
    a = []
    for j  in range(len(out[0])):
        if out2[i][j] > 0:
            a.append(out2[i][j])
        else:
            a.append(0)
    relu2.append(a)

relu3 = []
for i in range(len(out)):
    a = []
    for j  in range(len(out[0])):
        if out3[i][j] > 0:
            a.append(out3[i][j])
        else:
            a.append(0)
    relu3.append(a)

# pad relu1 with zeros
relu1_padded = []
for i in range(len(relu1)):
    a = []
    for j in range(len(relu1[0])):
        if j == len(relu1[0])-1:
            a.append(relu1[i][j])
            a.append(0)
        else:
            a.append(relu1[i][j])
    relu1_padded.append(a)
relu1_padded.append([0]*8)

# pad relu2 with zeros
relu2_padded = []
for i in range(len(relu2)):
    a = []
    for j in range(len(relu2[0])):
        if j == len(relu2[0])-1:
            a.append(relu2[i][j])
            a.append(0)
        else:
            a.append(relu2[i][j])
    relu2_padded.append(a)
relu2_padded.append([0]*8)

# pad relu3 with zeros
relu3_padded = []
for i in range(len(relu3)):
    a = []
    for j in range(len(relu3[0])):
        if j == len(relu3[0])-1:
            a.append(relu3[i][j])
            a.append(0)
        else:
            a.append(relu3[i][j])
    relu3_padded.append(a)
relu3_padded.append([0]*8)

# run 2x2 maxpooling on relu1_padded
maxpool1 = []
for i in range(0, len(relu1_padded), 2):
    a = []
    for j in range(0, len(relu1_padded[0]), 2):
        # if j == len(relu1_padded[0])-1:
        #     a.append(0)
        # else:
        if i+1 < len(relu1_padded) and j+1 < len(relu1_padded[0]):
            a.append(max(relu1_padded[i][j], relu1_padded[i][j + 1], relu1_padded[i+1][j], relu1_padded[i+1][j + 1]))
    maxpool1.append(a)

# run 2x2 maxpooling on relu1_padded
maxpool2 = []
for i in range(0, len(relu2_padded), 2):
    a = []
    for j in range(0, len(relu2_padded[0]), 2):
        # if j == len(relu1_padded[0])-1:
        #     a.append(0)
        # else:
        if i+1 < len(relu2_padded) and j+1 < len(relu2_padded[0]):
            a.append(max(relu2_padded[i][j], relu2_padded[i][j + 1], relu2_padded[i+1][j], relu2_padded[i+1][j + 1]))
    maxpool2.append(a)

# run 2x2 maxpooling on relu1_padded
maxpool3 = []
for i in range(0, len(relu3_padded), 2):
    a = []
    for j in range(0, len(relu3_padded[0]), 2):
        # if j == len(relu1_padded[0])-1:
        #     a.append(0)
        # else:
        if i+1 < len(relu3_padded) and j+1 < len(relu3_padded[0]):
            a.append(max(relu3_padded[i][j], relu3_padded[i][j + 1], relu3_padded[i+1][j], relu3_padded[i+1][j + 1]))
    maxpool3.append(a)

max1 = []
for i in range(0, len(maxpool1), 2):
    a = []
    for j in range(0, len(maxpool1[0]), 2):
        if i+1 < len(maxpool1) and j+1 < len(maxpool1[0]):
            a.append(max(maxpool1[i][j], maxpool1[i][j + 1], maxpool1[i+1][j], maxpool1[i+1][j + 1]))
    max1.append(a)

max2 = []
for i in range(0, len(maxpool2), 2):
    a = []
    for j in range(0, len(maxpool2[0]), 2):
        if i+1 < len(maxpool2) and j+1 < len(maxpool2[0]):
            a.append(max(maxpool2[i][j], maxpool2[i][j + 1], maxpool2[i+1][j], maxpool2[i+1][j + 1]))
    max2.append(a)

max3 = []
for i in range(0, len(maxpool3), 2):
    a = []
    for j in range(0, len(maxpool3[0]), 2):
        if i+1 < len(maxpool3) and j+1 < len(maxpool3[0]):
            a.append(max(maxpool3[i][j], maxpool3[i][j + 1], maxpool3[i+1][j], maxpool3[i+1][j + 1]))
    max3.append(a)

# flatten max1, max2, and max3 then concatenate flattened vectors
max1_flat = []
for i in range(len(max1)):
    for j in range(len(max1[0])):
        max1_flat.append(max1[i][j])
max2_flat = []
for i in range(len(max2)):
    for j in range(len(max2[0])):
        max2_flat.append(max2[i][j])
max3_flat = []
for i in range(len(max3)):
    for j in range(len(max3[0])):
        max3_flat.append(max3[i][j])
# combine max1_flat, max2_flat, and max3_flat into a single vector
max_flat = []
for i in range(len(max1_flat)):
    max_flat.append(max1_flat[i])
for i in range(len(max2_flat)):
    max_flat.append(max2_flat[i])
for i in range(len(max3_flat)):
    max_flat.append(max3_flat[i])
print(max_flat)