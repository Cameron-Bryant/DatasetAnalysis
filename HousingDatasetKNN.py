import matplotlib.pyplot as plt
import csv
import random
import math
from collections import Counter

#get data and store in list
with open("C://Users//camer//Downloads//train (1).csv", newline = '') as f:
    reader = csv.reader(f)
    house_data = list(reader)
with open("C://Users//camer//Downloads//test.csv", newline = '') as f:
    reader = csv.reader(f)
    test_house_data = list(reader)

#Step 1:> Visualize Data and find which columns show the best spread
determining_chars = []
prices = []
#get only the data i need
for i in range(len(house_data)):
    if (i > 0):
        determining_chars.append([int(house_data[i][18]), int(house_data[i][17])])
        prices.append(int(house_data[i][-1]))

test_det_chars = []

for j in range(len(test_house_data)):
    if j > 0:
        test_det_chars.append([int(test_house_data[j][18]), int(test_house_data[j][17])])

def ShowTrueSpread():
    for i in range(len(determining_chars)):
        if prices[i] < 100000:
            plt.plot(determining_chars[i][0], determining_chars[i][1], 'ro')
        elif 100000 < prices[i] < 200000:
            plt.plot(determining_chars[i][0], determining_chars[i][1], 'bo')
        elif 200000 < prices[i] < 300000:
            plt.plot(determining_chars[i][0], determining_chars[i][1], 'go')
        else:
            plt.plot(determining_chars[i][0], determining_chars[i][1], 'ko')
    plt.show()
#lotArea and YearBuilt didn't show a significant link, #lotArea and OverallCond- Maybe
#OverallQual and Overall Cond showed a significant spread, so i will choose that for this one
#because they are 1-10 ratings, the estimates from these two qualities will be rough
#decided to split the housing prices into ranges, and with that use a knn model
#to classify the test data

def getLabels(prices_to_categorize): #label the data as one of 4 categories
    lbls = []
    for i in range(len(prices_to_categorize)):
        if prices_to_categorize[i] < 100000:
            lbls.append(0)
        elif 100000 < prices_to_categorize[i] < 200000:
            lbls.append(1)
        elif 200000 < prices_to_categorize[i] < 300000:
            lbls.append(2)
        elif 300000 < prices_to_categorize[i] < 400000:
            lbls.append(3)
        else:
            lbls.append(4)
    return lbls

p_labels = getLabels(prices)

def plotPoints(points, lbls):
    for i in range(len(points)):
        if lbls[i] == 0:
            plt.plot(points[i][0], points[i][1], 'ro')
        elif lbls[i] == 1:
            plt.plot(points[i][0], points[i][1], 'bo')
        elif lbls[i] == 2:
            plt.plot(points[i][0], points[i][1], 'go')
        elif lbls[i] == 3:
            plt.plot(points[i][0], points[i][1], 'ko')
        elif lbls[i] == 4:
            plt.plot(points[i][0], points[i][1], 'co')
    plt.show(block=False)
    plt.pause(.1)
    plt.cla()

def plotSingle(point, label):
    if label == 0:
        plt.plot(point[0], point[1], 'ro')
    elif label == 1:
        plt.plot(point[0], point[1], 'bo')
    elif label == 2:
        plt.plot(point[0], point[1], 'go')
    elif label == 3:
        plt.plot(point[0], point[1], 'ko')
    elif label == 4:
        plt.plot(point[0], point[1], 'co')

def removeUnseenData(data, lbls):#remove unseen data in the visualization to vis faster
    visualization_data = []
    vis_labels = []
    for i in range(len(data)):
        if data[i] not in visualization_data:
            visualization_data.append(data[i])
            vis_labels.append(lbls[i])
    return visualization_data, vis_labels

def dist(p1, p2):
    return math.sqrt(((p1[0]-p2[0])**2) + ((p1[1] - p2[1])**2))

def sort_by_first(elem):
    return elem[0]

def knn(k, data, unclassified_point, labels):
    dists = []
    #Label and Plot unclassified point among the data
    vis_data, vis_labs = removeUnseenData(data, labels)
    plt.text(unclassified_point[0], unclassified_point[1], 'Unclassified Point')

    if unclassified_point in vis_data:
        vis_labs.remove(vis_labs[vis_data.index(unclassified_point)])
        vis_data.remove(unclassified_point)
        plt.plot(unclassified_point[0], unclassified_point[1], 'mo')
        plotPoints(vis_data, vis_labs)

    for i in range(len(data)):#get dist and index of all points from unclassified one
        dists.append([dist(unclassified_point, data[i]), i])
    sorted_dists = sorted(dists, key=sort_by_first)#sort by dist

    nearest_neighbors = []
    for j in range(k):#pick the first k entries
        nearest_neighbors.append(sorted_dists[j])
    #get the mode of the labels of the nearest neighbors
    nn_lbls = []
    for n in range(len(nearest_neighbors)):
        nn_lbls.append(labels[nearest_neighbors[n][1]])
    #that mode is the label for the unclassified point
    cat = Counter(nn_lbls).most_common(1)[0][0]

    #plot the new point among the data with its new classification
    vis_data.append(unclassified_point)
    vis_labs.append(cat)
    plotSingle(unclassified_point, cat)
    plt.text(unclassified_point[0], unclassified_point[1], 'Classified Point')
    plotPoints(vis_data, vis_labs)
    #add newly classified point to the data to be used
    data.append(unclassified_point)
    labels.append(cat)
    return cat #return the mode

for i in range(len(test_det_chars)):
    knn(10, determining_chars, test_det_chars[i], p_labels)
#when ran it shows visualization of knn. x and y values were overallqual and overallcond from the dataset. These seemed to be the only descriptors that could be predicted 
#accurately using knn. 
