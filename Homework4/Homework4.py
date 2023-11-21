#---Homework 4---
import numpy as np
import sigmoid as sigmoid
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from libsvm.svmutil import *


file = "iris.txt"
train_file = "train_data.txt"
verify_file = "verify_data.txt"
np.random.seed(20)


data = []
train = []
verify = []
setosa = []
versicolor = []
virginica = []


# Read in data from file
with open(file) as f:
    for line in f:
        if line == '\n':
            break
        line = line.strip()
        l = line.split(',')
        if l[4] == "Iris-setosa":
            l[4] = 1
            setosa.append(l)
        elif l[4] == "Iris-versicolor":
            l[4] = 2
            versicolor.append(l)
        elif l[4] == "Iris-virginica":
            l[4] = 3
            virginica.append(l)

# Turn all the data into floats
for i in range(len(setosa)):
    setosa[i] = [float(x) for x in setosa[i]]

for i in range(len(versicolor)):
    versicolor[i] = [float(x) for x in versicolor[i]]

for i in range(len(virginica)):
    virginica[i] = [float(x) for x in virginica[i]]


#data shuffled to separate into train/verify
np.random.shuffle(setosa)
np.random.shuffle(versicolor)
np.random.shuffle(virginica)

#train verify split 80% 20%
train.extend(setosa[:40])
train.extend(versicolor[:40])
train.extend(virginica[:40])

verify.extend(setosa[-10:])
verify.extend(versicolor[-10:])
verify.extend(virginica[-10:])

all_data = np.array(setosa + versicolor + virginica)

# Take the first two features for clustering
features_for_clustering = all_data[:, :2]

# Use K-means clustering to group the data
kmeans = KMeans(n_clusters=3, n_init = 10, random_state=0)
kmeans.fit(features_for_clustering)

# Assign the cluster labels to each data point
cluster_labels = kmeans.labels_

# Update the class labels based on the cluster assignments
for i in range(len(all_data)):
    if all_data[i, 4] == 1:
        all_data[i, 4] = cluster_labels[i] + 1

# Sepal length and sepal width for visualization based off of first two traits
sepal_length = all_data[:, 0]
sepal_width = all_data[:, 1]

# Create a scatter plot for each cluster
plt.figure(figsize=(8, 6))

# Plot points for Setosa, Versicolor, Virginica, and centroids
plt.scatter(sepal_length[all_data[:, 4] == 1], sepal_width[all_data[:, 4] == 1], label='Setosa', c='red', marker='o')
plt.scatter(sepal_length[all_data[:, 4] == 2], sepal_width[all_data[:, 4] == 2], label='Versicolor', c='blue', marker='s')
plt.scatter(sepal_length[all_data[:, 4] == 3], sepal_width[all_data[:, 4] == 3], label='Virginica', c='green', marker='^')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='black', marker='X', label='Final Centroid')

# Set plot labels and title
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('K-Means Clustering of Iris Data')
plt.legend()
plt.grid(True)
plt.show()

#Confusion matrix to verify data
#conf_matrix = confusion_matrix(all_data[:, 4], cluster_labels)
#print(conf_matrix)

#turning training and verify into an array for easy.py
train_data = np.array(train)
verify_data = np.array(verify)

# Extract features and labels for training and verification
X_train = train_data[:, :4]
y_train = train_data[:, 4]

X_verify = verify_data[:, :4]
y_verify = verify_data[:, 4]

np.savetxt(train_file, train_data, delimiter=',', fmt='%f')
np.savetxt(verify_file, verify_data, delimiter=',', fmt='%f')

# Print the final centroids
print("Final Centroids:")
for i, centroid in enumerate(kmeans.cluster_centers_):
    print(f"Cluster {i + 1} - x = {centroid[0]}, y = {centroid[1]}")