import cv2
import numpy as np
import pickle
import os
import random
from scipy import spatial
import matplotlib.pyplot as plt


# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.dim_embeddings = 128  # 128-dimensional embeddings
        self.facenet = cv2.dnn.readNetFromONNX("resnet50_128.onnx")  # extract deep features ie. embeddings

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)  # normalized embedding

    # Get dimensionality of the extracted embeddings.
    def get_embedding_dimensionality(self):
        return self.dim_embeddings


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=3, max_distance=0.9, min_prob=0.5): # num_neighbours=11, max_distance=0.8, min_prob=0.5
        # ToDo: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob
        
        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))  # (x,128)

        # Load face recognizer from pickle file if available.
        if os.path.exists("recognition_gallery.pkl"):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("recognition_gallery.pkl", 'wb') as f:
            pickle.dump((self.labels, self.embeddings), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("recognition_gallery.pkl", 'rb') as f:
            (self.labels, self.embeddings) = pickle.load(f)

    # ToDo
    # update the new embedding and label
    def update(self, face, label):
        # extract embedding from an aligned face
        embedding = self.facenet.predict(face)

        # store as a training sample in the gallery
        self.labels.append(label)
        self.embeddings = np.vstack((self.embeddings, embedding))
        return None

    # ToDo
    # predict the face label using k-NN
    def predict(self, face):

        embedding = self.facenet.predict(face)  # (128,)
        # calculate pair-wise distance
        dist = spatial.distance.cdist(embedding.reshape(1, -1), self.embeddings, metric='euclidean')  # (1,100=number of samples)
        dist = np.squeeze(dist)  # (100,)

        # find k nearest neighbors
        k_indices = np.argsort(dist)[:self.num_neighbours]  # indices in ascending order
        labels = np.array(self.labels)
        k_nearest_labels = labels[k_indices]

        # find the most frequent class label
        unique_labels, counts = np.unique(k_nearest_labels, return_counts=True)
        predicted_label = unique_labels[np.argmax(counts)]

        # calculate the posterior probability p(Ci|x)
        prob = np.max(counts) / self.num_neighbours  # all 1

        # calculate the minimum distance to the predicted class Ci
        dist_to_prediction = np.min(dist[k_indices][k_nearest_labels == predicted_label])

        # the decision rule for open-set identification
        if dist_to_prediction > self.max_distance or prob < self.min_prob:
            predicted_label = -1  # "unknown" label

        return predicted_label, prob, dist_to_prediction


# The FaceClustering class enables unsupervised clustering of face images according to their identity and
# re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=5, max_iter=25):  # num_clusters=2, max_iter=25
        # ToDo: Prepare FaceNet.
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, self.facenet.get_embedding_dimensionality()))  # (x,128)

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, self.facenet.get_embedding_dimensionality()))  # (n=2,128)
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists("clustering_gallery.pkl"):   # make errors if clustering_gallery.pkl is empty
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open("clustering_gallery.pkl", 'wb') as f:
            pickle.dump((self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership), f)

    # Load trained model from a pickle file.
    def load(self):
        with open("clustering_gallery.pkl", 'rb') as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = pickle.load(f)

    # ToDo:
    # update the new embedding
    def update(self, face):
        # extract embedding from an aligned face
        embedding = self.facenet.predict(face)
        # store as a sample in the gallery
        self.embeddings = np.vstack((self.embeddings, embedding))
        return None
    
    # ToDo:
    # cluster the embeddings using k-means
    def fit(self):
        # initialize cluster centers
        self.cluster_center = self.embeddings[np.random.choice(self.embeddings.shape[0], self.num_clusters, replace=False)]
        # initialize cluster membership
        self.cluster_membership = np.zeros(self.embeddings.shape[0])

        # iterate and save the value of the k-means objective function as a list
        obj_func = []
        for i in range(self.max_iter):
            # calculate the distance between each sample and each cluster center
            dist = spatial.distance.cdist(self.embeddings, self.cluster_center, metric='euclidean')

            # assign each sample to the nearest cluster center
            self.cluster_membership = np.argmin(dist, axis=1)

            # update the cluster centers
            for j in range(self.num_clusters):
                self.cluster_center[j] = np.mean(self.embeddings[self.cluster_membership == j], axis=0)

            # calculate the value of the k-means objective function
            obj_func.append(np.sum(np.min(dist, axis=1)**2))

        # plot the value of the k-means objective function over iterations
        plt.plot(obj_func)
        plt.xlabel('Iterations')
        plt.ylabel('Objective function')
        plt.show()

        return None

    # ToDo:
    def predict(self, face):

        embedding = self.facenet.predict(face)
        # compute the distribution of distances
        dist = spatial.distance.cdist(embedding.reshape(1, -1), self.cluster_center, metric='euclidean')  # (1,k)

        # the distances distribution dx
        distances_to_clusters = dist[0]  # (k,)
        # print('distances_to_clusters', distances_to_clusters)

        # the best matching cluster j
        predicted_label = np.argmin(distances_to_clusters)
        # print('predicted_label', predicted_label)

        return predicted_label, distances_to_clusters
