import numpy as np
import pickle
from classifier import NearestNeighborClassifier


# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(self,
                 classifier=NearestNeighborClassifier(),
                 false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True)):

        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):

        with open(train_data_file, 'rb') as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding='bytes')

        with open(test_data_file, 'rb') as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding='bytes')

    
    # Run the evaluation and find performance measure (identification rates) at different similarity thresholds.
    def run(self):

        similarity_thresholds = None
        identification_rates = None

        # Fit the classifier on the training data
        self.classifier.fit(self.train_embeddings, self.train_labels)

        # Predict similarities on the test data
        prediction_labels, self.similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        # calculate similarity threshold and identification rate
        similarity_thresholds = self.select_similarity_threshold(self.similarities[self.test_labels == UNKNOWN_LABEL],
                                                                 self.false_alarm_rate_range)
        self.similarity_thresholds = similarity_thresholds
        identification_rates = self.calc_identification_rate(prediction_labels)

        # Report all performance measures.
        evaluation_results = {'similarity_thresholds': similarity_thresholds,
                              'identification_rates': identification_rates}


        # Find suitable similarity thresholds
        print('Find suitable similarity thresholds from the DIR curve under the following requirements:')
        print('- The false alarm rate should not exceed 1 % and the identification rate should be maximized.')

        fars1 = np.sum(self.similarities[self.test_labels == UNKNOWN_LABEL] >= similarity_thresholds[:, np.newaxis],
                      axis=1) / np.sum(self.test_labels == UNKNOWN_LABEL)
        self.similarity_thresholds = similarity_thresholds[fars1 <= 0.01]
        dir= self.calc_identification_rate(prediction_labels)
        index1 = np.argmax(dir)
        threshold1 = self.similarity_thresholds[index1]
        print('Threshold:', threshold1)
        print('Maximum identification rate:', dir[index1])


        print('- The identification rate should not fall below 90 % and false alarms should be minimized.')

        thresholds = similarity_thresholds[identification_rates >= 0.9]
        fars2 = np.sum(self.similarities[self.test_labels == UNKNOWN_LABEL] >= thresholds[:, np.newaxis],
                      axis=1) / np.sum(self.test_labels == UNKNOWN_LABEL)
        index2 = np.argmin(fars2)
        threshold2 = thresholds[index2]
        print('Threshold:', threshold2)
        print('Minimum false alarm rate:', fars2[index2])

        return evaluation_results

    
    # calculates the similarity threshold for a given false alarm rate
    def select_similarity_threshold(self, similarity, false_alarm_rate):

        return np.percentile(similarity, (1 - false_alarm_rate) * 100)

    
    def calc_identification_rate(self, prediction_labels):
        # for test set of knowns
        f_knowns = self.test_labels != UNKNOWN_LABEL  # filter
        similarities_known = self.similarities[f_knowns]

        # computes the identification rate at rank 1
        identification_rates = np.sum(similarities_known[prediction_labels[f_knowns] == self.test_labels[f_knowns]] >=
                                      self.similarity_thresholds[:, np.newaxis], axis=1) / np.sum(f_knowns)

        return identification_rates
