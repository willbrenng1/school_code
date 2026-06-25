from sklearn.linear_model import SGDClassifier


class SVM(object):
    def __init__(self, random_seed=None):
        """
        Initialize instance of SGDClassifier using SVM for use in fit and predict in self.clf variable.
        Make sure to set the random_state of the classifier with the passed in random_seed.
        """
        self.clf = SGDClassifier(random_state=random_seed)

    def fit(self, data, labels):
        """
        Fit function for calculating the weights using SVM and SGDClassifier.

        Args:
                data: (N, D) TF-IDF features for the data.
                labels: (N, ) list of class labels
        """
        self.clf.fit(data, labels)

    def predict(self, data):
        """
        Predict function for predicting the class labels using SVM and SGDClassifier.

        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                predictedLabels: numpy array of predicted classes for the data.
        """
        return self.clf.predict(data)
