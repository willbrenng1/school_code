import numpy as np
from sklearn.naive_bayes import MultinomialNB, GaussianNB


class MNB(object):
    def __init__(self):
        """
        Initialize instance of MultinomialNB in self.clf for use in fit and transform
        """
        self.clf = MultinomialNB()

    def fit(self, data: np.ndarray, y: np.ndarray) -> None:
        """
        Use the initialized instance of MultinomialNB to fit to the given data

        Args:
            data: (N, D) numpy array of numerically encoded sentences
            y: (N,) numpy vector of labels

        Return:
            None
        """
        self.clf.fit(data, y)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Use the initialized instance of MultinomialNB to predict the label of the given data

        Args:
            data: (N, D) numpy array of numerically encoded sentences

        Return:
            y_hat: (N,) numpy vector of labels

        """

        return self.clf.predict(data)


class GNB(object):
    def __init__(self):
        """
        Initialize instance of GaussianNB in self.clf for use in fit and transform
        """

        self.clf = GaussianNB()

    def fit(self, data: np.ndarray, y: np.ndarray) -> None:
        """
        Use the initialized instance of GaussianNB to fit to the given data

        Args:
            data: (N, D) numpy array of numerically encoded sentences
            y: (N,) numpy vector of labels

        Return:
            None
        """
        self.clf.fit(data, y)

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Use the initialized instance of GaussianNB to predict the label of the given data

        Args:
            data: (N, D) numpy array of numerically encoded sentences

        Return:
            y_hat: (N,) numpy vector of labels

        """
        return self.clf.predict(data)
