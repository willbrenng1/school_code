import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class BagOfWords(object):
    def __init__(self):
        """
        Initialize instance of CountVectorizer in self.vectorizer for use in fit and transform
        """
        self.vectorizer = CountVectorizer()

    def fit(self, data: list[str]) -> None:
        """
        Use the initialized instance of CountVectorizer to fit to the given data

        Args:
            data: list of N strings

        Return:
            None
        """

        self.vectorizer.fit(data)

    def transform(self, data: list[str]) -> np.ndarray:
        """
        Use the initialized instance of CountVectorizer to transform the given data

        Args:
            data: list of N strings

        Return:
            x: (N, D) bag of words numpy array

        Hint: .toarray() may be helpful
        """

        return self.vectorizer.transform(data).toarray()


class TfIdf(object):
    def __init__(self):
        """
        Initialize instance of TfidfVectorizer in self.vectorizer for use in fit and transform

        """

        self.vectorizer = TfidfVectorizer()

    def fit(self, data: list[str]) -> None:
        """
        Use the initialized instance of TfidfVectorizer to fit to the given data

        Args:
            data: list of N strings

        Return:
            None
        """
        self.vectorizer.fit(data)

    def transform(self, data: list[str]) -> np.ndarray:
        """
        Use the initialized instance of TfidfVectorizer to transform the given data

        Args:
            data: list of N strings

        Return:
            x: (N, D) tfi-df numpy array

        Hint: .toarray() may be helpful
        """

        return self.vectorizer.transform(data).toarray()
