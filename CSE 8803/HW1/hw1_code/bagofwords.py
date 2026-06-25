import numpy as np
from sklearn.preprocessing import OneHotEncoder
from concurrent.futures import ProcessPoolExecutor  # You may not need to use this
from functools import partial  # You may not need to use this


class OHE_BOW(object):
    def __init__(self):
        """
        Initialize instance of OneHotEncoder in self.oh for use in fit and transform
        If needed, you may set handle_unknown to ignore when initalizing OneHotEncoder.
        """
        self.vocab_size = None  # keep
        self.oh = OneHotEncoder(handle_unknown="ignore")  # initialize

    def split_text(self, data: list[str]) -> list[list[str]]:
        """
        Helper function to separate each string into a list of individual words

        Args:
            data: list of N strings

        Return:
            data_split: list of N lists of individual words from each string

        """

        data_split = [e.split() for e in data]

        return data_split

    def flatten_list(self, data: list[list[str]]) -> np.ndarray:
        """
        Helper function to flatten a list of list of words into a single list

        Args:
            data: list of N lists of W_i words

        Return:
            data_split: (W,) numpy array of words,
                where W is the sum of the number of W_i words in each of the list of words

        """
        data_split = np.array([i for e in data for i in e])

        return data_split

    def fit(self, data: list[str]) -> None:
        """
        Fit the initialized instance of OneHotEncoder to the given data
        Use split_text to separate the given strings into a list of words and
        flatten_list to flatten the list of words in a sentence into a single list of words
        Reference the documentation for OneHotEncoder for the data shape it is expecting

        Set self.vocab_size to the number of unique words in the given data corpus

        Args:
            data: list of N strings

        Return:
            None

        Hint: You may find numpy's reshape function helpful when fitting the encoder

        """
        splits = self.split_text(data)
        flattened = self.flatten_list(splits)

        self.vocab_size = np.unique(flattened).shape[0]
        self.oh = self.oh.fit(flattened.reshape(-1, 1))

    def onehot(self, words: list[str]) -> np.ndarray:
        """
        Helper function to encode a list of words into one hot encoding format

        Args:
            words: list of W_i words from a string

        Return:
            onehotencoded: (W_i, D) numpy array where:
                W_i is the number of words in the current input list i
                D is the vocab size

        Hint:   .toarray() may be helpful in converting a sparse matrix into a numpy array
                You can use sklearn's built-in OneHotEncoder transform function
        """
        onehotencoded = self.oh.transform(np.array(words).reshape(-1, 1)).toarray()

        return onehotencoded

    def transform(self, data: list[str]) -> np.ndarray:
        """
        Use the already fitted instance of OneHotEncoder to help you transform the given
        data into a bag of words representation. You will need to separate each string
        into a list of words and then encode this list of words into the one-hot format.
        Using the one-hot encoding of each sentence, convert it to the bag-of-words representation.

        For any empty strings append a (1, D) array of zeros instead.


        Args:
            data: list of N strings

        Return:
            bow: (N, D) numpy array

        Hints:
            1. Using a try and except block during one hot encoding transform may be helpful
            2. If using ProcessPoolExecutor, you may create a helper function inside of the OHE_BOW class
        """
        sentences = []
        text_splits = self.split_text(data)
        print(text_splits)
        for w in text_splits:
            if len(w) < 1:
                print("x")
                sentences.append(np.zeros((1, self.vocab_size)))

            else:

                encodings = self.onehot(w)
                sentences.append(encodings.sum(axis=0, keepdims=False))

        return np.vstack(sentences)
