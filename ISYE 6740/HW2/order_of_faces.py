from sklearn.decomposition import PCA
from scipy.io import loadmat
import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_array
from scipy.sparse.csgraph import shortest_path


# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class OrderOfFaces:
    """
    This class handles loading and processing facial image data for dimensionality
    reduction using the ISOMAP algorithm, with PCA as an optional comparison.

    Attributes:
    ----------
    images_path : str
        Path to the .mat file containing the image dataset.

    Methods:
    -------
    get_adjacency_matrix(epsilon):
        Returns the adjacency matrix based on a given epsilon neighborhood.

    get_best_epsilon():
        Returns the best epsilon for the ISOMAP algorithm, likely based on
        graph connectivity or reconstruction error.

    isomap(epsilon):
        Computes a 2D embedding of the data using the ISOMAP algorithm.

    pca(num_dim):
        Returns a low-dimensional embedding of the data using PCA.
    """

    def __init__(self, images_path="data/isomap.mat"):
        """
        Initializes the OrderOfFaces object and loads image data from the given path.

        Parameters:
        ----------
        images_path : str
            Path to the .mat file containing the facial images dataset.
        """

        self.datas = loadmat(
            images_path
        )  # https://stackoverflow.com/questions/874461/read-mat-files-in-python
        data_array = np.array(self.datas["images"])
        data_array_t = data_array.T

        self.data_array_t = data_array_t
        data_norm = np.linalg.norm(data_array_t, axis=1, keepdims=True)
        data_normalized = data_array_t / data_norm
        self.data_norm = data_normalized

    def get_adjacency_matrix(self, epsilon: float) -> np.ndarray:
        """
        Constructs the adjacency matrix using epsilon neighborhoods.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius within which points are considered connected.

        Returns:
        -------
        np.ndarray
            A 2D adjacency matrix (m x m) where each entry represents distance between
            neighbors within the epsilon threshold.
        """

        distance_matrix = cdist(
            self.data_array_t, self.data_array_t, metric="euclidean"
        ).astype(
            np.float64
        )  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html

        adjancy_matrix = np.where(
            distance_matrix <= epsilon, distance_matrix, 0
        ).astype(np.float64)
        np.fill_diagonal(adjancy_matrix, 0.0)

        self.adjancy_matrix = adjancy_matrix
        return adjancy_matrix

    def get_best_epsilon(self) -> float:
        """
        Heuristically determines the best epsilon value for graph connectivity in ISOMAP.

        Returns:
        -------
        float
            Optimal epsilon value ensuring a well-connected neighborhood graph.


        """

        return 14.63
        raise NotImplementedError("Not Implemented")

    def isomap(self, epsilon: float) -> np.ndarray:
        """
        Applies the ISOMAP algorithm to compute a 2D low-dimensional embedding of the dataset.

        Parameters:
        ----------
        epsilon : float
            The neighborhood radius for building the adjacency graph.

        Returns:
        -------
        np.ndarray
            A (m x 2) array where each row is a 2D embedding of the original data point.


        """

        adj_matrix = self.get_adjacency_matrix(epsilon)

        graph = csr_array(adj_matrix)
        short_path_matrix = shortest_path(
            csgraph=graph, directed=False, method="FW"
        )  # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csgraph.shortest_path.html
        I = np.identity(short_path_matrix.shape[0])

        ones = np.ones((short_path_matrix.shape[0], 1))

        H = I - (1 / short_path_matrix.shape[0]) * (ones @ ones.T)

        G = (-1 / 2) * H @ (short_path_matrix**2) @ H

        e_vals, e_vectors = np.linalg.eigh(G)
        idx = np.argsort(e_vals)[
            ::-1
        ]  # https://stackoverflow.com/questions/8092920/sort-eigenvalues-and-associated-eigenvectors-after-using-numpy-linalg-eig-in-pyt
        e_vals = e_vals[idx]
        e_vectors = e_vectors[:, idx]

        Z = e_vectors[:, :2] * np.sqrt(e_vals[:2])

        return Z

    def pca(self, num_dim: int) -> np.ndarray:
        """
        Applies PCA to reduce the dataset to a specified number of dimensions.

        Parameters:
        ----------
        num_dim : int
            Number of principal components to project the data onto.

        Returns:
        -------
        np.ndarray
            A (m x num_dim) array representing the dataset in a reduced PCA space.


        """
        pca = PCA(n_components=num_dim)

        pca_embedding = pca.fit_transform(self.data_array_t)
        return pca_embedding
