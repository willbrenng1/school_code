import numpy as np


class SVD(object):
    def __init__(self):
        pass

    def svd(self, data):
        """
        Do SVD. You could use numpy SVD.
        Args:
                data: (N, D) TF-IDF features for the data.

        Return:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                V^T: (D,D) numpy array
        """

        u, s, v = np.linalg.svd(data, full_matrices=True)

        return u, s, v

    def rebuild_svd(self, U, S, VT, k):
        """
        Rebuild SVD by k componments.

        Args:
                U: (N,N) numpy array
                S: (min(N,D), ) numpy array
                VT: (D,D) numpy array, transpose of V
                k: int corresponding to number of components

        Return:
                data_rebuild: (N,D) numpy array

        Hint: numpy.matmul may be helpful for reconstruction.
        """

        s_t = np.diag(S[:k])
        u_t = U[:, :k]
        v_t = VT[:k, :]

        return u_t @ s_t @ v_t

    def compression_ratio(self, data, k):
        """
        Compute the compression ratio: (num stored values in compressed)/(num stored values in original)

        Args:
                data: (N, D) TF-IDF features for the data.
                k: int corresponding to number of components

        Return:
                compression_ratio: float of proportion of storage used
        """

        u, s, v = self.svd(data)

        return (
            k
            * (1 + u[:, :k].shape[0] + v[:k, :].shape[1])
            / (data.shape[0] * data.shape[1])
        )

    def recovered_variance_proportion(self, S, k):
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
                S: (min(N,D), ) numpy array
                k: int, rank of approximation

        Return:
                recovered_var: float corresponding to proportion of recovered variance
        """
        return np.sum(S[:k] ** 2) / np.sum(S**2)
