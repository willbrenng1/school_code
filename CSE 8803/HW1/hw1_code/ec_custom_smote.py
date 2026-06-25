import numpy as np
from sklearn.neighbors import NearestNeighbors

np.random.seed(42)


### SUBMIT THIS FILE TO THE BONUS EXTRA CREDIT GRADESCOPE SECTION ###
def custom_smote(samples: np.ndarray, n: int, k: int) -> np.ndarray:
    """
    Implement the SMOTE algorithm on the training data.
    For this function, you can use scikit-learn's NearestNeighbors function to find
    the k nearest neighbors.

    Hint:
        1. randomly pick a minority sample
        2. find its k nearest neighbors
        3. randomly pick one of the k nearest neighbors and generate a synthetic sample
        4. when k equals 1, the closest neighbor to the current point (not the point itself) is selected
           as the nearest neighbor

    Args:
        samples: (N, D) ndarray of minority class samples
        n: Number of synthetic samples to generate
        k: Number of nearest neighbors
    Return:
        synthetic: (n, D) ndarray of synthetic samples
    """

    knn = NearestNeighbors(n_neighbors=k + 1)
    knn.fit(samples)

    smote_dps = []
    for _ in range(n):
        samples_dp = samples[np.random.randint(samples.shape[0])]
        nearest_indicies = knn.kneighbors([samples_dp])[1][0][1:]
        smote_dp = samples[nearest_indicies[np.random.randint(len(nearest_indicies))]]
        smote_dps.append(samples_dp + np.random.random() * (smote_dp - samples_dp))
    return np.array(smote_dps)
