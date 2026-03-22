from PIL import Image
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class KMeansImpl:
    def __init__(self):

        ## TODO add any params to be needed for the clustering algorithm.
        pass

    def load_image(self, image_name="1.jpeg"):
        """
        Returns the image numpy array.
        It is important that image_name parameter defaults to the choice image name.
        """
        return np.array(Image.open(image_name))

    def compress(self, pixels, num_clusters, norm_distance=2):
        """
        Compress the image using K-Means clustering.

        Parameters:
            pixels: 3D image for each channel (a, b, 3), values range from 0 to 255.
            num_clusters: Number of clusters (k) to use for compression.
            norm_distance: Type of distance metric to use for clustering.
                            Can be 1 for Manhattan distance or 2 for Euclidean distance.
                            Default is 2 (Euclidean).
        Returns:
            Dictionary containing:
                "class": Cluster assignments for each pixel.
                "centroid": Locations of the cluster centroids.
                "img": Compressed image with each pixel assigned to its closest cluster.
                "number_of_iterations": total iterations taken by algorithm
                "time_taken": time taken by the compression algorithm
        """
        map = {
            "class": None,
            "centroid": None,
            "img": None,
            "number_of_iterations": None,
            "time_taken": None,
            "additional_args": {},
        }

        """
        # TODO - Add your implementation here.
        """

        start = (
            time.time()
        )  # https://stackoverflow.com/questions/7370801/how-do-i-measure-elapsed-time-in-python

        r_seed = np.random.default_rng(5)

        pixles_reshaped = pixels.reshape(-1, 3)

        centroids = r_seed.uniform(  # https://www.geeksforgeeks.org/python/numpy-random-uniform-in-python/
            pixles_reshaped.min(axis=0),
            pixles_reshaped.max(axis=0),
            size=(num_clusters, 3),
        )

        centroid_movement = 1
        run_counter = 1

        while centroid_movement > 0.001:

            if norm_distance == 2:
                diff_array = cdist(
                    pixles_reshaped, centroids, metric="euclidean"
                ).T  # https://www.geeksforgeeks.org/python/python-distance-between-collections-of-inputs/

            else:
                diff_array = cdist(
                    pixles_reshaped, centroids, metric="cityblock"
                ).T  # https://www.geeksforgeeks.org/python/python-distance-between-collections-of-inputs/

            assigments = diff_array.argmin(0) + 1

            non_empty_clusters = []
            new_centroids = []

            for i in range(1, num_clusters + 1):

                bool_mask = assigments == i
                if np.sum(
                    bool_mask
                ):  # https://stackoverflow.com/questions/6736590/fast-check-for-nan-in-numpy
                    new_centroids.append(pixles_reshaped[bool_mask].mean(axis=0))
                    non_empty_clusters.append(True)
                else:
                    non_empty_clusters.append(False)

            new_centroids = np.array(new_centroids, dtype=float)
            centroid_movement = np.linalg.norm(
                new_centroids - centroids[non_empty_clusters]
            )

            centroids = new_centroids
            num_clusters = centroids.shape[0]
            run_counter += 1

        cpixels = centroids[assigments - 1]
        compressed_image = cpixels.reshape(pixels.shape).astype(np.uint8)

        map["class"] = assigments
        map["centroid"] = centroids
        map["number_of_iterations"] = run_counter

        map["img"] = compressed_image

        end = time.time()
        time_taken = end - start
        map["time_taken"] = time_taken

        return map
