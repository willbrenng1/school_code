import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA

# -----------------------------------------------------------------------------
# NOTE: This file consists of 2 classes

# 1. EigenFacesResult - This class should not be modified. Gradescope will use the output of run()
# method in this format.
# 2. EigenFaces - This is class which will implement the eigen faces algorithm and return the results.
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# NOTE: This class should NOT be modified.
# Gradescope will depend on the structure of this class as defined.
# -----------------------------------------------------------------------------
class EigenFacesResult:
    """
    A structured container for storing the results of the EigenFaces computation.

    Attributes
    ----------
    subject_1_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 1.
        A plt.imshow(map['subject_1_eigen_faces'][0]) should display first in a eigen face for subject 1

    subject_2_eigen_faces : np.ndarray
        A (6, a, b) array representing the top 6 eigenfaces for subject 2.
        A plt.imshow(map['subject_2_eigen_faces'][0]) should display first in a eigen face for subject 2

    s11 : float
        Projection residual of subject 1 test image on subject 1 eigenfaces.

    s12 : float
        Projection residual of subject 2 test image on subject 1 eigenfaces.

    s21 : float
        Projection residual of subject 1 test image on subject 2 eigenfaces.

    s22 : float
        Projection residual of subject 2 test image on subject 2 eigenfaces.
    """

    def __init__(
        self,
        subject_1_eigen_faces: np.ndarray,
        subject_2_eigen_faces: np.ndarray,
        s11: float,
        s12: float,
        s21: float,
        s22: float,
    ):
        self.subject_1_eigen_faces = subject_1_eigen_faces
        self.subject_2_eigen_faces = subject_2_eigen_faces
        self.s11 = s11
        self.s12 = s12
        self.s21 = s21
        self.s22 = s22


# -----------------------------------------------------------------------------
# NOTE: Do not change the parameters / return types for pre defined methods.
# -----------------------------------------------------------------------------
class EigenFaces:
    """
    This class handles loading facial images for two subjects, computing eigenfaces
    via PCA, and evaluating projection residuals for test images.

    Methods
    -------
    run():
        Computes the eigenfaces for each subject and the projection residuals for test images.
    """

    def __init__(self, images_root_directory="data/yalefaces"):
        """
        Initializes the EigenFaces object and loads all relevant facial images from the specified directory.

        Parameters
        ----------
        images_root_directory : str
            The path to the root directory containing subject images.
        """

        self.downcast_images = None
        self.downcast_test_iamges = None
        self.matrix_images = None

        og_image_dict = {}
        for image in os.listdir(images_root_directory):
            og_image_dict[image] = np.array(
                Image.open(os.path.join(images_root_directory, image))
            )

        downcast_image_dict = defaultdict(list)
        for k, v in og_image_dict.items():
            if "test" in k:
                continue
            person = k.split(".")[0]
            downcast = v[::4, ::4]
            downcast = downcast[:-1, :]
            downcast_flat = downcast.flatten()
            downcast_image_dict[person].append(downcast_flat)

        downcast_test_images_dict = {}
        for k, v in og_image_dict.items():
            if "test" in k:
                person = k.split(".")[0]
                person = person.split("-")[0]
                downcast = v[::4, ::4]
                downcast = downcast[:-1, :]
                downcast_flat = downcast.flatten()
                downcast_test_images_dict[person] = downcast_flat

        matrix_dict = {k: np.array(v) for k, v in downcast_image_dict.items()}

        self.downcast_images = downcast_image_dict
        self.matrix_images = matrix_dict

        self.downcast_test_iamges = downcast_test_images_dict
        self.pca_comps = None
        self.res_dict = None

        # raise NotImplementedError("Not Implemented")

    def run(self) -> EigenFacesResult:
        """
        Computes eigenfaces for both subjects and projection residuals
        for test images using those eigenfaces.

        Returns
        -------
        EigenFacesResult
            Object containing eigenfaces and residuals for both subjects.
        """

        pca_dict = {}
        pca_comps = {}
        pca_means = {}

        for k, v in self.matrix_images.items():
            pca = PCA(n_components=6)
            X_pca = pca.fit_transform(v)
            pca_dict[k] = X_pca
            pca_comps[k] = pca.components_
            pca_means[k] = pca.mean_

        self.pca_comps = pca_comps
        reshaped_pca = defaultdict(list)

        for k, v in pca_comps.items():
            for vector_num in range(6):
                vector_reshaped = v[vector_num].reshape(60, 80)
                reshaped_pca[k].append(vector_reshaped)

        reshaped_pca_submission = {k: np.array(v) for k, v in reshaped_pca.items()}

        top_eigen_dict = {k: v[0].flatten() for k, v in pca_comps.items()}
        res_dict = {}
        for k, v in top_eigen_dict.items():
            means = pca_means[k]
            for test_key, test_i in self.downcast_test_iamges.items():
                test_centered = test_i - means
                keys = f"{k}-{test_key}"
                res_dict[keys] = (
                    np.linalg.norm(test_centered - np.dot(test_centered, v) * v) ** 2
                )

        self.res_dict = res_dict
        return EigenFacesResult(
            subject_1_eigen_faces=reshaped_pca_submission["subject01"],
            subject_2_eigen_faces=reshaped_pca_submission["subject02"],
            s11=res_dict["subject01-subject01"],
            s12=res_dict["subject01-subject02"],
            s21=res_dict["subject02-subject01"],
            s22=res_dict["subject02-subject02"],
        )
