"""
extraction_algorithm.py

Module for extracting power line clearings from satellite imagery.

This module provides a set of functions and classes for detecting power line clearings
in satellite images. It includes methods for creating image datasets for exact areas
and extracting clearings from land classification maps.

Classes:
    ImageDataset: A class representing a dataset of satellite images for exact area.
    LandClassModel: A class representing a land classification model.
    ClearingsExtractor: A class representing an extractor for power line clearings
        from satellite imagery.

Functions:
    find_clearings_algorithm: A function that finds power line clearings in a given land classification map.

Notes:
    This module is designed to work with Sentinel-2 satellite images.
    The functions and classes in this module are intended to be used together to
    extract power line clearings from satellite imagery.
"""

import os
import joblib
import numpy as np
import rioxarray as rxr
import spyndex
from PIL import Image, ImageDraw
from skimage.morphology import disk
from skimage.morphology import binary_opening
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from sklearn.preprocessing import MinMaxScaler


class ImageDataset:
    """
    A class representing a dataset of satellite images for exact area.

    Attributes:
      image_data_3d (3D numpy array): A 3D numpy array of size (height, width, number of channels) containing satellite image data.
      image_data_2d_arr (2D numpy array): A 2D numpy array of size (height * width, number of channels) containing satellite image data.

    Methods:
      add_channels(summer_image_path, winter_image_path): Adds channels from summer and winter images to the image dataset.
      saturation_contrast(): Applies contrast stretching to the satellite image channels.
      normalize_channels(num_channel_end, num_channel_start=0): Normalizes the channels of the satellite image data.
      compute_indices(): Computes spectral indices (SAVI, NDWI, NDBI) from the satellite image data.
      make_image_2d(): Creates a 2D numpy array from the 3D numpy array and normalizes the values.

    Notes:
      The `image_data_3d` attribute, which is a 3D numpy array, was created for channels visualization on the map.
      The `image_data_2d_arr` attribute is a 2D numpy array which was created for learning models.
    """

    def __init__(self, image_data_3d=None, image_data_2d_arr=None):
        """
        __init__ method for ImageDataset class.

        Parameters
        ----------
        image_data_3d : 3D numpy array
            3D numpy array of size (height, width, number of channels) containing satellite image data
        image_data_2d_arr : 2D numpy array
            2D numpy array of size (height*width, number of channels) containing satellite image data
        """
        self.image_data_3d = image_data_3d
        self.image_data_2d_arr = image_data_2d_arr

    def add_channels(self, summer_image_path, winter_image_path):
        """
        Method for adding channels from summer and winter images to image dataset.

        Parameters
        ----------
        summer_image_path : str
            path of the summer satellite image
        winter_image_path : str
            path of the winter satellite image
        """

        summer_image = rxr.open_rasterio(summer_image_path)
        winter_image = rxr.open_rasterio(winter_image_path)

        image_dataset = np.zeros(
            (summer_image.shape[1], summer_image.shape[2], 11), dtype=np.float64
        )

        for b in range(5):
            image_dataset[:, :, b] = summer_image[b, :, :]
        for b in range(5, 8):
            image_dataset[:, :, b] = winter_image[b - 5, :, :]

        self.image_data_3d = image_dataset

    def saturation_contrast(self):
        """
        Method for contrast stretching of satellite image channels.
        """

        def change_max_1(x):
            """
            A function for changing the maximum value of a numpy array to a 99% percentile.

            Parameters
            ----------
            x : numpy array
                The input array

            Returns
            -------
            numpy array
                The array with the maximum value changed to the 99% percentile
            """

            if x > perc_99:
                return perc_99
            else:
                return x

        for b in range(8):
            perc_99 = np.quantile(self.image_data_3d[:, :, b], 0.99)
            change_max_1_func = np.vectorize(change_max_1)
            self.image_data_3d[:, :, b] = change_max_1_func(self.image_data_3d[:, :, b])

    def normalize_channels(self, num_channel_end, num_channel_start=0):
        """
        Method for normalizing channels of satellite image data.

        Parameters
        ----------
        num_channel_end : int
            The last channel to normalize.
        num_channel_start : int, optional
            The first channel to normalize. Defaults to 0.

        """
        for b in range(num_channel_start, num_channel_end):
            min_val_b = np.min(self.image_data_3d[:, :, b])
            max_val_b = np.max(self.image_data_3d[:, :, b])
            self.image_data_3d[:, :, b] = (self.image_data_3d[:, :, b] - min_val_b) / (
                max_val_b - min_val_b
            )

    def compute_indices(self):
        """
        Method for computing spectral indices from satellite image data.

        It computes 3 indices: SAVI (Soil Adjusted Vegetation Index), NDWI (Normalized Difference Water Index) and NDBI (Normalized Difference Built-up Index).
        These indices are computed from the red, green, blue, near infrared, and short-wave infrared channels of the satellite image data.

        The computed indices are stored in the image_data_3d attribute of the ImageDataset object (3d numpy array) in the 8th, 9th, and 10th channels.

        """
        savi = spyndex.computeIndex(
            index=["SAVI"],
            params={
                "L": 0.25,
                "R": self.image_data_3d[:, :, 2],
                "N": self.image_data_3d[:, :, 3],
            },
        )
        ndwi = spyndex.computeIndex(
            index=["NDWI"],
            params={"G": self.image_data_3d[:, :, 1], "N": self.image_data_3d[:, :, 3]},
        )
        ndbi = spyndex.computeIndex(
            index=["NDBI"],
            params={
                "S1": self.image_data_3d[:, :, 4],
                "N": self.image_data_3d[:, :, 3],
            },
        )

        self.image_data_3d[:, :, 8] = savi
        self.image_data_3d[:, :, 9] = ndwi
        self.image_data_3d[:, :, 10] = ndbi

    def create_dataset(self, summer_image_path, winter_image_path):
        """
        Method for creating image dataset (image_data_3d attribute).

        Parameters
        ----------
        summer_image_path : str
            path of the summer satellite image
        winter_image_path : str
            path of the winter satellite image

        """

        self.add_channels(summer_image_path, winter_image_path)
        self.saturation_contrast()
        self.normalize_channels(num_channel_end=8)
        self.compute_indices()
        self.normalize_channels(8, 11)

    def make_image_2d(self):
        """
        Method for creating 2D numpy array of size (height*width, number of channels) from 3D numpy array of size (height, width, number of channels).

        It reshapes 3D numpy array to 2D numpy array and normalizes the values of the 2D array to be between 0 and 1.

        The resulting 2D numpy array is stored in the image_data_2d_arr attribute of the ImageDataset object.

        """
        new_shape = (
            self.image_data_3d.shape[0] * self.image_data_3d.shape[1],
            self.image_data_3d.shape[2],
        )
        img_as_2d_arr = self.image_data_3d[:, :, :].reshape(new_shape)
        scaler = MinMaxScaler()
        img_as_2d_arr = scaler.fit_transform(img_as_2d_arr)
        self.image_data_2d_arr = img_as_2d_arr


class LandClassModel:
    """
    A class used to represent a Land Classification Model.
    Attributes
        A model object (e.g. from scikit-learn) used for satellite image land classification.
    Methods
    __init__(model=joblib.load("logistic_regression.joblib"))
        Initializes the LandClassModel with a given model.
    predict_for_image(img_2d_array, image_dataset)
        Makes a prediction on the given 2D image array using the model.
    """

    def __init__(self, model_path=None):
        """
        Initializes the LandClassModel with a given model path.

        Parameters
        ----------
        model_path : str, optional
            Path to the model file. If None, defaults to 'logistic_regression.joblib' in the script's directory.
        """
        if model_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "logistic_regression.joblib")

        self.model = joblib.load(model_path)

    def predict_for_image(self, img_2d_array, image_dataset):
        """
        Make a prediction on the given 2D image array using the given model.

        Parameters
        ----------
        img_2d_array : 2D numpy array
            2D numpy array of size (height*width, number of channels) containing satellite image data
        image_dataset : 3D numpy array
            3D numpy array of size (height, width, number of channels) containing satellite image data

        Returns
        -------
        model_pred : 2D numpy array
            2D numpy array of size (height, width) containing predicted class labels
        """
        model_pred = self.model.predict(img_2d_array)
        model_pred = model_pred.reshape(image_dataset[:, :, 0].shape)

        return model_pred


class ClearingsExtractor:
    """
    A class representing an extractor for power line clearings from satellite imagery.

    Methods:
      edge_detector(self, pred_map): Detects edges of bare lands in a given land classification map.
      find_lines(self, bare_lands_edges): Finds power line clearings in a given edge image.
      extract(self, pred_map): Finds power line clearings on a given classification map.

    Notes:
      The `edge_detector` method detects the edges using the Canny edge detector.
      The `find_power_line_clearings` method finds power line clearings in an edge image using the probabilistic Hough transform.
    """

    def edge_detector(self, pred_map):
        """
        Detects edges of bare lands in a given land classification map.

        Parameters
        ----------
        pred_map : 2D numpy array
            A 2D numpy array of size (height, width) containing predicted class labels

        Returns
        -------
        bare_lands_edges : 2D numpy array
            A 2D numpy array of the same size as `pred_map` with the edges of 'Луга' (class №4) after applying Canny edge detection
        """
        model_bare_lands = pred_map == 4
        bare_lands_without_noise = binary_opening(model_bare_lands, disk(3))
        bare_lands_edges = canny(bare_lands_without_noise, sigma=5)

        return bare_lands_edges

    def find_lines(self, bare_lands_edges):
        """
        Finds power line clearings in a given edge image.

        Parameters
        ----------
        bare_lands_edges : 2D numpy array
            A 2D numpy array of size (height, width) containing the edges of Class №4 ('Луга') after applying Canny edge detection

        Returns
        -------
        power_line_clearings : 3D numpy array
            A 3D numpy array of size (height, width, 4) RGBA image containing the detected power line clearings as yellow lines of width 3
        """
        lines = probabilistic_hough_line(
            image=bare_lands_edges, threshold=100, line_length=100, line_gap=50
        )
        power_line_clearings = Image.new(
            "RGBA", (bare_lands_edges.shape[1], bare_lands_edges.shape[0]), (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(power_line_clearings)
        for line_coordinates in lines:
            draw.line(xy=line_coordinates, fill="yellow", width=3)
        power_line_clearings = np.array(power_line_clearings)

        return power_line_clearings

    def extract(self, pred_map):
        """
        Finds power line clearings on a given classification map.

        Parameters
        ----------
        pred_map : 2D numpy array
            A 2D numpy array of size (height, width) containing predicted class labels

        Returns
        -------
        power_line_clearings : 3D numpy array
            A 3D numpy array of size (height, width, 4) RGBA image containing the detected power line clearings as yellow lines of width 3
        """

        bare_lands_edges = self.edge_detector(pred_map)
        power_line_clearings = self.find_lines(bare_lands_edges)

        return power_line_clearings


def find_clearing_algorithm(summer_image_path, winter_image_path):
    """
    Final algorithm that finds power line clearings in a given land classification map.

    Parameters
    ----------
    model : sklearn model
        Trained machine learning model for land classification
    summer_image_path : str
        Path of the summer satellite image
    winter_image_path : str
        Path of the winter satellite image

    Returns
    -------
    power_line_clearings : 3D numpy array
        A 3D numpy array of size (height, width, 4) RGBA image containing the detected power line clearings as yellow lines of width 3
    """

    image_dataset = ImageDataset()
    image_dataset.create_dataset(summer_image_path, winter_image_path)
    image_dataset.make_image_2d()
    logreg_class_model = LandClassModel()
    pred_map = logreg_class_model.predict_for_image(
        image_dataset.image_data_2d_arr, image_dataset.image_data_3d
    )
    clearing_extractor = ClearingsExtractor()
    power_line_clearings = clearing_extractor.extract(pred_map)

    return power_line_clearings
