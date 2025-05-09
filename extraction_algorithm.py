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
from skimage.morphology import binary_opening, binary_erosion
from scipy.ndimage import gaussian_filter
from skimage.transform import probabilistic_hough_line
from sklearn.preprocessing import MinMaxScaler
import rasterio
import cv2
from ultralytics import YOLO


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
            model_path = os.path.join(
                script_dir, "models", "logistic_regression.joblib"
            )

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

        Parameters:
        ----------
        pred_map : 2D numpy array
            A 2D numpy array of size (height, width) containing predicted class labels.

        Returns:
        -------
        model_bare_lands : 2D numpy array
            A 2D numpy array of size (height, width) containing binary mask of bare lands (Class №4) from land classification map.
        binary_smoothed_bare_lands : 2D numpy array
            A 2D numpy array of size (height, width) containing binary mask of smoothed bare lands by gaussian filter.
        edges : 2D numpy array
            A 2D numpy array of size (height, width) containing binary mask of edges of bare lands.
        """
        model_bare_lands = pred_map == 4
        bare_lands_without_noise = binary_opening(model_bare_lands, disk(3))
        smoothed_bare_lands = gaussian_filter(
            bare_lands_without_noise.astype(float), sigma=5
        )
        binary_smoothed_bare_lands = smoothed_bare_lands > 0.5

        eroded_bare_lands = binary_erosion(binary_smoothed_bare_lands, disk(1))
        edges = binary_smoothed_bare_lands.astype(int) - eroded_bare_lands.astype(int)

        return model_bare_lands, binary_smoothed_bare_lands, edges

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
            image=bare_lands_edges, threshold=65, line_length=50, line_gap=50
        )
        power_line_clearings = Image.new(
            "RGBA", (bare_lands_edges.shape[1], bare_lands_edges.shape[0]), (0, 0, 0, 0)
        )
        draw = ImageDraw.Draw(power_line_clearings)
        for line_coordinates in lines:
            draw.line(xy=line_coordinates, fill="red", width=3)
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

        bare_lands_edges = self.edge_detector(pred_map)[-1]
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


def load_image(image_path):
    """
    Loads an image from the specified file path and processes it into an RGB format.
    This function reads the last three bands of the image file, converts them into an RGB array,
    and normalizes the pixel values to the range [0, 255] if necessary. It also extracts metadata
    such as the transform, CRS (coordinate reference system), dimensions, and profile of the image.
    Args:
        image_path (str): The file path to the image to be loaded.
    Returns:
        tuple:
            - image_rgb (numpy.ndarray): A 3D array representing the RGB image with pixel values in the range [0, 255].
            - metadata (dict): A dictionary containing metadata about the image, including:
                - 'transform' (affine.Affine): The affine transformation matrix for the image.
                - 'crs' (rasterio.crs.CRS): The coordinate reference system of the image.
                - 'height' (int): The height of the image in pixels.
                - 'width' (int): The width of the image in pixels.
                - 'profile' (dict): A copy of the image's profile containing additional metadata.
    """
    with rasterio.open(image_path) as src:
        bands = src.read()[-3:]
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width
        profile = src.profile.copy()

        image_rgb = np.transpose(bands, (1, 2, 0))

        if image_rgb.dtype == np.uint16:
            min_val = image_rgb.min()
            max_val = image_rgb.max()
            if max_val > min_val:
                image_rgb = ((image_rgb - min_val) / (max_val - min_val) * 255).astype(
                    np.uint8
                )
            else:
                image_rgb = np.zeros_like(image_rgb, dtype=np.uint8)
        elif image_rgb.dtype != np.uint8:
            if image_rgb.max() <= 1:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            else:
                min_val = image_rgb.min()
                max_val = image_rgb.max()
                if max_val > min_val:
                    image_rgb = (
                        (image_rgb - min_val) / (max_val - min_val) * 255
                    ).astype(np.uint8)
                else:
                    image_rgb = np.zeros_like(image_rgb, dtype=np.uint8)

        metadata = {
            "transform": transform,
            "crs": crs,
            "height": height,
            "width": width,
            "profile": profile,
        }

        return image_rgb, metadata


def predict(image_rgb):
    """
    Perform object detection on the given RGB image using a pre-trained YOLO model.
    Args:
        image_rgb (numpy.ndarray): The input image in RGB format.
    Returns:
        list: A list of detection results, where each result contains information
              about detected objects such as bounding boxes, confidence scores,
              and class labels.
    Raises:
        FileNotFoundError: If the YOLO model file is not found at the expected path.
    Notes:
        - The YOLO model file is expected to be located in the 'models' directory
          within the plugin's 'clearings_extractor' folder.
        - The model uses a fixed image size of 384 and a confidence threshold of 0.446.
        - Inference is performed on the CPU.
    """

    plugin_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(plugin_dir, "clearings_extractor", "models", "best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"YOLO model not found at {model_path}")

    model = YOLO(model_path)
    results = model.predict(
        source=image_rgb,
        imgsz=384,
        conf=0.446,
        save=False,
        save_txt=False,
        device="cpu",
    )
    return results


def extract_clearings(results, metadata):
    """
    Extracts a binary mask representing clearings from the prediction results and metadata.
    Args:
        results (list): A list of prediction results, where each result contains:
            - `masks.data` (torch.Tensor): The predicted masks for each object.
            - `boxes.cls` (torch.Tensor): The predicted class labels for each object.
        metadata (dict): A dictionary containing metadata about the input image, including:
            - `height` (int): The height of the image.
            - `width` (int): The width of the image.
    Returns:
        numpy.ndarray: A binary mask of shape (height, width) where pixels belonging to
        clearings (class label 4) are set to 1, and all other pixels are set to 0.
    """
    height = metadata["height"]
    width = metadata["width"]

    result = results[0]
    pred_masks = result.masks.data if result.masks is not None else []
    pred_classes = result.boxes.cls.cpu().numpy() if result.boxes is not None else []

    binary_mask = np.zeros((height, width), dtype=np.uint8)
    if len(pred_masks) > 0 and len(pred_classes) > 0:
        for mask, cls in zip(pred_masks, pred_classes):
            if int(cls) == 4:
                mask = mask.cpu().numpy()
                if mask.shape != (height, width):
                    mask = cv2.resize(
                        mask, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                binary_mask = np.logical_or(binary_mask, mask > 0).astype(np.uint8)

    return binary_mask


def save_mask(binary_mask, metadata, output_path):
    """
    Saves a binary mask as a 4-channel RGBA raster image.
    Args:
        binary_mask (numpy.ndarray): A 2D binary array where 1 represents the mask area
            and 0 represents the background.
        metadata (dict): A dictionary containing metadata about the raster.
            Expected keys are:
                - 'height' (int): The height of the raster.
                - 'width' (int): The width of the raster.
                - 'profile' (dict): The rasterio profile dictionary containing raster metadata.
        output_path (str): The file path where the RGBA raster image will be saved.
    Returns:
        None: The function writes the RGBA raster to the specified output path.
    """
    height = metadata["height"]
    width = metadata["width"]
    profile = metadata["profile"]

    rgba_mask = np.zeros((height, width, 4), dtype=np.uint8)
    rgba_mask[binary_mask == 1, 0:3] = 255
    rgba_mask[:, :, 3] = binary_mask * 255

    profile.update({"count": 4, "dtype": "uint8", "nodata": None})

    with rasterio.open(output_path, "w", **profile) as dst:
        rgba_mask = np.transpose(rgba_mask, (2, 0, 1))
        dst.write(rgba_mask)


def predict_clearings_yolo(summer_path, output_path):
    """
    Predict clearings in a given image using a YOLO model, extract the clearings mask,
    and save the resulting mask to the specified output path.
    Args:
        summer_path (str): The file path to the input image (e.g., a summer satellite image).
        output_path (str): The file path where the resulting clearings mask will be saved.
    Returns:
        tuple: A tuple containing:
            - image (numpy.ndarray): The loaded input image.
            - clearings_mask (numpy.ndarray): The extracted clearings mask.
    """
    image, metadata = load_image(summer_path)
    results = predict(image)
    clearings_mask = extract_clearings(results, metadata)
    save_mask(clearings_mask, metadata, output_path)

    return image, clearings_mask


def calculate_wdrvi(image, clearings_mask, metadata, output_path):
    """
    Calculate the Wide Dynamic Range Vegetation Index (WDRVI) for a given image and save the result.
    The WDRVI is calculated using the formula:
        WDRVI = (0.2 * NIR - Red) / (0.2 * NIR + Red)
    where NIR is the near-infrared channel and Red is the red channel of the image.
    Parameters:
        image (numpy.ndarray): The input image as a 3D NumPy array (e.g., height x width x channels).
                               The first channel is assumed to be the red channel, and the second channel is the NIR channel.
        clearings_mask (numpy.ndarray): A 2D binary mask (same height and width as the image) where 1 indicates areas of interest
                                        (clearings) and 0 indicates areas to be masked out.
        metadata (dict): Metadata dictionary containing image profile information. The 'profile' key should contain
                         a dictionary with rasterio-compatible metadata.
        output_path (str): The file path where the resulting WDRVI raster will be saved.
    Returns:
        None: The function saves the WDRVI raster to the specified output path.
    Notes:
        - Areas where the sum of NIR and Red channels is zero are assigned NaN to avoid division by zero.
        - The output raster is saved with a single band, a data type of float32, and NaN as the nodata value.
    """

    masked_image = image.copy()
    masked_image[clearings_mask == 0] = 0

    nir = masked_image[..., 1].astype(float)
    red = masked_image[..., 0].astype(float)

    wdrvi = np.where((nir + red) == 0, np.nan, (nir * 0.2 - red) / (nir * 0.2 + red))

    profile = metadata["profile"]
    profile.update({"count": 1, "dtype": "float32", "nodata": np.nan})

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(wdrvi, 1)
