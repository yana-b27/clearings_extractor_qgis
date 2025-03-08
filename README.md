# Power line clearings extractor

A QGIS plugin to extract forest clearings under power lines from Sentinel-2 satellite imagery.

## Description

The plugin processes Sentinel-2 summer and winter images to detect forest clearings beneath power lines. It applies pre-processing (normalization, contrast reduction), uses Logistic Regression for classification, and employs Canny/Hough transforms to identify clearing boundaries, outputting results as a raster.

## Features

- Extracts power line clearings using summer (5 bands) and winter (3 bands) Sentinel-2 images.
- Provides a user-friendly dialog with real-time logging.
- Outputs a raster map of detected clearings.
