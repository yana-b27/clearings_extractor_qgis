# Power line clearings extractor

A QGIS plugin to extract forest clearings under power lines from Sentinel-2 satellite imagery.

## Description

The plugin extracts forest clearings under power lines on Sentinel-2 satellite images. Before applying the algorithm, the images are pre-processed, including normalization and contrast reduction. The plugin uses the previously trained Logistic Regression model to create a classification map. From the resulting map, a binary mask of the class of low vegetation, or "grassland", is extracted. The mask is then used to find lines using the Canny Boundary Detector and the Hough Probabilistic Transform. Selected lines refer to forest clearings boundaries in raster format.

## Features

- Extracts power line clearings using summer and winter Sentinel-2 images.
- Provides a user-friendly dialog with real-time logging.
- Outputs a raster map of detected clearings.

## Directory Structure

Below is the structure of the plugin directory:

clearings_extractor_qgis/
📜 __init__.py
📜 clearings_extractor.py
📜 clearings_extractor_dialog.py
📜 clearings_extractor_dialog_base.ui
📜 extraction_algorithm.py
📜 resources.py
📁 help/
   └─ 📜 index.rst
📁 i18n/
   └─ 📜 ClearingsExtractor_en.ts
📜 README.md
📜 pb_tool.cfg
📜 plugin_upload.py
