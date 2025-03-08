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

```
clearings_extractor_qgis/
├── __init__.py                   # Plugin metadata and initialization
├── clearings_extractor.py        # Main plugin logic and dialog handling
├── clearings_extractor_dialog.py # Dialog class generated from UI file
├── clearings_extractor_dialog_base.ui # QGIS UI design file (Qt Designer)
├── extraction_algorithm.py       # Core algorithm for clearing detection
├── resources.py                  # Compiled resources (e.g., icons)
├── help/                         # Documentation files
│   └── index.rst                 # Main help file in reStructuredText
├── i18n/                         # Internationalization files
│   └── ClearingsExtractor_en.ts  # English translation file
├── README.md                     # This file, plugin overview
├── pb_tool.cfg                   # Configuration for pb_tool (optional)
└── plugin_upload.py              # Script for uploading to QGIS repository
```
