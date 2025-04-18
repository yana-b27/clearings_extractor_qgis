# Power line clearings extractor

"Power line clearings extractor" is a QGIS plugin designed to extract forest clearings under power lines from pairs of summer and winter Sentinel-2 satellite imagery. It provides a user-friendly dialog interface and advanced tools integrated into the QGIS Processing Toolbox, allowing users to process individual image pairs or batches of images efficiently. Plugin uses the algorithm implemented in the repository at the [link](https://github.com/yana-b27/clearings_extraction_algorithm).

The plugin automates:
- Validation of input raster images (band count and dimensions).
- Extraction of clearings using a custom algorithm (`find_clearing_algorithm`).
- Saving results as GeoTIFF files with geospatial metadata.
- Visualization of results on the QGIS map canvas.

## Features

1. **Dialog Interface**
   - Intuitive GUI for selecting summer and winter images and an output directory.
   - Option to add source images to the map alongside results.
   - Real-time logging with colored feedback (green for success, red for errors).

2. **Processing Toolbox Integration**
   - `Extract Clearings`: Process a single pair of images.
   - `Iterate Clearings Extractor`: Batch process multiple image pairs from directories.

3. **Core Functionality**
   - Validates summer images (≥5 bands) and winter images (≥3 bands).
   - Ensures matching dimensions between image pairs.
   - Overwrites existing output files to prevent duplication (*in development*).
   - Automatically adds results to the QGIS map with unique layer names.

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

## Video example

Сlick on the preview below to open the video in Google Drive:

[![Plugin example](./assets/preview.png)](https://drive.google.com/file/d/1oe7grTqy6GXvybtimDTYfABXdhE61Yso/view?usp=sharing)



