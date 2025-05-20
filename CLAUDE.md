# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Pixel Art Scaler is a Python application that restores the crispness of pixel art images that have been shared online. It detects the original scale, removes compression artifacts by downscaling to a true 1:1 pixel ratio, and then re-upscales using nearest neighbor interpolation for perfect pixel edges.

## Development Setup
```bash
# Install just (if needed) and run quickstart
just quickstart

# Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Run the application (GUI)
uv run python gui.py

# Process a file (CLI)
uv run python cli.py /path/to/image.png
```

## Key Components
- `cli.py`: Command-line interface and core image processing algorithms
- `gui.py`: PySide6 GUI application with options dialog
- `PixelArtDownscaler` class: Contains the detection, downscaling, and upscaling logic

## Architecture
The application uses:
- PySide6 (Qt for Python) for the GUI
- PIL (Pillow) for image processing
- NumPy for efficient array operations
- SciPy for frequency domain analysis (FFT)

The pixel art scaling process works in these steps:
1. Detect if the pixel grid is offset from the image edges
2. Crop to align the pixel grid if needed
3. Detect the pixel scale using multiple approaches:
   - Frequency domain analysis with FFT
   - Edge alignment detection
   - Uniform block recognition
4. Downscale to true 1:1 pixel ratio to remove compression artifacts
5. Re-upscale using nearest neighbor interpolation for perfect crisp pixels
6. Export both the clean 1:1 version and upscaled versions

## Commands
- `uv run python gui.py` - Run the GUI application
- `uv run python cli.py IMAGE_PATH` - Process an image with the CLI tool