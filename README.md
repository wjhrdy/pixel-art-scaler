# Pixel Art Scaler

A Python application that restores the crispness of pixel art images that have been shared online by detecting scale, removing compression artifacts, and reupscaling with nearest neighbor interpolation.

## Features

- Detects pixel art scale using frequency domain analysis (FFT)
- Detects and corrects grid alignment offset from image edges
- Downscales to true 1:1 pixel ratio to remove compression artifacts
- Re-upscales using nearest neighbor for perfect pixel edges
- Creates both clean 1:1 version and upscaled versions
- Preserves transparency
- Exports to PNG format
- Modern Qt-based GUI with PySide6
- Command-line interface for batch processing

## Installation

1. Clone this repository
2. Install dependencies using `uv` (recommended) or `pip`

### Using `uv` (recommended)

1. Install `uv` from [uv's documentation](https://docs.astral.sh/uv/)
2. Run the application:
   ```bash
   # For GUI version
   ./run.sh
   
   # For CLI version
   ./run-cli.sh --help
   ```

### Using `just`

```bash
# Install just if you don't have it (macOS)
brew install just

# Setup environment and install dependencies
just quickstart
```

## Usage

### GUI Application

Run the GUI application:

```bash
just run-gui
```

or

```bash
python gui.py
```

With the GUI, you can:
- Drag and drop a pixel art image onto the window
- Choose downscaling and upscaling options
- Create clean versions at original size or custom sizes
- Preview the result in the app window

### Command Line Interface

Process an image using the CLI:

```bash
just process /path/to/your/pixel-art.png
```

Or run manually:

```bash
python cli.py /path/to/your/pixel-art.png
```

Optional parameters:
- `--scale N`: Force a specific downscale factor instead of auto-detection
- `--upscale N`: Factor to upscale after downscaling
- `--no-upscale`: Skip creating an upscaled version
- `--custom-upscale N`: Create additional upscaled version with this factor

## Why Use Pixel Art Scaler

When pixel art is shared on social media platforms, the original crisp pixel edges are often lost due to:
1. Compression artifacts (especially in JPG format)
2. Platform-specific image processing
3. Loss of the original pixel grid alignment

This app restores pixel art to its intended appearance by:
1. Detecting the original pixel scale using frequency analysis
2. Downscaling to true 1:1 pixel ratio to remove artifacts
3. Re-upscaling with nearest neighbor interpolation for perfect crisp pixels

The result is perfect for sharing on social media without losing the distinctive pixel art aesthetic.

## How it works

The app uses multiple strategies to detect the pixel art scale:
1. **Frequency domain analysis** using Fast Fourier Transform (FFT) to detect repeating patterns
2. **Edge alignment analysis** to find where color changes align with grid boundaries
3. **Uniform block detection** to identify pixel clusters of the same color

After detecting the scale, it:
1. Downscales the image to a true 1:1 pixel ratio using averaging
2. Re-upscales using nearest neighbor interpolation to maintain crisp pixel boundaries

## Requirements

- Python 3.8+
- Pillow
- NumPy
- SciPy
- PySide6 (for GUI version)