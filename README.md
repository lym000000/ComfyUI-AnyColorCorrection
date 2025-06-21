# ComfyUI Any Color Correction Node

This is a fork of [ComfyUI-ColorCorrection](https://github.com/zaheenrahman/ComfyUI-ColorCorrection) by [zaheenrahman](https://github.com/zaheenrahman)

A flexible custom node for **ComfyUI** that performs color correction on any masked region (e.g., clothing, background, skin) based on a reference image. Originally designed to fix color shifts during face-swapping. It now supports broader use cases with batch processing, optional masks, and multiple correction methods.

## 🔄 Changelog

### (2025-06-21)
- ✅ Added **batch dimension support** (works with 1-to-N and N-to-N inputs)
- ✅ Fully optional `mask_ref` and `mask_target` inputs
- ✅ Safer handling of shape mismatches (with proper error messages)
- ✅ Better normalized color difference metric (for thresholding)
- ✅ Ensured single-image input returns a single tensor output

## Features

- **Multiple Color Correction Methods**: Choose from statistical, histogram, or LAB color space transfers
- **Fine Control**: Adjust the strength and threshold of the color correction
- **Preserve Luminosity**: Option to maintain the original brightness while correcting colors
- **Masked Processing**: Only applies corrections to specified regions

## Installation

1. Navigate to your ComfyUI custom nodes directory:
   ```bash
   cd ComfyUI/custom_nodes/
   ```

2. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/ComfyUI_AnyColorCorrection
   ```

3. Install requirements:
   ```bash
   pip install -r ComfyUI_AnyColorCorrection/requirements.txt
   ```

4. Restart ComfyUI

## Usage

The node requires the following inputs:
## ⚙️ Node Inputs

| Input Name            | Type    | Required | Default | Description                                                                 |
|-----------------------|---------|----------|---------|-----------------------------------------------------------------------------|
| `image_ref`           | IMAGE   | ✅       | —       | Reference image to draw color information from.                            |
| `image_target`        | IMAGE   | ✅       | —       | Target image where color correction will be applied.                        |
| `mask_ref`            | MASK    | ❌       | None    | Optional mask for the reference image to localize color sampling.          |
| `mask_target`         | MASK    | ❌       | None    | Optional mask for the target image to localize correction.                 |
| `color_threshold`     | FLOAT   | ❌       | 0.15    | Minimum color difference to trigger correction. Range: 0.01 to 1.0.         |
| `strength`            | FLOAT   | ❌       | 0.8     | Strength of color correction. Range: 0.0 (off) to 1.0 (full correction).    |
| `preserve_luminosity` | BOOLEAN | ❌       | True    | Whether to preserve original brightness/luminosity in correction.           |
| `method`              | ENUM    | ❌       | lab_transfer | Method to use: `statistical`, `histogram`, or `lab_transfer`.         |


## Example Workflow

1. Load your reference and target images
2. Create masks for the areas in both images (using segmentation or manual masking)
3. Connect all inputs to the Color Correction node
4. Adjust parameters as needed

## Advanced Options

### Color Correction Methods

- **Statistical**: Uses mean and standard deviation to transfer color characteristics
- **Histogram**: Matches color distributions using histogram equalization
- **LAB Transfer**: Works in LAB color space for more perceptually accurate corrections

## Acknowledgements

This node was originally created by [zaheenrahman](https://github.com/zaheenrahman) to solve color shifts that commonly occur during face swapping, particularly when the original and swapped faces have different skin tones or when lighting conditions change. Extended to support more general, region-specific color transfer.


## License

[MIT License](https://opensource.org/licenses/MIT)