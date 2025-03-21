import numpy as np
import cv2
import torch
from PIL import Image
import os

class ClothingColorCorrectionNode:
    """
    A custom node for ComfyUI that performs color correction on clothing items.
    
    This node compares the colors between original clothing and a face-swapped output,
    then adjusts the colors to maintain the original clothing appearance while preserving
    the face swap.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),  # Original clothing image
                "clothing_mask": ("MASK",),    # Mask for clothing in original
                "output_image": ("IMAGE",),    # Output/face-swapped image
                "output_mask": ("MASK",),      # Mask for clothing area in output
                "color_threshold": ("FLOAT", {"default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "preserve_luminosity": ("BOOLEAN", {"default": True}),
                "method": (["statistical", "histogram", "lab_transfer"], {"default": "lab_transfer"})
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "correct_colors"
    CATEGORY = "image/color_correction"
    DESCRIPTION = "Corrects colors of clothing in face-swapped images to match original item colors"
    URL = {
        "GitHub Repository": "https://github.com/zaheenrahman/ComfyUI_ClothingColorCorrection",
        "Example Usage": "https://github.com/zaheenrahman/ComfyUI_ClothingColorCorrection/examples"
    }
    
    def correct_colors(self, original_image, clothing_mask, output_image, output_mask, 
                       color_threshold, strength, preserve_luminosity, method):
        """
        Corrects colors in the output image to match the original clothing colors
        
        Args:
            original_image: Original clothing image tensor (B, H, W, C)
            clothing_mask: Mask for clothing in original (B, H, W, 1)
            output_image: Output/face-swapped image tensor (B, H, W, C)
            output_mask: Mask for clothing area in output (B, H, W, 1)
            color_threshold: Threshold for determining if color correction is needed
            strength: Strength of the color correction effect (0.0 to 1.0)
            preserve_luminosity: Whether to preserve luminosity during color correction
            method: Color transfer method to use
            
        Returns:
            Corrected image tensor
        """
        # Convert tensors to numpy arrays for OpenCV processing
        device = original_image.device
        
        # Handle batch dimension
        results = []
        for b in range(len(original_image)):
            # Get single batch item
            orig_img = original_image[b].cpu().numpy()
            orig_mask = clothing_mask[b].cpu().numpy()
            out_img = output_image[b].cpu().numpy()
            out_mask = output_mask[b].cpu().numpy()
            
            # Convert to uint8 for OpenCV
            orig_img_uint8 = (np.clip(orig_img * 255.0, 0, 255)).astype(np.uint8)
            out_img_uint8 = (np.clip(out_img * 255.0, 0, 255)).astype(np.uint8)
            
            # Ensure masks are binary and in the right format
            orig_mask_bin = (orig_mask > 0.5).astype(np.uint8) * 255
            out_mask_bin = (out_mask > 0.5).astype(np.uint8) * 255
            
            # Apply masks to isolate clothing regions
            orig_masked = cv2.bitwise_and(orig_img_uint8, orig_img_uint8, mask=orig_mask_bin)
            out_masked = cv2.bitwise_and(out_img_uint8, out_img_uint8, mask=out_mask_bin)
            
            # Extract color statistics from masked regions
            orig_stats = self._get_masked_color_stats(orig_img_uint8, orig_mask_bin)
            out_stats = self._get_masked_color_stats(out_img_uint8, out_mask_bin)
            
            # Calculate color difference
            color_diff = self._calculate_color_difference(orig_stats, out_stats)
            
            # Create result image (start with output image)
            result_img = out_img_uint8.copy()
            
            # If difference exceeds threshold, perform correction
            if color_diff > color_threshold:
                if method == "statistical":
                    corrected = self._match_colors_statistical(
                        out_img_uint8, out_mask_bin, orig_stats, out_stats, strength, preserve_luminosity
                    )
                elif method == "histogram":
                    corrected = self._match_colors_histogram(
                        out_img_uint8, orig_img_uint8, out_mask_bin, strength
                    )
                elif method == "lab_transfer":
                    corrected = self._match_colors_lab(
                        out_img_uint8, orig_img_uint8, out_mask_bin, orig_mask_bin, strength, preserve_luminosity
                    )
                else:
                    corrected = out_img_uint8  # Fallback
                
                # Apply the corrected colors only to the masked region
                mask_3ch = cv2.merge([out_mask_bin, out_mask_bin, out_mask_bin])
                mask_inv = cv2.bitwise_not(mask_3ch)
                bg = cv2.bitwise_and(result_img, mask_inv)
                fg = cv2.bitwise_and(corrected, mask_3ch)
                result_img = cv2.add(bg, fg)
            
            # Convert back to float32 and tensor
            result_float = result_img.astype(np.float32) / 255.0
            results.append(torch.from_numpy(result_float))
        
        # Stack results back into a batch tensor
        return (torch.stack(results).to(device),)
    
    def _get_masked_color_stats(self, image, mask):
        """
        Extract color statistics from the masked area of an image
        
        Args:
            image: A numpy array of shape (H, W, 3) with RGB values
            mask: A binary mask of shape (H, W, 1) where non-zero indicates areas to analyze
            
        Returns:
            Dictionary containing color statistics (mean, std) for each channel
        """
        # Make sure mask is single channel
        if len(mask.shape) > 2 and mask.shape[2] > 1:
            mask = mask[:, :, 0]
        
        # Get non-zero mask pixels coordinates
        y, x = np.where(mask > 0)
        
        # If mask is empty, return zeros
        if len(y) == 0:
            return {
                'mean': np.zeros(3),
                'std': np.zeros(3)
            }
        
        # Extract pixel values at mask locations
        pixels = image[y, x]
        
        # Calculate statistics
        mean = pixels.mean(axis=0)
        std = pixels.std(axis=0)
        
        return {
            'mean': mean,
            'std': std
        }
    
    def _calculate_color_difference(self, stats1, stats2):
        """
        Calculate the color difference between two sets of color statistics
        
        Args:
            stats1: Color statistics dictionary for first image
            stats2: Color statistics dictionary for second image
            
        Returns:
            A floating point value representing color difference (0.0 to 1.0)
        """
        # Simple Euclidean distance between means, normalized
        mean_diff = np.sqrt(np.sum((stats1['mean'] - stats2['mean'])**2)) / (255 * np.sqrt(3))
        
        # Difference in standard deviations (color variability)
        std_diff = np.sqrt(np.sum((stats1['std'] - stats2['std'])**2)) / (255 * np.sqrt(3))
        
        # Weighted combination
        return 0.7 * mean_diff + 0.3 * std_diff
    
    def _match_colors_statistical(self, target_img, target_mask, source_stats, target_stats, 
                                strength, preserve_luminosity):
        """
        Match colors using statistical color transfer
        
        Args:
            target_img: Image to adjust
            target_mask: Mask of the region to adjust
            source_stats: Statistics of the source colors
            target_stats: Statistics of the target colors
            strength: Strength of adjustment (0.0 to 1.0)
            preserve_luminosity: Whether to preserve luminosity
            
        Returns:
            Color corrected image
        """
        # Create a copy of the target image
        result = target_img.copy()
        
        # If preserving luminosity, convert to LAB color space
        if preserve_luminosity:
            # Convert to LAB
            result_lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
            
            # Only adjust a and b channels (chromaticity)
            for i in range(1, 3):  # LAB channels 1 and 2 (a and b)
                # Calculate adjustment
                mean_s = source_stats['mean'][i]
                mean_t = target_stats['mean'][i]
                std_s = source_stats['std'][i]
                std_t = target_stats['std'][i]
                
                # Apply adjustment with strength factor
                if std_t > 0:
                    result_lab[:, :, i] = ((result_lab[:, :, i] - mean_t) * (std_s / std_t) * strength + 
                                          mean_t * (1 - strength) + mean_s * strength).clip(0, 255)
            
            # Convert back to RGB
            result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        else:
            # Adjust all channels
            for i in range(3):  # RGB channels
                mean_s = source_stats['mean'][i]
                mean_t = target_stats['mean'][i]
                std_s = source_stats['std'][i]
                std_t = target_stats['std'][i]
                
                # Apply adjustment with strength factor
                if std_t > 0:
                    result[:, :, i] = ((result[:, :, i] - mean_t) * (std_s / std_t) * strength + 
                                      mean_t * (1 - strength) + mean_s * strength).clip(0, 255)
        
        return result
    
    def _match_colors_histogram(self, target_img, source_img, target_mask, strength):
        """
        Match colors using histogram matching
        
        Args:
            target_img: Image to adjust
            source_img: Reference image with desired colors
            target_mask: Mask of the region to adjust
            strength: Strength of adjustment (0.0 to 1.0)
            
        Returns:
            Color corrected image
        """
        # Create a copy of the target image
        result = target_img.copy()
        
        # Convert to HSV for better color manipulation
        target_hsv = cv2.cvtColor(target_img, cv2.COLOR_RGB2HSV)
        source_hsv = cv2.cvtColor(source_img, cv2.COLOR_RGB2HSV)
        result_hsv = target_hsv.copy()
        
        # Apply histogram matching separately for each channel
        for i in range(3):  # HSV channels
            if i == 0:  # Special handling for Hue channel
                # Hue is circular, require special handling
                source_hist, _ = np.histogram(source_hsv[:, :, i], 256, [0, 256])
                target_hist, _ = np.histogram(target_hsv[:, :, i], 256, [0, 256])
                
                # Cumulative distribution functions
                source_cdf = source_hist.cumsum() / source_hist.sum()
                target_cdf = target_hist.cumsum() / target_hist.sum()
                
                # Map target to source through CDFs
                mapping = np.zeros(256, dtype=np.uint8)
                for j in range(256):
                    mapping[j] = np.argmin(np.abs(source_cdf - target_cdf[j]))
                
                # Apply mapping with strength factor
                original = target_hsv[:, :, i].copy()
                matched = mapping[target_hsv[:, :, i]]
                result_hsv[:, :, i] = (original * (1 - strength) + matched * strength).astype(np.uint8)
            else:
                # For S and V channels, standard histogram matching
                source_hist, _ = np.histogram(source_hsv[:, :, i].flatten(), 256, [0, 256])
                target_hist, _ = np.histogram(target_hsv[:, :, i].flatten(), 256, [0, 256])
                
                source_cdf = source_hist.cumsum() / source_hist.sum()
                target_cdf = target_hist.cumsum() / target_hist.sum()
                
                # Map target to source through CDFs
                mapping = np.zeros(256, dtype=np.uint8)
                for j in range(256):
                    mapping[j] = np.argmin(np.abs(source_cdf - target_cdf[j]))
                
                # Apply mapping with strength factor
                original = target_hsv[:, :, i].copy()
                matched = mapping[target_hsv[:, :, i]]
                result_hsv[:, :, i] = (original * (1 - strength) + matched * strength).astype(np.uint8)
        
        # Convert back to RGB
        result = cv2.cvtColor(result_hsv, cv2.COLOR_HSV2RGB)
        
        return result
    
    def _match_colors_lab(self, target_img, source_img, target_mask, source_mask, strength, preserve_luminosity):
        """
        Match colors using LAB color space transfer
        
        Args:
            target_img: Image to adjust
            source_img: Reference image with desired colors
            target_mask: Mask of the region to adjust
            source_mask: Mask of the region in source to sample colors from
            strength: Strength of adjustment (0.0 to 1.0)
            preserve_luminosity: Whether to preserve original luminosity
            
        Returns:
            Color corrected image
        """
        # Create a copy of the target image
        result = target_img.copy()
        
        # Convert to LAB color space
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_RGB2LAB)
        source_lab = cv2.cvtColor(source_img, cv2.COLOR_RGB2LAB)
        result_lab = target_lab.copy()
        
        # Get masked regions
        y_t, x_t = np.where(target_mask > 0)
        y_s, x_s = np.where(source_mask > 0)
        
        # If either mask is empty, return original
        if len(y_t) == 0 or len(y_s) == 0:
            return result
        
        # Extract LAB values for masked regions
        target_lab_values = target_lab[y_t, x_t]
        source_lab_values = source_lab[y_s, x_s]
        
        # Calculate statistics for each channel
        target_means = np.mean(target_lab_values, axis=0)
        target_stds = np.std(target_lab_values, axis=0)
        source_means = np.mean(source_lab_values, axis=0)
        source_stds = np.std(source_lab_values, axis=0)
        
        # Apply color transfer
        for i in range(3):  # LAB channels
            # Skip L channel if preserving luminosity
            if i == 0 and preserve_luminosity:
                continue
                
            # Calculate adjustment
            if target_stds[i] > 0:
                factor = (source_stds[i] / target_stds[i]) * strength
                offset = source_means[i] * strength + target_means[i] * (1 - strength) - target_means[i] * factor
                
                # Apply transformation
                channel = result_lab[:, :, i].astype(np.float32)
                channel = channel * factor + offset
                result_lab[:, :, i] = np.clip(channel, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB)
        
        return result

# Node display name
NODE_CLASS_MAPPINGS = {
    "ClothingColorCorrection": ClothingColorCorrectionNode
}

# Display name for the node in UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "ClothingColorCorrection": "Clothing Color Correction"
} 