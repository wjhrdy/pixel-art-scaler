#!/usr/bin/env python3
import argparse
import os
import numpy as np
from PIL import Image
from scipy import fft, signal, ndimage

class PixelArtDownscaler:
    def __init__(self):
        pass
    
    def detect_pixel_scale(self, img):
        """
        Detect the pixel scale by analyzing the image for repeated patterns.
        Returns the detected scale (an integer representing how many pixels make up one 'art pixel').
        Also detects if the pixel grid has an offset from the top-left corner.
        """
        # Convert image to RGBA to ensure alpha channel is preserved
        img = img.convert("RGBA")
        
        # Get image dimensions
        width, height = img.size
        
        # First, check if there's a grid offset
        offset_x, offset_y, img_array = self._detect_grid_offset(img)
        
        # If there's an offset, crop the image to align with the grid
        if offset_x > 0 or offset_y > 0:
            # Crop to align with grid
            img = img.crop((offset_x, offset_y, width, height))
            # Update dimensions and array after cropping
            width, height = img.size
            img_array = np.array(img)
        else:
            # If no offset, just use the original image array
            img_array = np.array(img)
        
        # Use multiple detection methods and combine results
        scale_votes = {}
        
        # First method: Autocorrelation (most accurate for pixel art)
        try:
            autocorr_scale = self._detect_scale_with_autocorrelation(img_array)
            if autocorr_scale > 1:
                scale_votes[autocorr_scale] = scale_votes.get(autocorr_scale, 0) + 3  # Weight: highest
                print(f"Autocorrelation suggests scale: {autocorr_scale}")
        except Exception as e:
            print(f"Autocorrelation failed: {str(e)}")
        
        # Second method: Color quantization and clustering
        try:
            color_scale = self._detect_scale_with_color_clustering(img_array)
            if color_scale > 1:
                scale_votes[color_scale] = scale_votes.get(color_scale, 0) + 2  # Weight: medium
                print(f"Color clustering suggests scale: {color_scale}")
        except Exception as e:
            print(f"Color clustering failed: {str(e)}")
            
        # Third method: FFT frequency analysis
        try:
            fft_scale = self._detect_scale_with_fft(img)
            if fft_scale > 1:
                scale_votes[fft_scale] = scale_votes.get(fft_scale, 0) + 2  # Weight: medium
                print(f"FFT analysis suggests scale: {fft_scale}")
        except Exception as e:
            print(f"FFT analysis failed: {str(e)}")
        
        # Fourth method: Edge detection
        try:
            edge_scale = self._detect_scale_with_edges(img_array)
            if edge_scale > 1:
                scale_votes[edge_scale] = scale_votes.get(edge_scale, 0) + 1  # Weight: low
                print(f"Edge detection suggests scale: {edge_scale}")
        except Exception as e:
            print(f"Edge detection failed: {str(e)}")
            
        # Fifth method: Uniform block analysis
        try:
            block_scale = self._detect_scale_with_blocks(img_array)
            if block_scale > 1:
                scale_votes[block_scale] = scale_votes.get(block_scale, 0) + 1  # Weight: low
                print(f"Block analysis suggests scale: {block_scale}")
        except Exception as e:
            print(f"Block analysis failed: {str(e)}")
        
        # Find the scale with the highest vote count
        if scale_votes:
            best_scale = max(scale_votes.items(), key=lambda x: x[1])[0]
            return best_scale
            
        # If all methods fail, check if dimensions can help
        if width > 0 and height > 0:
            # For very small images, we should default to larger scales
            # Small pixel art is often rendered at larger scales
            
            total_pixels = width * height
            if total_pixels <= 100:  # 10x10 or smaller
                print("Very small image detected, defaulting to 10x scale")
                return 10
            elif total_pixels <= 400:  # 20x20 or smaller
                print("Small image detected, defaulting to 8x scale")
                return 8
            elif total_pixels <= 1600:  # 40x40 or smaller
                print("Medium-small image detected, defaulting to 6x scale")
                return 6
            else:
                print("Medium image detected, defaulting to 4x scale")
                return 4
                    
        # Default to 4 if no pattern is found (common for pixel art)
        print("No clear scale pattern found, using default 4x scale")
        return 4
        
    def _detect_scale_with_autocorrelation(self, img_array):
        """
        Detect scale using autocorrelation in spatial domain.
        This method is very effective for pixel art with repeating patterns.
        """
        # Use grayscale to simplify
        if img_array.ndim == 3:
            # If RGB/RGBA, convert to grayscale
            if img_array.shape[2] >= 3:
                # Use luminance formula: 0.299R + 0.587G + 0.114B
                gray = np.dot(img_array[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
            else:
                gray = img_array[..., 0]  # Just use first channel
        else:
            gray = img_array
            
        # Get dimensions
        height, width = gray.shape
        
        # Limit analysis to a reasonable center portion for performance
        center_h, center_w = height // 2, width // 2
        max_offset = min(width, height) // 2
        max_offset = min(max_offset, 64)  # Cap for performance
        
        # Candidate scales and their scores
        scale_scores = {}
        
        # Check both horizontal and vertical autocorrelation
        for direction in ['horizontal', 'vertical']:
            # For each potential scale
            for scale in range(2, max_offset):
                score = 0
                samples = 0
                
                if direction == 'horizontal':
                    # Check a band of rows in the center
                    for y in range(center_h - 10, center_h + 10):
                        if y < 0 or y >= height - scale:
                            continue
                            
                        # Compare pixels with their neighbors at distance 'scale'
                        for x in range(width - scale):
                            # If pixels match across the scale distance
                            if gray[y, x] == gray[y, x + scale]:
                                score += 1
                            samples += 1
                else:  # vertical
                    # Check a band of columns in the center
                    for x in range(center_w - 10, center_w + 10):
                        if x < 0 or x >= width - scale:
                            continue
                            
                        # Compare pixels with their neighbors at distance 'scale'
                        for y in range(height - scale):
                            # If pixels match across the scale distance
                            if gray[y, x] == gray[y + scale, x]:
                                score += 1
                            samples += 1
                
                # Calculate match ratio if we have samples
                if samples > 0:
                    match_ratio = score / samples
                    # Store the score for this scale
                    scale_scores[scale] = scale_scores.get(scale, 0) + match_ratio
        
        # Find the scale with the highest score
        if scale_scores:
            max_score = 0
            best_scale = 0
            
            for scale, score in scale_scores.items():
                # Prefer scales that are common in pixel art (powers of 2, etc.)
                scale_preference = 1.0
                if scale in [2, 4, 8, 16, 32]:
                    scale_preference = 1.2
                elif scale in [3, 6, 12, 24]:
                    scale_preference = 1.1
                
                adjusted_score = score * scale_preference
                
                if adjusted_score > max_score:
                    max_score = adjusted_score
                    best_scale = scale
                    
            if best_scale > 0:
                return best_scale
        
        # Analyze using signal processing approach if the direct method fails
        try:
            # Compute autocorrelation - this is more robust but slower
            # Do it for both horizontal and vertical directions
            h_autocorr = np.zeros(max_offset)
            v_autocorr = np.zeros(max_offset)
            
            # Horizontal autocorrelation - average across rows
            for y in range(center_h - 10, center_h + 10):
                if y >= 0 and y < height:
                    row = gray[y, :]
                    row_autocorr = signal.correlate(row, row, mode='same')
                    # Take only the right half (positive lags)
                    h_autocorr += row_autocorr[len(row_autocorr)//2:len(row_autocorr)//2+max_offset]
            
            # Vertical autocorrelation - average across columns
            for x in range(center_w - 10, center_w + 10):
                if x >= 0 and x < width:
                    col = gray[:, x]
                    col_autocorr = signal.correlate(col, col, mode='same')
                    # Take only the right half (positive lags)
                    v_autocorr += col_autocorr[len(col_autocorr)//2:len(col_autocorr)//2+max_offset]
            
            # Find peaks in the autocorrelation
            h_peaks, _ = signal.find_peaks(h_autocorr, distance=2)
            v_peaks, _ = signal.find_peaks(v_autocorr, distance=2)
            
            # Get the first significant peak (if any)
            h_scale = h_peaks[0] if len(h_peaks) > 0 else 0
            v_scale = v_peaks[0] if len(v_peaks) > 0 else 0
            
            # Use the most convincing direction, or average if both are good
            if h_scale > 0 and v_scale > 0:
                scale = (h_scale + v_scale) // 2
            elif h_scale > 0:
                scale = h_scale
            elif v_scale > 0:
                scale = v_scale
            else:
                scale = 0
                
            if scale > 1:
                return scale
        except Exception:
            pass
            
        # No clear scale detected
        return 0
        
    def _detect_scale_with_color_clustering(self, img_array):
        """
        Detect scale by analyzing color clusters in the image.
        Pixel art typically has areas of solid color.
        """
        height, width = img_array.shape[:2]
        
        # Skip if image is too small
        if width < 8 or height < 8:
            return 0
            
        # For each potential scale, check if pixels form clear clusters
        max_scale = min(width, height) // 4
        max_scale = min(max_scale, 16)  # Cap at reasonable value
        
        best_scale = 0
        best_cluster_ratio = 0
        
        for scale in range(2, max_scale + 1):
            # Skip scales that don't divide evenly
            if width % scale != 0 or height % scale != 0:
                continue
                
            # Count uniform color clusters
            total_blocks = 0
            uniform_blocks = 0
            
            # Use a stride for efficiency on larger images
            stride = max(1, scale // 2)
            
            # Check blocks across the image
            for y in range(0, height - scale + 1, stride):
                for x in range(0, width - scale + 1, stride):
                    # Get the block
                    block = img_array[y:y+scale, x:x+scale]
                    
                    # Check if all pixels in the block have the same color
                    if block.ndim == 3:  # Color image with channels
                        ref_color = tuple(block[0, 0])
                        is_uniform = True
                        
                        for by in range(scale):
                            for bx in range(scale):
                                if not np.array_equal(block[by, bx], ref_color):
                                    is_uniform = False
                                    break
                            if not is_uniform:
                                break
                    else:  # Grayscale
                        ref_color = block[0, 0]
                        is_uniform = np.all(block == ref_color)
                    
                    if is_uniform:
                        uniform_blocks += 1
                    total_blocks += 1
            
            # Calculate ratio of uniform blocks
            if total_blocks > 0:
                cluster_ratio = uniform_blocks / total_blocks
                
                # Apply a bias toward common pixel art scales
                scale_preference = 1.0
                if scale in [2, 4, 8, 16]:
                    scale_preference = 1.2
                elif scale in [3, 6, 12]:
                    scale_preference = 1.1
                
                adjusted_ratio = cluster_ratio * scale_preference
                
                if adjusted_ratio > best_cluster_ratio:
                    best_cluster_ratio = adjusted_ratio
                    best_scale = scale
        
        # Return the scale if we found a convincing pattern
        if best_cluster_ratio > 0.2:  # Reasonable threshold
            return best_scale
            
        return 0
        
    def _detect_grid_offset(self, img):
        """
        Detect if the pixel art grid is offset from the top-left corner.
        Returns the detected offsets (x, y) and the numpy array of the image.
        """
        # Convert to array for analysis
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        
        # We'll search for grid alignments by finding where color transitions occur
        best_offset_x = 0
        best_offset_y = 0
        best_edge_count = 0
        
        # Only check a reasonable range for offsets
        max_offset = min(width, height) // 4
        max_offset = min(max_offset, 8)  # Cap at 8 pixels for performance
        
        # Sample stride to avoid checking every pixel (for performance)
        stride = max(1, min(width, height) // 50)
        
        # Check different potential offsets
        for offset_y in range(max_offset):
            for offset_x in range(max_offset):
                edge_count = 0
                
                # Check horizontal edges at this offset
                for y in range(offset_y, height - 1, stride):
                    for x in range(offset_x, width - 1, stride):
                        # Check for color transitions
                        if not np.array_equal(img_array[y, x], img_array[y+1, x]):
                            edge_count += 1
                            
                # Check vertical edges at this offset
                for x in range(offset_x, width - 1, stride):
                    for y in range(offset_y, height - 1, stride):
                        if not np.array_equal(img_array[y, x], img_array[y, x+1]):
                            edge_count += 1
                
                # If this offset results in more edges aligned with the grid,
                # it's likely the correct grid alignment
                if edge_count > best_edge_count:
                    best_edge_count = edge_count
                    best_offset_x = offset_x
                    best_offset_y = offset_y
        
        return best_offset_x, best_offset_y, img_array
    
    def _detect_scale_with_fft(self, img):
        """Detect scale using frequency domain analysis (FFT)."""
        # Convert to grayscale for FFT analysis
        gray_img = img.convert("L")
        img_array = np.array(gray_img)
        
        # Apply FFT
        fft_result = fft.fft2(img_array)
        
        # Shift zero frequency component to center
        fft_shifted = fft.fftshift(fft_result)
        
        # Get magnitude spectrum (log scale for better visualization)
        magnitude = np.abs(fft_shifted)
        magnitude_log = np.log1p(magnitude)
        
        # Find peaks in frequency domain (excluding the DC component at center)
        height, width = magnitude_log.shape
        center_y, center_x = height // 2, width // 2
        
        # Create a mask to ignore the central peak (DC component)
        mask = np.ones_like(magnitude_log, dtype=bool)
        mask_radius = min(width, height) // 20  # Adjust this radius as needed
        y_indices, x_indices = np.ogrid[:height, :width]
        mask_area = (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2 <= mask_radius ** 2
        mask[mask_area] = False
        
        # Apply mask and find highest peaks
        masked_magnitude = magnitude_log.copy()
        masked_magnitude[~mask] = 0
        
        # Find potential peaks
        peak_y, peak_x = np.unravel_index(np.argmax(masked_magnitude), magnitude_log.shape)
        
        # Calculate distances from center to peak (which gives frequency)
        dy = abs(peak_y - center_y)
        dx = abs(peak_x - center_x)
        
        # The dominant frequency corresponds to the scale
        # We need to convert from frequency to spatial domain
        if dy > dx and dy > 0:
            scale = height // dy
        elif dx > 0:
            scale = width // dx
        else:
            # No clear frequency pattern detected, default to 4
            scale = 4
            
        # Validate the scale
        if scale < 2 or scale > 16:
            # These are unlikely scales for pixel art, default to 4
            return 4
            
        return scale
        
    def _detect_scale_with_edges(self, img_array):
        """Detect scale by analyzing edges in the image."""
        height, width = img_array.shape[:2]
        
        # Calculate potential scales based on image dimensions
        max_scale = min(width, height) // 2
        if max_scale > 16:
            max_scale = 16  # Limit to reasonable pixel art scales
            
        scales_to_check = range(2, max_scale + 1)
        
        best_scale = 0
        best_score = 0
        
        # Check each potential scale
        for scale in scales_to_check:
            # Skip if dimensions aren't reasonably divisible by the scale
            if width % scale > scale // 2 or height % scale > scale // 2:
                continue
                
            # Count edges that align with this scale's grid
            edge_score = self._calculate_edge_score(img_array, scale)
            
            # Slight bias toward 4 as it's common in pixel art
            scale_bias = 1.2 if scale == 4 else 1.0
            adjusted_score = edge_score * scale_bias
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_scale = scale
                
        # Only return if we found a reasonable score
        if best_score > 0.2:
            return best_scale
        
        # No clear edge pattern
        return 0
        
    def _calculate_edge_score(self, img_array, scale):
        """Calculate how well edges align with the proposed scale grid."""
        height, width = img_array.shape[:2]
        
        edge_aligned = 0
        edge_count = 0
        
        # Optimize by sampling rather than checking every pixel
        # Use stride to sample the image more efficiently
        stride = max(1, min(width, height) // 100)
        
        # Check horizontal edges
        for y in range(1, height, stride):
            for x in range(0, width, stride):
                # If there's a color difference between adjacent pixels (vertical)
                if y < height - 1 and not np.array_equal(img_array[y, x], img_array[y-1, x]):
                    edge_count += 1
                    # Check if this edge falls on a scale boundary
                    if y % scale == 0:
                        edge_aligned += 1
                        
        # Check vertical edges
        for x in range(1, width, stride):
            for y in range(0, height, stride):
                # If there's a color difference between adjacent pixels (horizontal)
                if x < width - 1 and not np.array_equal(img_array[y, x], img_array[y, x-1]):
                    edge_count += 1
                    # Check if this edge falls on a scale boundary
                    if x % scale == 0:
                        edge_aligned += 1
        
        # Return ratio of aligned edges to total edges
        if edge_count > 0:
            return edge_aligned / edge_count
        return 0
    
    def _detect_scale_with_blocks(self, img_array):
        """Detect scale by looking for uniform blocks of pixels."""
        height, width = img_array.shape[:2]
        
        # Calculate potential scales based on image dimensions
        max_scale = min(width, height) // 2
        if max_scale > 16:
            max_scale = 16  # Limit to reasonable pixel art scales
            
        scales_to_check = range(2, max_scale + 1)
        
        best_scale = 0
        best_score = 0
        
        for scale in scales_to_check:
            # Skip if dimensions aren't reasonably divisible by the scale
            if width % scale > scale // 2 or height % scale > scale // 2:
                continue
                
            # Check for uniform blocks of this size
            block_score = self._calculate_block_score(img_array, scale)
            
            # Slight bias toward 4 as it's common in pixel art
            scale_bias = 1.2 if scale == 4 else 1.0
            adjusted_score = block_score * scale_bias
            
            if adjusted_score > best_score:
                best_score = adjusted_score
                best_scale = scale
                
        # Only return if we found a reasonable score
        if best_score > 0.15:
            return best_scale
        
        # No clear block pattern
        return 0
    
    def _calculate_block_score(self, img_array, scale):
        """Calculate how uniform blocks of the proposed scale are."""
        height, width = img_array.shape[:2]
        
        uniform_blocks = 0
        total_blocks = 0
        
        # Optimize by limiting the number of blocks we check for large images
        max_blocks_to_check = 500
        
        # Calculate step size for sampling blocks
        y_step = max(scale, (height // scale) // int(np.sqrt(max_blocks_to_check)))
        x_step = max(scale, (width // scale) // int(np.sqrt(max_blocks_to_check)))
        
        # Sample blocks across the image
        for y_start in range(0, height - scale + 1, y_step):
            for x_start in range(0, width - scale + 1, x_step):
                # Extract block
                block = img_array[y_start:y_start+scale, x_start:x_start+scale]
                
                # Check if block has uniform color by sampling
                center_y, center_x = scale // 2, scale // 2
                center_pixel = block[center_y, center_x]
                
                # First check corners
                corners = [
                    block[0, 0],              # Top-left
                    block[0, scale-1],        # Top-right
                    block[scale-1, 0],        # Bottom-left
                    block[scale-1, scale-1]   # Bottom-right
                ]
                
                is_uniform = all(np.array_equal(center_pixel, corner) for corner in corners)
                
                # If corners match, do a more detailed check
                if is_uniform and scale > 4:
                    # Sample a grid pattern
                    sample_step = max(1, scale // 4)
                    for y in range(0, scale, sample_step):
                        for x in range(0, scale, sample_step):
                            if not np.array_equal(block[y, x], center_pixel):
                                is_uniform = False
                                break
                        if not is_uniform:
                            break
                
                if is_uniform:
                    uniform_blocks += 1
                total_blocks += 1
        
        # Return ratio of uniform blocks to total blocks
        if total_blocks > 0:
            return uniform_blocks / total_blocks
        return 0
    
    def downscale_image(self, img, scale, color_threshold=15, use_median=False, ignore_outer_pixels=True):
        """
        Downscale the image by the given scale factor (can be fractional), grouping similar colors.
        
        Parameters:
        - img: Input PIL image
        - scale: Scale factor to reduce by (can be fractional, e.g., 4.5)
        - color_threshold: Maximum distance between colors to be considered the same group
        - use_median: If True, use median color instead of most frequent
        - ignore_outer_pixels: If True, ignore the outermost pixels when determining color
        """
        # Convert image to RGBA to ensure alpha channel is preserved
        img = img.convert("RGBA")
        
        width, height = img.size
        
        # Handle fractional scale factor
        # The new dimensions are original dimensions divided by scale
        new_width = int(width / scale)
        new_height = int(height / scale)
        
        # Create a new image for the result
        result = Image.new("RGBA", (new_width, new_height))
        
        # Convert to numpy array for efficient processing
        img_array = np.array(img)
        
        # Pre-process to reduce noise if dealing with JPEG or similar compression
        if color_threshold > 0:
            # Quick pass to smooth out JPEG artifacts within each channel
            for channel in range(3):  # RGB channels only, leave alpha alone
                img_array[:, :, channel] = ndimage.median_filter(img_array[:, :, channel], size=2)
        
        for y in range(new_height):
            for x in range(new_width):
                # Calculate the exact boundaries of this pixel in the original image
                # For fractional scales, we need to handle differently
                x_start = int(x * scale)
                y_start = int(y * scale)
                x_end = int((x + 1) * scale)
                y_end = int((y + 1) * scale)
                
                # Make sure we don't go out of bounds
                x_end = min(x_end, width)
                y_end = min(y_end, height)
                
                # Extract the block of pixels that make up this 'art pixel'
                block = img_array[y_start:y_end, x_start:x_end]
                
                if block.size == 0:  # Skip empty blocks
                    continue
                
                # For alpha channel, we need special handling
                alpha_values = block[:, :, 3]
                
                # If block is completely transparent, just use transparent pixel
                if np.all(alpha_values == 0):
                    result.putpixel((x, y), (0, 0, 0, 0))
                    continue
                
                # For pixels with some transparency, separate processing
                if np.any(alpha_values < 255):
                    # Handle semi-transparent blocks by using weighted average
                    # (fully transparent pixels shouldn't contribute to color)
                    weights = alpha_values.astype(float) / 255.0
                    total_weight = np.sum(weights)
                    
                    if total_weight > 0:
                        # Weighted color average (ignoring fully transparent pixels)
                        avg_r = np.sum(block[:, :, 0] * weights) / total_weight
                        avg_g = np.sum(block[:, :, 1] * weights) / total_weight
                        avg_b = np.sum(block[:, :, 2] * weights) / total_weight
                        avg_a = np.mean(alpha_values)
                        
                        result.putpixel((x, y), (int(avg_r), int(avg_g), int(avg_b), int(avg_a)))
                    else:
                        result.putpixel((x, y), (0, 0, 0, 0))
                    continue
                
                # For fully opaque blocks, use color quantization
                # Determine which pixels to consider for color selection
                block_height, block_width = block.shape[:2]
                
                if ignore_outer_pixels and block_width > 2 and block_height > 2:
                    # Ignore the outermost pixels for color determination
                    # Extract the inner block by removing the outer border
                    inner_block = block[1:-1, 1:-1, :3]
                    colors = inner_block.reshape(-1, 3)
                else:
                    # Use all pixels if block is too small or ignoring outer pixels is disabled
                    colors = block[:, :, :3].reshape(-1, 3)
                
                if use_median:
                    # Simple median color (less affected by outliers than mean)
                    median_color = np.median(colors, axis=0).astype(np.uint8)
                    result.putpixel((x, y), (median_color[0], median_color[1], median_color[2], 255))
                else:
                    # Group similar colors
                    if color_threshold > 0:
                        # This is our color clustering approach
                        clusters = self._cluster_similar_colors(colors, threshold=color_threshold)
                        
                        # Find the most frequent cluster
                        largest_cluster = max(clusters, key=len)
                        
                        # Use the centroid/average of the largest cluster
                        dominant_color = np.mean(largest_cluster, axis=0).astype(np.uint8)
                        result.putpixel((x, y), (dominant_color[0], dominant_color[1], dominant_color[2], 255))
                    else:
                        # Just use simple mean if no threshold is specified
                        avg_color = np.mean(colors, axis=0).astype(np.uint8)
                        result.putpixel((x, y), (avg_color[0], avg_color[1], avg_color[2], 255))
        
        return result
        
    def _cluster_similar_colors(self, colors, threshold=15):
        """
        Group colors that are within a certain distance threshold of each other.
        
        Parameters:
        - colors: Array of RGB color values
        - threshold: Maximum Euclidean distance to consider colors similar
        
        Returns:
        - List of clusters, where each cluster is a list of similar RGB colors
        """
        # For very small blocks, skip complex clustering
        if len(colors) <= 4:
            return [colors]
            
        # Initialize clusters with lists of np.array objects
        clusters = []
        
        # Convert colors to a numpy array if it's not already
        if not isinstance(colors, np.ndarray):
            colors = np.array(colors)
        
        # Process each color
        for color in colors:
            # Create a properly shaped color array
            color_array = np.array(color).reshape(1, -1)
            
            # Check if this color fits in any existing cluster
            found_cluster = False
            
            for cluster in clusters:
                # Calculate distance to cluster centroid
                # Ensure cluster is a numpy array
                cluster_array = np.array(cluster)
                centroid = np.mean(cluster_array, axis=0)
                
                # Handle different color shapes
                if color.ndim == 1:
                    # If color is a 1D array
                    distance = np.sqrt(np.sum((color - centroid)**2))
                else:
                    # If color is already a 2D array (unlikely but handle it)
                    distance = np.sqrt(np.sum((np.squeeze(color) - centroid)**2))
                
                if distance <= threshold:
                    # Add to this cluster (as a list)
                    cluster.append(color)
                    found_cluster = True
                    break
            
            # If not similar to any cluster, create a new one
            if not found_cluster:
                clusters.append([color])  # Start with a list containing this color
        
        # Merge overlapping clusters
        i = 0
        while i < len(clusters) and len(clusters) > 1:  # Only try merging if we have multiple clusters
            merged = False
            j = i + 1
            while j < len(clusters):
                # Ensure we're working with numpy arrays for calculation
                cluster_i_array = np.array(clusters[i])
                cluster_j_array = np.array(clusters[j])
                
                # Check if clusters i and j should be merged
                centroid_i = np.mean(cluster_i_array, axis=0)
                centroid_j = np.mean(cluster_j_array, axis=0)
                
                distance = np.sqrt(np.sum((centroid_i - centroid_j)**2))
                
                if distance <= threshold:
                    # Merge j into i (extend the list)
                    clusters[i].extend(clusters[j])
                    clusters.pop(j)
                    merged = True
                else:
                    j += 1
            
            if not merged:
                i += 1
        
        # Make sure we have at least one cluster (safety check)
        if not clusters:
            # If clustering failed, just return all colors as one cluster
            return [colors]
            
        return clusters
    
    def get_output_path(self, input_path, suffix="downscaled"):
        """Generate an output path for the processed image."""
        dir_name = os.path.dirname(input_path)
        file_name = os.path.basename(input_path)
        name, ext = os.path.splitext(file_name)
        
        output_name = f"{name}_{suffix}.png"
        output_path = os.path.join(dir_name, output_name)
        
        # Ensure we don't overwrite existing files
        counter = 1
        while os.path.exists(output_path):
            output_name = f"{name}_{suffix}_{counter}.png"
            output_path = os.path.join(dir_name, output_name)
            counter += 1
            
        return output_path
    
    def process_image(self, file_path, force_scale=None, upscale_factor=None, export_original_size=True, 
                  color_threshold=15, use_median=False, ignore_outer_pixels=True):
        """
        Process a single image file.
        
        Parameters:
        - file_path: Path to the image file
        - force_scale: Force a specific scale factor instead of auto-detecting (can be fractional)
        - upscale_factor: Factor to upscale after downscaling (using nearest neighbor)
        - export_original_size: Whether to export an image matching original dimensions
        - color_threshold: Threshold for color similarity when clustering (0-255)
        - use_median: Use median color instead of color clustering
        - ignore_outer_pixels: If True, ignore the outermost pixels when determining color
        """
        try:
            # Load the image
            img = Image.open(file_path)
            original_size = img.size
            
            # Use manually specified scale (can be fractional)
            pixel_scale = force_scale if force_scale else 1
            print(f"Using pixel scale: {pixel_scale}")
            
            if pixel_scale <= 1:
                print("No scaling needed. Image is already at 1:1 pixel ratio.")
                return None, None
            
            # Downscale the image to true 1:1 pixel ratio (removes compression artifacts)
            downscaled_img = self.downscale_image(img, pixel_scale, color_threshold, use_median, ignore_outer_pixels)
            
            # Add suffix based on settings used
            suffix = "downscaled"
            if use_median:
                suffix += "_median"
            elif color_threshold != 15:  # Only add suffix if not using default
                suffix += f"_t{color_threshold}"
            
            # Add fractional indicator if scale is not an integer
            if pixel_scale != int(pixel_scale):
                suffix += f"_f{pixel_scale:.2f}".replace('.', '_')
            
            # Add indicator for outer pixel handling
            if ignore_outer_pixels:
                suffix += "_io"  # io = ignore outer
            
            # Save the pure 1:1 pixel ratio version
            downscaled_path = self.get_output_path(file_path, suffix=suffix)
            downscaled_img.save(downscaled_path)
            
            # Log the color processing method used
            if use_median:
                print(f"Image successfully downscaled using median color and saved to: {downscaled_path}")
            else:
                print(f"Image successfully downscaled using color clustering (threshold={color_threshold}) and saved to: {downscaled_path}")
            
            # Additional info for fractional scales and pixel ignoring
            if pixel_scale != int(pixel_scale):
                print(f"Used fractional scale of {pixel_scale:.2f}")
            if ignore_outer_pixels:
                print("Ignored outermost pixels for color determination")

            # If no upscale factor is specified, but original size preservation is requested
            if upscale_factor is None and export_original_size:
                # Calculate what factor would restore original dimensions
                orig_width, orig_height = original_size
                down_width, down_height = downscaled_img.size
                width_factor = orig_width / down_width
                height_factor = orig_height / down_height
                # Use the average as the upscale factor (rounded to nearest integer)
                # For very small images, ensure scale is at least 4x
                if orig_width <= 32 or orig_height <= 32:
                    upscale_factor = max(round((width_factor + height_factor) / 2), 4)
                else:
                    upscale_factor = round((width_factor + height_factor) / 2)
                print(f"Auto-calculated upscale factor to match original size: {upscale_factor}x")
            
            # Clean version upscaled with nearest neighbor for social media
            if upscale_factor and upscale_factor > 1:
                # Upscale using nearest neighbor to keep pixel art look
                clean_width = downscaled_img.width * upscale_factor
                clean_height = downscaled_img.height * upscale_factor
                
                # Use nearest neighbor (NEAREST) for sharp pixel boundaries
                clean_img = downscaled_img.resize((clean_width, clean_height), Image.NEAREST)
                
                # Save the clean upscaled version
                clean_path = self.get_output_path(file_path, suffix=f"clean_{upscale_factor}x")
                clean_img.save(clean_path)
                print(f"Clean upscaled version saved to: {clean_path}")
                
                return downscaled_path, clean_path
            
            return downscaled_path, None
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
            return None, None

def main():
    parser = argparse.ArgumentParser(description='Clean up pixel art for social media by downscaling to true 1:1 ratio then upscaling with nearest neighbor')
    parser.add_argument('image_path', help='Path to the image file to process')
    parser.add_argument('--scale', type=float, help='Force a specific downscale factor instead of auto-detecting (can be fractional, e.g. 4.5)')
    parser.add_argument('--pixels', type=int, help='Number of pixels selected (used with --selection to calculate fractional scale)')
    parser.add_argument('--selection', type=int, help='Size of selection in pixels (used with --pixels to calculate fractional scale)')
    parser.add_argument('--upscale', type=int, help='Factor to upscale after downscaling (default: auto-calculated to match original size)')
    parser.add_argument('--no-upscale', action='store_true', help='Skip creating an upscaled version')
    parser.add_argument('--custom-upscale', type=int, help='Create additional upscaled version with this factor')
    parser.add_argument('--color-threshold', type=int, default=15, help='Color similarity threshold (0-255, default: 15). Higher values group more colors together')
    parser.add_argument('--use-median', action='store_true', help='Use median color instead of color clustering')
    parser.add_argument('--include-outer-pixels', action='store_true', help='Include outermost pixels when determining colors (by default they are ignored)')
    
    args = parser.parse_args()
    
    # Calculate fractional scale if pixels and selection are provided
    calculated_scale = None
    if args.pixels is not None and args.selection is not None and args.pixels > 0:
        calculated_scale = args.selection / args.pixels
        print(f"Calculated fractional scale: {calculated_scale:.2f} (selection size {args.selection} / {args.pixels} pixels)")
    
    # Use calculated scale if provided, otherwise use scale argument
    force_scale = calculated_scale if calculated_scale is not None else args.scale
    
    downscaler = PixelArtDownscaler()
    
    # Process with main options
    downscaler.process_image(
        args.image_path, 
        force_scale=force_scale,
        upscale_factor=args.upscale,
        export_original_size=not args.no_upscale,
        color_threshold=args.color_threshold,
        use_median=args.use_median,
        ignore_outer_pixels=not args.include_outer_pixels
    )
    
    # Process additional custom upscale if requested
    if args.custom_upscale and args.custom_upscale > 1:
        print(f"\nCreating additional {args.custom_upscale}x upscaled version:")
        downscaled_path, _ = downscaler.process_image(
            args.image_path,
            force_scale=force_scale,
            upscale_factor=args.custom_upscale,
            export_original_size=False,
            color_threshold=args.color_threshold,
            use_median=args.use_median,
            ignore_outer_pixels=not args.include_outer_pixels
        )

if __name__ == "__main__":
    main()