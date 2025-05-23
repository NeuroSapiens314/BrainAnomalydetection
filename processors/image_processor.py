"""
Module for image processing operations
"""
from typing import List, Tuple, Optional, Union

import numpy as np
import pydicom
import tensorflow as tf
import cv2

class ImageProcessor:
    """Handles image processing operations."""
    
    @staticmethod
    def apply_windowing(series: np.ndarray,
                       window_center: int,
                       window_width: int) -> np.ndarray:
        """Returns an array for given window.

        Args:
            series: numpy array of shape (n_slices, h, w) or (h, w) in haunsfield units.
            window_center: for example, brain window's center is 40
            window_width: for example, brain window's width is 80

        Returns:
            numpy array of shape (n_sclies, h, w) or (h, w) in range(0, 1)
        """
        w_min = int(window_center - (window_width / 2))
        w_max = int(window_center + (window_width / 2))

        clipped = np.clip(series, w_min, w_max)
        windowed = (clipped - w_min) / (w_max - w_min)

        return windowed

    @staticmethod
    def apply_windowing_using_header_on_series(series: np.ndarray, 
                                              headers: List[pydicom.FileDataset]) -> np.ndarray:
        """Applies windowing based on DICOM header information.

        Args:
            series: numpy array of shape (num_slices, h, w) in haunsfield units.
            headers: dicom header containing WindowCenter and WindowWidth

        Returns:
            numpy array of shape (h, w) in range(0, 1)
        """
        windowed_series = []
        for i, header in enumerate(headers):
            window_center = header.get('WindowCenter')
            window_width = header.get('WindowWidth')
            windowed_series.append(
                ImageProcessor.apply_windowing(series[i], window_center, window_width)
            )

        return np.array(windowed_series)

    @staticmethod
    def resize_and_pad(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        """Resizes and pads image to target dimensions."""
        if (image.shape[1] != target_h) or (image.shape[2] != target_w):
            return np.squeeze(tf.image.resize_with_pad(
                np.expand_dims(image, axis=-1), 
                target_h, 
                target_w
            ))
        return image

    @staticmethod
    def adjust_slices(image: np.ndarray, expected_slices: int) -> np.ndarray:
        """Adjusts number of slices to match expected count."""
        if len(image) != expected_slices:
            remainder = len(image) - expected_slices
            return image[int(remainder / 2): expected_slices + int(remainder / 2)]
        return image

    @staticmethod
    def apply_window(image: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
        """
        Apply windowing to the image.
        
        Args:
            image (np.ndarray): Input image
            window_center (float): Window center in Hounsfield units
            window_width (float): Window width in Hounsfield units
            
        Returns:
            np.ndarray: Windowed image
        """
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        window_image = np.clip(image, img_min, img_max)
        
        return ((window_image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    
    @staticmethod
    def resize_image(image: np.ndarray, target_size: Tuple[int, int],
                    interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image to target size.
        
        Args:
            image (np.ndarray): Input image
            target_size (Tuple[int, int]): Target size (height, width)
            interpolation (int): Interpolation method
            
        Returns:
            np.ndarray: Resized image
        """
        return cv2.resize(image, target_size, interpolation=interpolation)
    
    @staticmethod
    def normalize_intensity(image: np.ndarray, out_min: float = 0.0,
                          out_max: float = 1.0) -> np.ndarray:
        """
        Normalize image intensity to specified range.
        
        Args:
            image (np.ndarray): Input image
            out_min (float): Minimum output value
            out_max (float): Maximum output value
            
        Returns:
            np.ndarray: Normalized image
        """
        if image.max() == image.min():
            return np.zeros_like(image)
        
        normalized = (image - image.min()) / (image.max() - image.min())
        normalized = normalized * (out_max - out_min) + out_min
        return normalized
    
    @staticmethod
    def apply_clahe(image: np.ndarray, clip_limit: float = 2.0,
                    tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
        
        Args:
            image (np.ndarray): Input image
            clip_limit (float): Threshold for contrast limiting
            tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization
            
        Returns:
            np.ndarray: CLAHE enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        if len(image.shape) == 3:
            # For RGB images, apply CLAHE to luminance channel
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # For grayscale images
            return clahe.apply(image)
    
    @staticmethod
    def extract_patches(image: np.ndarray, patch_size: Tuple[int, int],
                       stride: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Extract patches from image.
        
        Args:
            image (np.ndarray): Input image
            patch_size (Tuple[int, int]): Size of patches (height, width)
            stride (Optional[Tuple[int, int]]): Stride for patch extraction
            
        Returns:
            np.ndarray: Array of extracted patches
        """
        if stride is None:
            stride = patch_size
            
        h, w = image.shape[:2]
        ph, pw = patch_size
        sh, sw = stride
        
        patches = []
        for i in range(0, h - ph + 1, sh):
            for j in range(0, w - pw + 1, sw):
                patch = image[i:i + ph, j:j + pw]
                patches.append(patch)
                
        return np.array(patches)
    
    @staticmethod
    def augment_image(image: np.ndarray, rotation_range: float = 15.0,
                     zoom_range: Tuple[float, float] = (0.9, 1.1),
                     brightness_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """
        Apply random augmentation to image.
        
        Args:
            image (np.ndarray): Input image
            rotation_range (float): Maximum rotation angle in degrees
            zoom_range (Tuple[float, float]): Range for random zoom
            brightness_range (Tuple[float, float]): Range for random brightness adjustment
            
        Returns:
            np.ndarray: Augmented image
        """
        # Random rotation
        angle = np.random.uniform(-rotation_range, rotation_range)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h))
        
        # Random zoom
        zoom = np.random.uniform(zoom_range[0], zoom_range[1])
        h, w = image.shape[:2]
        zh = int(np.round(h * zoom))
        zw = int(np.round(w * zoom))
        top = (h - zh) // 2
        left = (w - zw) // 2
        
        image = cv2.resize(image, (zw, zh))
        if zoom < 1:
            # Padding
            pad_h = h - zh
            pad_w = w - zw
            image = cv2.copyMakeBorder(image, top, pad_h - top, left, pad_w - left,
                                     cv2.BORDER_CONSTANT, value=0)
        else:
            # Cropping
            image = image[top:top + h, left:left + w]
        
        # Random brightness
        brightness = np.random.uniform(brightness_range[0], brightness_range[1])
        image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
        
        return image 