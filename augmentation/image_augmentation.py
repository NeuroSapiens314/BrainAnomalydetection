"""
Module for image augmentation
"""
import albumentations as A
import torchio as tio
import tensorflow as tf
import numpy as np

class ImageAugmentation:
    """Handles image augmentation operations."""
    
    @staticmethod
    def get_albumentation_transforms(height: int = 256, width: int = 256):
        """Returns Albumentations transformation pipeline."""
        return A.Compose([
            # Mild noise
            A.OneOf([
                A.GaussNoise(var_limit=(5.0, 20.0), p=0.6),
                A.MultiplicativeNoise(multiplier=(0.95, 1.05), p=0.6)
            ], p=0.6),

            # Blurring
            A.GaussianBlur(blur_limit=(3, 5), sigma_limit=(0.7, 1.2), p=0.7),

            # Geometric transforms
            A.Rotate(limit=(-15, 15), p=0.7),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(height=height, width=width, scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.6),
            A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=None, p=0.5),
            A.GridDistortion(num_steps=4, distort_limit=0.2, p=0.5),

            # Intensity transforms
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),

            # Final resize
            A.Resize(height=height, width=width, p=1)
        ], p=1)

    @staticmethod
    def get_torchio_transforms():
        """Returns TorchIO transformation pipeline."""
        return tio.Compose([
            tio.RescaleIntensity(out_min_max=(0, 1)),
            tio.RandomNoise(mean=0, std=0.05, p=0.3),
            tio.RandomBiasField(coefficients=0.5, p=0.3),
            tio.RandomAffine(scales=(0.9, 1.1), degrees=5, translation=5, p=0.5),
            tio.RandomElasticDeformation(num_control_points=5, max_displacement=3, p=0.3),
            tio.RandomFlip(axes=(0, 1), p=0.5)
        ])

    @staticmethod
    def tf_augmentation(image, label):
        """Applies TensorFlow-based augmentations."""
        # Rescale intensity
        image = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image))

        # Add random noise
        noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.05)
        image = image + noise

        # Random bias field
        bias = tf.random.uniform(shape=tf.shape(image), minval=0, maxval=0.5)
        image = image + bias

        # Random affine transformation
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)

        return image, label 