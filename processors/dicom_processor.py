"""
Module for processing DICOM files
"""
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk
import os
from pydicom.dataset import FileDataset

class DicomProcessor:
    """Class for processing DICOM medical image files."""
    
    def __init__(self, data_dir: str):
        """
        Initialize the DICOM processor.
        
        Args:
            data_dir (str): Directory containing DICOM files
        """
        self.data_dir = data_dir
        
    def load_scan(self, path: str) -> List[FileDataset]:
        """
        Load all DICOM files in the specified directory.
        
        Args:
            path (str): Path to directory containing DICOM files
            
        Returns:
            List[FileDataset]: List of DICOM datasets
        """
        slices = []
        for s in os.listdir(path):
            if s.endswith('.dcm'):
                ds = pydicom.dcmread(os.path.join(path, s))
                slices.append(ds)
        
        # Sort slices based on Z position
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        return slices
    
    def get_pixels_hu(self, scans: List[FileDataset]) -> np.ndarray:
        """
        Convert DICOM image pixels to Hounsfield units.
        
        Args:
            scans (List[FileDataset]): List of DICOM datasets
            
        Returns:
            np.ndarray: Array of pixel values in Hounsfield units
        """
        image = np.stack([s.pixel_array for s in scans])
        image = image.astype(float)
        
        # Convert to Hounsfield units (HU)
        for slice_number in range(len(scans)):
            intercept = scans[slice_number].RescaleIntercept
            slope = scans[slice_number].RescaleSlope
            
            if slope != 1:
                image[slice_number] = slope * image[slice_number].astype(float)
            image[slice_number] = image[slice_number].astype(float) + intercept
        
        return np.array(image, dtype=np.int16)
    
    def normalize_pixels(self, image: np.ndarray, min_bound: float = -1000.0, 
                       max_bound: float = 400.0) -> np.ndarray:
        """
        Normalize pixel values to [0, 1] range.
        
        Args:
            image (np.ndarray): Input image array
            min_bound (float): Minimum HU value for normalization
            max_bound (float): Maximum HU value for normalization
            
        Returns:
            np.ndarray: Normalized image array
        """
        image = (image - min_bound) / (max_bound - min_bound)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image
    
    def resample_volume(self, image: np.ndarray, scan: List[FileDataset], 
                       target_spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Resample the volume to a given voxel spacing.
        
        Args:
            image (np.ndarray): Input image volume
            scan (List[FileDataset]): List of DICOM datasets
            target_spacing (Tuple[float, float, float]): Target voxel spacing
            
        Returns:
            np.ndarray: Resampled image volume
        """
        # Determine current pixel spacing
        spacing = np.array([scan[0].SliceThickness] + list(scan[0].PixelSpacing), dtype=np.float32)
        
        resize_factor = spacing / np.array(target_spacing)
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize = new_shape / image.shape
        
        # Resize using scipy's interpolation
        from scipy.ndimage import zoom
        image = zoom(image, real_resize)
        
        return image
    
    def process_scan(self, patient_path: str, target_spacing: Optional[Tuple[float, float, float]] = None,
                    normalize: bool = True) -> np.ndarray:
        """
        Process a complete patient scan.
        
        Args:
            patient_path (str): Path to patient's DICOM directory
            target_spacing (Optional[Tuple[float, float, float]]): Target voxel spacing
            normalize (bool): Whether to normalize pixel values
            
        Returns:
            np.ndarray: Processed image volume
        """
        # Load scan
        scan = self.load_scan(patient_path)
        
        # Convert to Hounsfield units
        image = self.get_pixels_hu(scan)
        
        # Resample if target_spacing is provided
        if target_spacing is not None:
            image = self.resample_volume(image, scan, target_spacing)
        
        # Normalize if requested
        if normalize:
            image = self.normalize_pixels(image)
        
        return image

    @staticmethod
    def read_dicom_series(study_path: Path, series_instance_uid: Optional[str] = None) -> Dict[str, Any]:
        """Reads DICOM series and returns pixel array, headers and file paths."""
        if series_instance_uid is None:
            series_id = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(study_path))[0]
        else:
            series_id = series_instance_uid

        series_file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(study_path), series_id)
        headers = [pydicom.dcmread(str(fn), stop_before_pixels=True) for fn in series_file_names]

        volume = sitk.ReadImage(series_file_names, sitk.sitkInt32)
        volume = np.array(sitk.GetArrayFromImage(volume), dtype=np.float32)

        sorted_data = DicomProcessor._sort_dicom_data(headers, series_file_names)
        
        return {
            'array': volume,
            'headers': sorted_data['headers'],
            'dcm_paths': sorted_data['file_names']
        }

    @staticmethod
    def _sort_dicom_data(headers: List[pydicom.FileDataset], 
                         file_names: List[str]) -> Dict[str, List]:
        """Sorts DICOM data based on instance numbers."""
        slice_number_tag = DicomProcessor._get_slice_number_tag(headers)
        
        if slice_number_tag:
            sorted_headers = sorted(headers, key=lambda x: int(x.get(slice_number_tag)))
            file_name_to_index = {k: v for v, k in enumerate(file_names)}
            sorted_file_names = sorted(
                file_names,
                key=lambda x: int(headers[file_name_to_index[x]].get(slice_number_tag))
            )
        else:
            sorted_headers = headers
            sorted_file_names = file_names
            
        return {'headers': sorted_headers, 'file_names': sorted_file_names}

    @staticmethod
    def _get_slice_number_tag(headers: List[pydicom.FileDataset]) -> Optional[str]:
        """Determines the appropriate slice number tag."""
        if all(h.get('InstanceNumber') is not None for h in headers):
            return 'InstanceNumber'
        elif all(h.get('ImageNumber') is not None for h in headers):
            return 'ImageNumber'
        return None 