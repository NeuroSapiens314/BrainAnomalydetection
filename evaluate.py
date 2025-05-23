"""
Script for evaluating trained models
"""
import argparse
from pathlib import Path

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score

from processors.dicom_processor import DicomProcessor
from processors.image_processor import ImageProcessor

def load_and_prepare_data(study_path: Path, target_h: int = 256, target_w: int = 256,
                         expected_slices: int = 16):
    """Loads and prepares DICOM data for prediction."""
    # Read DICOM series
    series = DicomProcessor.read_dicom_series(study_path)
    
    # Apply windowing
    windowed_series = ImageProcessor.apply_windowing_using_header_on_series(
        series['array'],
        series['headers']
    )
    
    # Resize if needed
    if (windowed_series.shape[1] != target_h) or (windowed_series.shape[2] != target_w):
        arr = ImageProcessor.resize_and_pad(windowed_series, target_h, target_w)
    else:
        arr = windowed_series
    
    # Adjust number of slices
    if len(arr) != expected_slices:
        arr = ImageProcessor.adjust_slices(arr, expected_slices)
    
    return np.expand_dims(arr, axis=(0, -1))

def evaluate_model(model_path: Path, data_dir: Path, predictions_file_path: Path):
    """Evaluates model on test data."""
    # Load model
    model = tf.keras.models.load_model(str(model_path))
    
    # Get list of studies
    series_instance_uid_list = [i.name for i in data_dir.iterdir() if i.is_dir()]
    
    # Make predictions
    predictions = []
    for siuid in series_instance_uid_list:
        study_path = data_dir / siuid
        prepared_data = load_and_prepare_data(study_path)
        prediction = model.predict(prepared_data)[0][0]
        predictions.append(prediction)
    
    # Save predictions
    predictions_df = pd.DataFrame({
        'SeriesInstanceUID': series_instance_uid_list,
        'prediction': predictions
    })
    predictions_df.to_csv(predictions_file_path, index=False)
    
    # If ground truth is available, calculate metrics
    labels_path = data_dir / 'labels.csv'
    if labels_path.exists():
        labels_df = pd.read_csv(labels_path)
        merged_df = predictions_df.merge(labels_df, on='SeriesInstanceUID')
        
        y_true = merged_df['abnormal']
        y_pred = (merged_df['prediction'] > 0.5).astype(int)
        y_score = merged_df['prediction']
        
        metrics = {
            'auc': roc_auc_score(y_true, y_score),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        }
        
        print("\nEvaluation Metrics:")
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper()}: {value:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate MRI classification model')
    parser.add_argument('--model-path', type=Path, required=True,
                      help='Path to saved model')
    parser.add_argument('--data-dir', type=Path, required=True,
                      help='Directory containing test data')
    parser.add_argument('--predictions-file', type=Path, required=True,
                      help='Path to save predictions CSV file')
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.data_dir, args.predictions_file)

if __name__ == '__main__':
    main() 