# Brain MRI Anomaly Classification

This project aims to develop a deep learning pipeline to classify brain MRI scans into **normal** and **abnormal** categories. Abnormal cases include anomalies such as tumors, infections (e.g., sinusitis), or other brain irregularities.

## Dataset

- Brain MRI scans in **DICOM** format, including subjects from both pediatric and elderly populations.
- Each sample is a series of MRI slices.
- We apply a **windowing technique** to extract meaningful levels from the DICOM series for classification.
- Dataset is split into training, validation, and testing sets.

## Project Structure

```
project/
├── requirements.txt          # Project dependencies
├── README.md                # This file
├── train.py                 # Main training script
├── evaluate.py              # Model evaluation script
├── processors/              # Data processing modules
│   ├── __init__.py
│   ├── dicom_processor.py   # DICOM file processing
│   └── image_processor.py   # Image processing operations
├── models/                  # Model architectures
│   ├── __init__.py
│   ├── base_model.py       # Base model class
│   ├── cnn_models.py       # CNN model implementations
│   └── transfer_models.py  # Transfer learning models
└── augmentation/           # Data augmentation
    ├── __init__.py
    └── image_augmentation.py
```

## Available Models

1. **Simple CNN**: A basic 3D CNN architecture
2. **TinyVGG**: A lightweight VGG-style network
3. **Custom ResNet**: A custom ResNet implementation for 3D MRI data
4. **Transfer Learning Models**:
   - ResNet50
   - EfficientNetB0
   - VGG16

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/iAAA-event/iAAA-MRI-Challenge.git
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

To train a model, use the `train.py` script:

```bash
python train.py --model simple_cnn \
                --data-dir /path/to/data \
                --batch-size 4 \
                --epochs 10 \
                --checkpoints-dir checkpoints
```

Available model options:
- `simple_cnn`
- `tiny_vgg`
- `custom_resnet`
- `transfer_resnet50`
- `transfer_efficientnet`
- `transfer_vgg16`

### Evaluation

To evaluate a trained model:

```bash
python evaluate.py --model-path /path/to/model \
                  --data-dir /path/to/test/data \
                  --predictions-file predictions.csv
```

## Features

- Multiple model architectures for comparison
- Data augmentation using:
  - Albumentations
  - TorchIO
  - TensorFlow built-in augmentations
- Comprehensive evaluation metrics:
  - AUC-ROC
  - Precision
  - Recall
  - Sensitivity at Specificity
  - Specificity at Sensitivity
- Model checkpointing and monitoring
- DICOM processing pipeline
- Configurable training parameters

## Contact

Elena Morshedloo — elenamor314@gmail.com

Project Link: https://github.com/NeuroSapiens314/BrainAnomalyDetection

## License

This project is licensed under the MIT License - see the LICENSE file for details. 