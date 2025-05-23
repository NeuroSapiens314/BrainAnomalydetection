"""
Main training script
"""
import argparse
from pathlib import Path

import tensorflow as tf
import pandas as pd
import numpy as np

from processors.dicom_processor import DicomProcessor
from processors.image_processor import ImageProcessor
from models.cnn_models import SimpleCNN, TinyVGG, CustomResNet
from models.transfer_models import TransferResNet50, TransferEfficientNet, TransferVGG16
from augmentation.image_augmentation import ImageAugmentation

def create_dataset(df: pd.DataFrame, batch_size: int, augment: bool = False) -> tf.data.Dataset:
    """Creates a TensorFlow dataset from DataFrame."""
    paths = [str(i) for i in df['prepared_path']]
    
    def read_npy_file(item):
        data = np.load(item.decode())
        return np.expand_dims(data.astype(np.float32), axis=-1)
    
    volumes = tf.data.Dataset.from_tensor_slices(paths).map(
        lambda item: tf.numpy_function(read_npy_file, [item], [tf.float32,])[0]
    )
    labels = tf.data.Dataset.from_tensor_slices(df['prediction'].values)
    zipped = tf.data.Dataset.zip((volumes, labels))
    
    if augment:
        zipped = zipped.map(ImageAugmentation.tf_augmentation)
    
    ds = zipped.shuffle(len(paths), seed=7).batch(batch_size).repeat()
    return ds

def train_model(model_name: str, training_df: pd.DataFrame, validation_df: pd.DataFrame,
                batch_size: int, epochs: int, checkpoints_dir: Path):
    """Trains the specified model."""
    # Create datasets
    training_ds = create_dataset(training_df, batch_size, augment=True)
    validation_ds = create_dataset(validation_df, batch_size)
    
    n_iter_training = (len(training_df) // batch_size) + int((len(training_df) % batch_size) > 0)
    n_iter_validation = (len(validation_df) // batch_size) + int((len(validation_df) % batch_size) > 0)
    
    # Create model
    expected_num_slices = 16
    target_h = target_w = 256
    input_shape = (expected_num_slices, target_h, target_w, 1)
    
    model_classes = {
        'simple_cnn': SimpleCNN,
        'tiny_vgg': TinyVGG,
        'custom_resnet': CustomResNet,
        'transfer_resnet50': TransferResNet50,
        'transfer_efficientnet': TransferEfficientNet,
        'transfer_vgg16': TransferVGG16
    }
    
    model_class = model_classes.get(model_name)
    if not model_class:
        raise ValueError(f"Unknown model name: {model_name}")
    
    if model_name in ['transfer_resnet50', 'transfer_efficientnet', 'transfer_vgg16']:
        model = model_class.create_model(input_shape=(target_h, target_w, 3), output_units=1)
    else:
        model = model_class.create_model(expected_num_slices, target_h, target_w)
    
    # Compile model
    model.compile(
        optimizer=model_class.get_default_optimizer(),
        loss=model_class.get_default_loss(),
        metrics=model_class.get_default_metrics()
    )
    
    # Setup checkpointing
    checkpoints_dir.mkdir(exist_ok=True)
    to_track = 'val_auc'
    checkpoint_path = str(checkpoints_dir / f"model-{{epoch:04d}}-{{{to_track}:4.5f}}.h5")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            save_best_only=True,
            save_weights_only=False,
            monitor=to_track
        )
    ]
    
    # Train model
    history = model.fit(
        training_ds,
        steps_per_epoch=n_iter_training,
        validation_data=validation_ds,
        validation_steps=n_iter_validation,
        epochs=epochs,
        callbacks=callbacks
    )
    
    return history

def main():
    parser = argparse.ArgumentParser(description='Train MRI classification model')
    parser.add_argument('--model', type=str, required=True,
                      choices=['simple_cnn', 'tiny_vgg', 'custom_resnet',
                              'transfer_resnet50', 'transfer_efficientnet', 'transfer_vgg16'],
                      help='Model architecture to use')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10,
                      help='Number of epochs to train')
    parser.add_argument('--data-dir', type=Path, required=True,
                      help='Directory containing the data')
    parser.add_argument('--checkpoints-dir', type=Path, default='checkpoints',
                      help='Directory to save model checkpoints')
    
    args = parser.parse_args()
    
    # Load and prepare data
    labels_path = args.data_dir / 'labels.csv'
    annotations = pd.read_csv(labels_path).drop(['normal', 'corrupted'], axis=1)
    annotations.rename({"abnormal": "prediction"}, axis=1, inplace=True)
    
    # Split data
    from sklearn.model_selection import train_test_split
    training_df, validation_df = train_test_split(
        annotations,
        test_size=0.3,
        stratify=annotations['prediction'],
        random_state=7
    )
    
    # Train model
    history = train_model(
        args.model,
        training_df,
        validation_df,
        args.batch_size,
        args.epochs,
        args.checkpoints_dir
    )

if __name__ == '__main__':
    main() 