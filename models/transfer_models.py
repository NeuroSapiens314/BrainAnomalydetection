"""
Module containing transfer learning model architectures
"""
from tensorflow import keras
from tensorflow.keras import layers, applications
from .base_model import BaseModel

class TransferResNet50(BaseModel):
    """ResNet50 transfer learning model."""
    
    @staticmethod
    def create_model(input_shape, output_units):
        """Creates a ResNet50-based model."""
        base_model = applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze the base model
        base_model.trainable = False

        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(output_units, activation='sigmoid')(x)

        return keras.Model(inputs, outputs)

class TransferEfficientNet(BaseModel):
    """EfficientNet transfer learning model."""
    
    @staticmethod
    def create_model(input_shape, output_units):
        """Creates an EfficientNet-based model."""
        base_model = applications.EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(output_units, activation='sigmoid')(x)

        return keras.Model(inputs, outputs)

class TransferVGG16(BaseModel):
    """VGG16 transfer learning model."""
    
    @staticmethod
    def create_model(input_shape, output_units):
        """Creates a VGG16-based model."""
        base_model = applications.VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )

        inputs = keras.Input(shape=input_shape)
        x = base_model(inputs)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        outputs = layers.Dense(output_units, activation='sigmoid')(x)

        return keras.Model(inputs, outputs) 