"""
Base model module containing common model functionality
"""
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseModel(nn.Module, ABC):
    """Base class for all neural network models in the project."""
    
    def __init__(self, input_shape: tuple, num_classes: int, learning_rate: float = 0.001):
        """
        Initialize the base model.
        
        Args:
            input_shape (tuple): Shape of input images (channels, depth, height, width)
            num_classes (int): Number of output classes
            learning_rate (float): Initial learning rate for optimization
        """
        super().__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Model output
        """
        pass
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dict[str, Any]: Model configuration dictionary
        """
        return {
            "input_shape": self.input_shape,
            "num_classes": self.num_classes,
            "learning_rate": self.learning_rate
        }
    
    @classmethod
    def load_from_checkpoint(cls, 
                           checkpoint_path: str, 
                           map_location: Optional[str] = None) -> 'BaseModel':
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
            map_location (Optional[str]): Device to load the model to
            
        Returns:
            BaseModel: Loaded model instance
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        model = cls(**checkpoint['model_config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def save_checkpoint(self, path: str, additional_info: Optional[Dict] = None):
        """
        Save model checkpoint.
        
        Args:
            path (str): Path to save checkpoint
            additional_info (Optional[Dict]): Additional information to save
        """
        checkpoint = {
            'state_dict': self.state_dict(),
            'model_config': self.get_config()
        }
        if additional_info:
            checkpoint.update(additional_info)
        torch.save(checkpoint, path)

    @staticmethod
    def get_default_metrics():
        """Returns default metrics for models."""
        return [
            keras.metrics.SensitivityAtSpecificity(0.95),
            keras.metrics.SpecificityAtSensitivity(0.95),
            keras.metrics.AUC(name='auc'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall')
        ]

    @staticmethod
    def get_default_loss():
        """Returns default loss function."""
        return keras.losses.BinaryCrossentropy()

    @staticmethod
    def get_default_optimizer():
        """Returns default optimizer."""
        return keras.optimizers.Adam(0.001) 