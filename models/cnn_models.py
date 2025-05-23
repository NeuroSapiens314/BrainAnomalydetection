"""
Module containing various CNN model architectures
"""
import torch
import torch.nn as nn
from .base_model import BaseModel

class SimpleCNN(BaseModel):
    """A simple 3D CNN architecture for brain MRI analysis."""
    
    def __init__(self, input_shape: tuple, num_classes: int, learning_rate: float = 0.001):
        super().__init__(input_shape, num_classes, learning_rate)
        
        self.features = nn.Sequential(
            nn.Conv3d(input_shape[0], 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            n_features = dummy_output.numel() // dummy_output.size(0)
            
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class TinyVGG(BaseModel):
    """A smaller version of VGG architecture adapted for 3D MRI data."""
    
    def __init__(self, input_shape: tuple, num_classes: int, learning_rate: float = 0.001):
        super().__init__(input_shape, num_classes, learning_rate)
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(input_shape[0], 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            # Block 2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            
            # Block 3
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2)
        )
        
        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.features(dummy_input)
            n_features = dummy_output.numel() // dummy_output.size(0)
        
        self.classifier = nn.Sequential(
            nn.Linear(n_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class ResidualBlock(nn.Module):
    """Residual block for CustomResNet."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class CustomResNet(BaseModel):
    """Custom ResNet architecture for 3D MRI analysis."""
    
    def __init__(self, input_shape: tuple, num_classes: int, learning_rate: float = 0.001):
        super().__init__(input_shape, num_classes, learning_rate)
        
        self.in_channels = 64
        self.conv1 = nn.Conv3d(input_shape[0], 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        
        # Calculate the size of flattened features
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            dummy_output = self.forward_features(dummy_input)
            n_features = dummy_output.numel() // dummy_output.size(0)
        
        self.fc = nn.Linear(n_features, num_classes)
    
    def make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = torch.nn.functional.adaptive_avg_pool3d(out, (1, 1, 1))
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.forward_features(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out 