import torch
import torch.nn as nn


# --- 1. CONV5_Net Model Definition ---
class CONV5_Net(nn.Module):
    def __init__(self, num_classes=10):
        super(CONV5_Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 32x16x16 (for CIFAR10) or 32x14x14 (for MNIST 28x28)
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Output: 64x8x8 (for CIFAR10) or 64x7x7 (for MNIST)
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        # Adjusting the linear layer input based on dataset and image size after conv layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
