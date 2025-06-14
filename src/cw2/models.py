import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_channels: int, img_height: int, img_width: int, num_classes: int):
        super(Net, self).__init__()
        self.hidden_units = 256
        self.num_classes = num_classes
        self.in_features = n_channels*img_height*img_width
        self.fc1 = nn.Linear(self.in_features, self.hidden_units)
        self.fc2 = nn.Linear(self.hidden_units, self.hidden_units)
        self.fc3 = nn.Linear(self.hidden_units, self.hidden_units)
        self.fc4 = nn.Linear(self.hidden_units, num_classes)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        x = x.view(x.shape[0], -1)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.tanh(self.fc3(x))
        logits = self.fc4(x)
        return logits


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
