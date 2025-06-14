from typing import Callable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


import os
import sys

from utils import get_datasets, plot_images
from models import CONV5_Net, Net


class CarliniWagnerL2:
    def __init__(self, model: nn.Module, device: str, targeted=False, c=1e-4, kappa=0, steps=1000, lr=0.01):
        self.model = model.eval()
        self.targeted = targeted
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.device = device
        self.mean = torch.tensor((0.1307, 0.1307, 0.1307)).view(
            1, 3, 1, 1).to(self.device)
        self.std = torch.tensor((0.3081, 0.3081, 0.3081)).view(
            1, 3, 1, 1).to(self.device)

    def _loss(self, outputs, labels, l2dist):
        one_hot_labels = F.one_hot(
            labels, num_classes=outputs.shape[1]).float()

        real = torch.sum(one_hot_labels * outputs, dim=1)
        other = torch.max((1 - one_hot_labels) * outputs -
                          one_hot_labels * 1e4, dim=1)[0]

        if self.targeted:
            f_loss = torch.clamp(other - real + self.kappa, min=0)
        else:
            f_loss = torch.clamp(real - other + self.kappa, min=0)

        return self.c * f_loss + l2dist

    def generate(self, images, labels):
        self.model.to(self.device)
        # Unnormalize images
        images_denorm = images * self.std + self.mean

        w = self._to_tanh_space(images_denorm.clone().detach())
        w = w.clone().detach().requires_grad_(True).to(self.device)

        optimizer = torch.optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            adv_images = self._from_tanh_space(w)

            # Normalize before feeding into model
            norm_adv_images = (adv_images - self.mean) / self.std
            outputs = self.model(norm_adv_images)

            l2dist = F.mse_loss(adv_images, images_denorm, reduction='none')
            l2dist = l2dist.view(images.size(0), -1).sum(dim=1)

            loss = self._loss(outputs, labels, l2dist).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Return normalized adversarial image for evaluation
        return (self._from_tanh_space(w).detach() - self.mean) / self.std

    def _to_tanh_space(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x + 1e-12))

    def _from_tanh_space(self, w):
        return torch.tanh(w)


# --- 4. Training Function (to be executed by the user) ---
def train_model(model, device, train_loader, test_loader, num_epochs=10, learning_rate=0.001):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()

    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        rloss = 0.0
        vloss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            rloss += loss.item()*inputs.shape[0]

            loss.backward()
            optimizer.step()

        # Evaluate on test set after each epoch
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)

                loss = loss_fn(outputs, labels)
                vloss += loss.item()*inputs.shape[0]

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        rloss /= len(train_loader.dataset)
        vloss /= len(test_loader.dataset)
        print(
            f'Epoch [{epoch+1}/{num_epochs}] Test Accuracy: {accuracy:.2f}% | rloss: {rloss:.3f} | vloss: {vloss:.3f}')
        model.train()
    print("Training finished.")


class AdvLoader(Dataset):
    def __init__(self, adv_samples, targets):
        self.adv_samples = adv_samples
        self.targets = targets

    def __len__(self):
        return len(self.adv_samples)

    def __getitem__(self, idx):
        return self.adv_samples[idx], self.targets[idx]


def eval_model(model: nn.Module, device: str, test_loader: DataLoader, loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    model.to(device)
    model.eval()

    vloss = 0
    preds = 0
    for x, y_true in test_loader:
        x, y_true = x.to(device), y_true.to(device)

        y = model(x)
        loss = loss_fn(y, y_true)
        vloss += loss.item()*x.shape[0]

        preds += (torch.argmax(y, dim=1) == y_true).sum().item()

    vloss /= len(test_loader.dataset)
    acc = preds/len(test_loader.dataset)*100.0
    print(f'Acc: {acc:.2f} % | Loss: {vloss:.3f}')


# --- Example Usage (to be executed by the user) ---
if __name__ == '__main__':
    DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    # Choose dataset
    DATASET = 'mnist'  # or 'cifar10'
    train_dataset, test_dataset, input_channels, num_classes = get_datasets(
        DATASET)

    batch_size = 32
    epochs = 20
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    # model = CONV5_Net(num_classes=num_classes)
    model = Net(3, 28, 28, 10)
    model.to(DEVICE)
    model_pth = f'{DATASET}-{model._get_name()}-{epochs}.pth'

    # --- Loading a pre-trained model (for demonstration if you skip training) ---
    print(f"\n--- Loading pre-trained model for {DATASET} ---")
    try:
        model.load_state_dict(torch.load(model_pth, map_location=DEVICE))
        print("Pre-trained model loaded successfully.")
    except FileNotFoundError:
        print(
            f"Pre-trained model not found. Applying training...")
        train_model(model,
                    DEVICE,
                    train_loader,
                    test_loader,
                    num_epochs=epochs,
                    learning_rate=0.001)
        torch.save(model.state_dict(), model_pth)
        print("Model trained and saved.")

    print("\n--- Carlini & Wagner Attack Demonstration ---")
    # --- Adversarial Attack Demonstration ---
    model.eval()

    # Get one batch from test_loader
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    # Initialize and run CW L2 attack
    cw_attack = CarliniWagnerL2(model,
                                device=DEVICE,
                                targeted=False,
                                c=1,
                                kappa=10,
                                steps=10,
                                lr=0.01)
    adv_images = cw_attack.generate(images, labels)
    adv_loader = DataLoader(AdvLoader(adv_images, labels),
                            batch_size=batch_size, shuffle=False)

    eval_model(model, DEVICE, test_loader, nn.CrossEntropyLoss())
    eval_model(model, DEVICE, adv_loader, nn.CrossEntropyLoss())

    # Predict on original and adversarial images
    with torch.no_grad():
        orig_preds = model(images).argmax(dim=1)
        adv_preds = model(adv_images).argmax(dim=1)

    adversarial_labels = 0
    # Compare results
    for i in range(len(labels)):
        print(
            f"[{i}] True: {labels[i].item()} | Orig: {orig_preds[i].item()} | Adv: {adv_preds[i].item()}")

        if (orig_preds[i].item() == labels[i].item()) and (labels[i].item() != adv_preds[i].item()):
            adversarial_labels += 1
            if os.getenv("DEBUG"):
                print(f'Idx: {i}: Attack was successful for {labels[i]}')

    print(adversarial_labels)
    plot_images(images, orig_preds, adv_images, adv_preds, 15)
