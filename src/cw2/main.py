from typing import Callable
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


import os
import sys

from utils import get_datasets, plot_images
from models import CONV5_Net, Net, ResNet18
import time
from numerical import NTanh, ParamTanh


class FGSMAttack:
    def __init__(self, model, device, epsilon=0.3):
        self.model = model.eval()
        self.device = device
        self.epsilon = epsilon  # perturbation size

    def generate(self, images, labels):
        images = images.clone().detach().to(self.device)
        labels = labels.to(self.device)
        images.requires_grad = True

        outputs = self.model(images)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        self.model.zero_grad()
        loss.backward()

        # Collect gradient sign
        grad_sign = images.grad.data.sign()

        # Create perturbed image by adjusting each pixel of the input image
        adv_images = images + self.epsilon * grad_sign

        # Clamp to maintain [min, max] range (usually 0-1 for normalized images)
        adv_images = torch.clamp(adv_images, 0, 1)

        return adv_images.detach()


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


def test_act_fct_replacement(model_type: str, model: nn.Module, device: torch.device | str):
    INTERVAL = 0.1

    def sample_deltas(std=0.1, low=-INTERVAL, high=INTERVAL, n=4):
        deltas = torch.randn(n) * std
        return torch.clamp(deltas, low, high).tolist()

    model.eval()
    model.to(device)

    if model_type == "linear":
        for k, v in model.named_children():
            if isinstance(v, nn.Tanh):
                delta_1, delta_2, delta_3, delta_4 = sample_deltas()
                print(f'Params: {delta_1, delta_2, delta_3, delta_4}')
                model._modules[k] = ParamTanh(delta1=delta_1,
                                              delta2=delta_2,
                                              delta3=delta_3,
                                              delta4=delta_4)
    elif model_type == "conv":
        for idx, layer in enumerate(model.features):
            if isinstance(layer, nn.Tanh):
                delta_1, delta_2, delta_3, delta_4 = sample_deltas()
                print(f'Params: {delta_1, delta_2, delta_3, delta_4}')
                model.features[idx] = ParamTanh(delta1=delta_1,
                                                delta2=delta_2,
                                                delta3=delta_3,
                                                delta4=delta_4)


# 2. Function to replace the model's activation function
def replace_model_act_function(resnet_model):
    # This function now doesn't "replace" in-place on an existing F.tanh model,
    # but rather initializes a new model with ParamTanh and attempts to copy
    # the weights. This is because F.tanh is not a nn.Module that can be replaced.
    # We need to construct a new model with the desired activation.

    print("\nReplacing activation functions with ParamTanh (by creating a new model and loading weights)...")

    # Create a new ResNet model with ParamTanh
    # Generate random deltas for this specific replacement
    import random
    interval = 0.09
    delta1 = random.uniform(-interval, interval)
    delta2 = random.uniform(-interval, interval)
    delta3 = random.uniform(-interval, interval)
    delta4 = random.uniform(-interval, interval)
    print(f'Params: {delta1, delta2, delta3, delta4}')
    new_param_tanh_instance = ParamTanh(delta1, delta2, delta3, delta4)

    replaced_model = ResNet18(act_fn=new_param_tanh_instance)

    # Copy the state dictionary from the original model to the new model.
    # This assumes the architecture (excluding the activation function implementation)
    # remains the same.
    replaced_model.load_state_dict(resnet_model.state_dict(), strict=False)
    # strict=False is used because the new model will have a `ParamTanh` instance
    # which has parameters (delta1, delta2, delta3, delta4, C1, C2, C3, C4)
    # that were not in the original model's state_dict if F.tanh was used.
    # If the original model was already built with `ParamTanh` (e.g. from ResNet18_ParamTanh()),
    # then strict=True might work, depending on how deltas are handled.

    print("Activation functions successfully replaced with ParamTanh in the new model.")
    return replaced_model


# --- Example Usage (to be executed by the user) ---
if __name__ == '__main__':
    # torch.manual_seed(1137)

    DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    # Choose dataset
    DATASET = 'cifar10'  # or 'cifar10'
    train_dataset, test_dataset, input_channels, num_classes = get_datasets(
        DATASET)

    batch_size = 128
    epochs = 20
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model
    model_type = "resnet"
    # model = CONV5_Net()
    model = ResNet18()
    # model = Net(3, 28, 28, 10)
    # model = Net(3, 32, 32, 10)
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

    # print("\n--- Carlini & Wagner Attack Demonstration ---")
    print("\n--- Adversarial Attack Demonstration ---")
    model.eval()

    # eval1_start = time.time()
    # eval_model(model, DEVICE, test_loader, nn.CrossEntropyLoss())
    # eval1_duration = time.time()-eval1_start

    # if model_type == "resnet":
    #     model = replace_model_act_function(resnet_model=model)
    # else:
    #     test_act_fct_replacement(model_type, model, DEVICE)

    # eval2_start = time.time()
    # eval_model(model, DEVICE, test_loader, nn.CrossEntropyLoss())
    # eval2_duration = time.time()-eval2_start
    # print(eval1_duration)
    # print(eval2_duration)
    # sys.exit(1)

    # Get one batch from test_loader
    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    # Run CW L2 attack
    cw_attack = CarliniWagnerL2(
        model=model,
        device=DEVICE,
        targeted=False,
        c=0.008,         # strong tradeoff
        kappa=1,       # confident misclassification
        steps=100,     # sufficient optimization
        lr=0.01         # stable optimizer
    )
    adv_images = cw_attack.generate(images, labels)

    # run FGSM attack
    # fgsm_attack = FGSMAttack(model=model,
    #                          device=DEVICE,
    #                          epsilon=0.6)
    # adv_images = fgsm_attack.generate(images, labels)

    adv_loader = DataLoader(AdvLoader(adv_images, labels),
                            batch_size=batch_size, shuffle=False)

    eval_model(model, DEVICE, test_loader, nn.CrossEntropyLoss())
    eval_model(model, DEVICE, adv_loader, nn.CrossEntropyLoss())

    if model_type == "resnet":
        model = replace_model_act_function(resnet_model=model)
    else:
        test_act_fct_replacement(model_type, model, DEVICE)

    eval_model(model, DEVICE, test_loader, nn.CrossEntropyLoss())
    eval_model(model, DEVICE, adv_loader, nn.CrossEntropyLoss())

    sys.exit(1)

    # Predict on original and adversarial images
    with torch.no_grad():
        orig_preds = model(images).argmax(dim=1)
        adv_preds = model(adv_images).argmax(dim=1)

    compare_results = True

    if compare_results:
        adversarial_labels = 0
        for i in range(len(labels)):
            if os.getenv("DEBUG"):
                print(
                    f"[{i}] True: {labels[i].item()} | Orig: {orig_preds[i].item()} | Adv: {adv_preds[i].item()}")

            if (orig_preds[i].item() == labels[i].item()) and (labels[i].item() != adv_preds[i].item()):
                adversarial_labels += 1
                if os.getenv("DEBUG"):
                    print(f'Idx: {i}: Attack was successful for {labels[i]}')
        print(adversarial_labels)
        # plot_images(images, orig_preds, adv_images, adv_preds, 15)

"""
python3 main.py
Using device: mps

--- Loading pre-trained model for cifar10 ---
Pre-trained model loaded successfully.

--- Carlini & Wagner Attack Demonstration ---
Acc: 75.69 % | Loss: 1.176 (before attack, on all validation set 10000 samples)
Acc: 13.28 % | Loss: 3.346 (after attack, but evaluation is done only for the first batch (i.e, 128 samples))

Replacing activation functions with ParamTanh (by creating a new model and loading weights)...
Params: (0.05520621723583746, -0.06867610101078478, 0.060870514150619504, -0.03263180807517824)
Activation functions successfully replaced with ParamTanh in the new model.
Acc: 61.07 % | Loss: 2.907 (on all validation set)
Acc: 27.34 % | Loss: 5.889 (only for the first batch (i.e, 128 samples))
"""
