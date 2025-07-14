import torch
import torch.nn as nn

import random

from models import ResNet18
from numerical import ParamTanh

from utils import get_datasets

from torch.utils.data import DataLoader

from typing import Callable

INTERVAL = 0.1


def test_act_fct_replacement(model_type: str, model: nn.Module, device: torch.device | str):
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
    print("\nReplacing activation functions with ParamTanh (by creating a new model and loading weights)...")

    delta1 = random.uniform(-INTERVAL, INTERVAL)
    delta2 = random.uniform(-INTERVAL, INTERVAL)
    delta3 = random.uniform(-INTERVAL, INTERVAL)
    delta4 = random.uniform(-INTERVAL, INTERVAL)
    print(f'Params: {delta1, delta2, delta3, delta4}')
    new_param_tanh_instance = ParamTanh(delta1, delta2, delta3, delta4)

    replaced_model = ResNet18(act_fn=new_param_tanh_instance)

    replaced_model.load_state_dict(resnet_model.state_dict(), strict=False)

    print("Activation functions successfully replaced with ParamTanh in the new model.")
    return replaced_model


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


if __name__ == "__main__":
    DEVICE = "mps"
    print(f"Using device: {DEVICE}")

    DATASET = 'cifar10'
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
    model = ResNet18()
    model.to(DEVICE)
    model_pth = f'{DATASET}-{model._get_name()}-{epochs}.pth'

    model.eval()

    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    if model_type == "resnet":
        model = replace_model_act_function(resnet_model=model)
    else:
        test_act_fct_replacement(model_type, model, DEVICE)

    eval_model(model, DEVICE, test_loader, nn.CrossEntropyLoss())
