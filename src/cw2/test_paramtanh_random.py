import torch
import torch.nn as nn

import random

from models import ResNet18, Net, CONV5_Net
from numerical import ParamTanh

from utils import get_datasets

from torch.utils.data import DataLoader

from typing import Callable

import time
import sys
import math

INTERVAL = 0.1
DEVICE = "mps"


def test_act_fct_replacement(model_type: str, model: nn.Module, device: torch.device | str, debug: bool = False):
    def sample_deltas(std=0.1, low=-INTERVAL, high=INTERVAL, n=4):
        deltas = torch.randn(n) * std
        return torch.clamp(deltas, low, high).tolist()

    model.eval()
    model.to(device)

    if model_type == "linear":
        for k, v in model.named_children():
            if isinstance(v, nn.Tanh):
                delta_1, delta_2, delta_3, delta_4 = sample_deltas()
                if debug:
                    print(f'Params: {delta_1, delta_2, delta_3, delta_4}')
                model._modules[k] = ParamTanh(delta1=delta_1,
                                              delta2=delta_2,
                                              delta3=delta_3,
                                              delta4=delta_4)
    elif model_type == "conv":
        for idx, layer in enumerate(model.features):
            if isinstance(layer, nn.Tanh):
                delta_1, delta_2, delta_3, delta_4 = sample_deltas()
                if debug:
                    print(f'Params: {delta_1, delta_2, delta_3, delta_4}')
                model.features[idx] = ParamTanh(delta1=delta_1,
                                                delta2=delta_2,
                                                delta3=delta_3,
                                                delta4=delta_4)


def replace_model_act_function(resnet_model, debug: bool = False):
    resnet_model.to(DEVICE)
    if debug:
        print("\nReplacing activation functions with ParamTanh (by creating a new model and loading weights)...")

    delta1 = random.uniform(-INTERVAL, INTERVAL)
    delta2 = random.uniform(-INTERVAL, INTERVAL)
    delta3 = random.uniform(-INTERVAL, INTERVAL)
    delta4 = random.uniform(-INTERVAL, INTERVAL)
    if debug:
        print(f'Params: {delta1, delta2, delta3, delta4}')
    new_param_tanh_instance = ParamTanh(delta1, delta2, delta3, delta4)

    replaced_model = ResNet18(act_fn=new_param_tanh_instance)

    replaced_model.load_state_dict(resnet_model.state_dict(), strict=False)

    if debug:
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


def test_l4(batch_size: int = 256, num_iterations: int = 100):
    _, mnist_dataset, _, _ = get_datasets(
        "mnist")
    _, cifar_dataset, _, _ = get_datasets(
        "cifar10")
    mnist_loader = DataLoader(
        mnist_dataset, batch_size=batch_size, shuffle=False)
    cifar_loader = DataLoader(
        cifar_dataset, batch_size=batch_size, shuffle=False)
    mnist_images, _ = next(iter(mnist_loader))
    cifar_images, _ = next(iter(cifar_loader))
    mnist_images = mnist_images.to(DEVICE)
    cifar_images = cifar_images.to(DEVICE)

    print(f'MNIST')
    print(f'L4_Net: Tanh')
    model = Net(3, 28, 28, 10)
    model.to(DEVICE)
    model.eval()

    durations = []
    for _ in range(num_iterations):
        start = time.time()
        model(mnist_images)
        durations.append(time.time()-start)
    avg_duration = sum(durations)/num_iterations
    s1 = sum(list(map(lambda x: pow(x-avg_duration, 2), durations)))
    std_dev = math.sqrt(
        1/num_iterations*s1)
    print(
        f'{avg_duration} [s] (for {num_iterations} iterations) | Std dev: {std_dev}')

    print(f'L4_Net: ParamTanh')
    test_act_fct_replacement("linear", model, DEVICE)
    model.to(DEVICE)
    model.eval()

    durations = []
    for _ in range(num_iterations):
        start = time.time()
        model(mnist_images)
        durations.append(time.time()-start)
    avg_duration = sum(durations)/num_iterations
    s1 = sum(list(map(lambda x: pow(x-avg_duration, 2), durations)))
    std_dev = math.sqrt(
        1/num_iterations*s1)
    print(
        f'{avg_duration} [s] (for {num_iterations} iterations) | Std dev: {std_dev}')

    print(f'CIFAR10')
    print(f'L4_Net: Tanh')
    model = Net(3, 32, 32, 10)
    model.to(DEVICE)
    model.eval()

    durations = []
    for _ in range(num_iterations):
        start = time.time()
        model(cifar_images)
        durations.append(time.time()-start)
    avg_duration = sum(durations)/num_iterations
    s1 = sum(list(map(lambda x: pow(x-avg_duration, 2), durations)))
    std_dev = math.sqrt(
        1/num_iterations*s1)
    print(
        f'{avg_duration} [s] (for {num_iterations} iterations) | Std dev: {std_dev}')

    print(f'L4_Net: ParamTanh')
    test_act_fct_replacement("linear", model, DEVICE)
    model.to(DEVICE)
    model.eval()

    durations = []
    for _ in range(num_iterations):
        start = time.time()
        model(cifar_images)
        durations.append(time.time()-start)
    avg_duration = sum(durations)/num_iterations
    s1 = sum(list(map(lambda x: pow(x-avg_duration, 2), durations)))
    std_dev = math.sqrt(
        1/num_iterations*s1)
    print(
        f'{avg_duration} [s] (for {num_iterations} iterations) | Std dev: {std_dev}')


def test_convnet(batch_size: int = 256, num_iterations: int = 100):
    _, test_dataset, _, _ = get_datasets(
        "cifar10")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    print(f'ConvNet: Tanh')
    model = CONV5_Net()
    model.to(DEVICE)
    model.eval()

    durations = []
    for _ in range(num_iterations):
        start = time.time()
        model(images)
        durations.append(time.time()-start)
    avg_duration = sum(durations)/num_iterations
    s1 = sum([pow(x-avg_duration, 2) for x in durations])
    std_dev = math.sqrt(1/num_iterations*s1)
    print(
        f'{avg_duration} [s] (for {num_iterations} iterations) | Std dev: {std_dev}')

    print(f'ConvNet: ParamTanh')
    test_act_fct_replacement("conv", model, DEVICE)
    model.to(DEVICE)
    model.eval()

    durations = []
    for _ in range(num_iterations):
        start = time.time()
        model(images)
        durations.append(time.time()-start)
    avg_duration = sum(durations)/num_iterations
    s1 = sum([pow(x-avg_duration, 2) for x in durations])
    std_dev = math.sqrt(1/num_iterations*s1)
    print(
        f'{avg_duration} [s] (for {num_iterations} iterations) | Std dev: {std_dev}')


def test_resnet(batch_size: int = 256, num_iterations: int = 100):
    _, test_dataset, _, _ = get_datasets(
        "cifar10")

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    images, labels = next(iter(test_loader))
    images, labels = images.to(DEVICE), labels.to(DEVICE)

    model = ResNet18()
    model.to(DEVICE)
    model.eval()

    print(f'Resnet: Tanh')
    durations = []
    for _ in range(num_iterations):
        start = time.time()
        model(images)
        durations.append(time.time()-start)
    avg_duration = sum(durations)/num_iterations
    s1 = sum([pow(x-avg_duration, 2) for x in durations])
    std_dev = math.sqrt(1/num_iterations*s1)
    print(
        f'{avg_duration} [s] (for {num_iterations} iterations) | Std dev: {std_dev}')

    print(f'Resnet: ParamTanh')
    model = replace_model_act_function(resnet_model=model)
    model.to(DEVICE)
    model.eval()

    durations = []
    for _ in range(num_iterations):
        start = time.time()
        model(images)
        durations.append(time.time()-start)
    avg_duration = sum(durations)/num_iterations
    s1 = sum([pow(x-avg_duration, 2) for x in durations])
    std_dev = math.sqrt(1/num_iterations*s1)
    print(
        f'{avg_duration} [s] (for {num_iterations} iterations) | Std dev: {std_dev}')


if __name__ == "__main__":
    batch_size: int = 256
    num_iterations: int = 200
    test_l4(batch_size, num_iterations)
    print("\n")
    test_convnet(batch_size, num_iterations)
    print("\n")
    test_resnet(batch_size, num_iterations)

    # print(f"Using device: {DEVICE}")

    # DATASET = 'cifar10'
    # train_dataset, test_dataset, input_channels, num_classes = get_datasets(
    #     DATASET)

    # batch_size = 1
    # epochs = 20
    # train_loader = DataLoader(
    #     train_dataset, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(
    #     test_dataset, batch_size=batch_size, shuffle=False)

    # # Initialize model
    # model_type = "resnet"
    # model = ResNet18()

    # images, labels = next(iter(test_loader))
    # images, labels = images.to(DEVICE), labels.to(DEVICE)

    # if model_type == "resnet":
    #     model = replace_model_act_function(resnet_model=model)
    #     model.to(DEVICE)
    # else:
    #     test_act_fct_replacement(model_type, model, DEVICE)
    #     model.to(DEVICE)

    # print(model)

    # model.eval()
    # model(images)
