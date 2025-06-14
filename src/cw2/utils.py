import torchvision
import torchvision.transforms as transforms

import torch
import matplotlib.pyplot as plt


# --- 3. Data Loading and Preprocessing ---
def get_datasets(dataset_name='mnist'):
    if dataset_name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            # Convert MNIST to 3 channels for CONV5_Net
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])
        train_dataset = torchvision.datasets.MNIST(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(
            root='./data', train=False, download=True, transform=transform)
        input_channels = 3  # After repeating
        num_classes = 10
    elif dataset_name == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        input_channels = 3
        num_classes = 10
    else:
        raise ValueError("Dataset not supported. Choose 'mnist' or 'cifar10'.")
    return train_dataset, test_dataset, input_channels, num_classes


def plot_images(images: torch.Tensor, orig_preds: torch.Tensor, adv_images: torch.Tensor, adv_preds: torch.Tensor, label_idx: int):
    def unnormalize(img_tensor):
        mean = torch.tensor([0.1307, 0.1307, 0.1307]).view(3, 1, 1)
        std = torch.tensor([0.3081, 0.3081, 0.3081]).view(3, 1, 1)

        return img_tensor * std + mean

    i = label_idx  # index to visualize
    img = unnormalize(images[i].cpu().squeeze()).permute(1, 2, 0).clamp(0, 1)
    adv_img = unnormalize(adv_images[i].cpu().squeeze()).permute(
        1, 2, 0).clamp(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img.numpy())
    axes[0].set_title(f"Original: {orig_preds[i].item()}")
    axes[0].axis("off")

    axes[1].imshow(adv_img.numpy())
    axes[1].set_title(f"Adversarial: {adv_preds[i].item()}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()
