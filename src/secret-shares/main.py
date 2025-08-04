import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import matplotlib.pyplot as plt

# Set seed for reproducibility
torch.manual_seed(1137)

# Parameters
embedding_dim = 512
num_shares = 10
num_samples = 10000
batch_size = 128
num_epochs = 100
learning_rate = 1e-3

# Generate synthetic embeddings dataset
embeddings = torch.randn(num_samples, embedding_dim)
dataset = TensorDataset(embeddings)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define alphas vector (e.g., linear scaling from 1.0 to 2.0)
alphas_train = torch.linspace(
    1.0, 2.0, steps=num_shares).view(1, num_shares, 1)

# Lightning Module


class ShareGeneratorModule(pl.LightningModule):
    def __init__(self, embedding_dim, num_shares, alphas, lr):
        super().__init__()
        self.save_hyperparameters()
        self.embedding_dim = embedding_dim
        self.num_shares = num_shares
        self.alphas = alphas
        self.lr = lr
        self.fc = nn.Linear(embedding_dim, embedding_dim * num_shares)
        self.criterion = nn.MSELoss()
        self.train_losses = []

    def forward(self, e_x):
        shares = self.fc(e_x).view(-1, self.num_shares, self.embedding_dim)
        return shares

    def reconstruct(self, shares, alphas):
        weighted_shares = shares * alphas
        return weighted_shares.sum(dim=1)

    def training_step(self, batch, batch_idx):
        e_x = batch[0]
        shares = self.forward(e_x)
        e_x_hat = self.reconstruct(shares, self.alphas.to(self.device))
        loss = self.criterion(e_x_hat, e_x)
        self.log("train_loss", loss)
        self.train_losses.append(loss.item())
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

    def evaluate_metrics(self, e_x, e_x_hat, prefix="eval"):
        mse = F.mse_loss(e_x_hat, e_x).item()
        l2 = torch.norm(e_x_hat - e_x, p=2).item()
        cosine = F.cosine_similarity(e_x_hat, e_x).mean().item()
        rel_error = l2 / torch.norm(e_x, p=2).item()

        print(
            f"{prefix} - MSE: {mse:.6f}, L2: {l2:.6f}, Cosine: {cosine:.6f}, RelError: {rel_error:.6f}")
        return {
            "mse": mse,
            "l2": l2,
            "cosine": cosine,
            "rel_error": rel_error
        }

    def plot_training_curve(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Training Loss')
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        plt.grid(True)
        plt.show()


# Instantiate model
model = ShareGeneratorModule(
    embedding_dim, num_shares, alphas_train, learning_rate)

# Train
logger = TensorBoardLogger("tb_logs", name="share_generator")
trainer = Trainer(max_epochs=num_epochs, logger=logger, log_every_n_steps=10)
trainer.fit(model, dataloader)

# Plot training loss curve
model.plot_training_curve()

# --- Evaluation ---
e_x_test = torch.randn(1, embedding_dim)
print("\nOriginal e(x) first 5 dims:", e_x_test[0, :5])

# Evaluate with training alphas
with torch.no_grad():
    shares_test = model(e_x_test)
    e_x_hat_train_alpha = model.reconstruct(shares_test, alphas_train)
    print("Reconstructed e(x) with training alphas first 5 dims:",
          e_x_hat_train_alpha[0, :5])
    model.evaluate_metrics(e_x_test, e_x_hat_train_alpha, prefix="Train Alpha")

# Evaluate with test alphas (e.g., scaled)
alphas_test = alphas_train * 1.5
print("\nTesting with different alphas:", alphas_test.view(-1))

with torch.no_grad():
    e_x_hat_test_alpha = model.reconstruct(shares_test, alphas_test)
    print("Reconstructed e(x) with test alphas first 5 dims:",
          e_x_hat_test_alpha[0, :5])
    model.evaluate_metrics(e_x_test, e_x_hat_test_alpha, prefix="Test Alpha")
