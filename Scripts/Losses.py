# Losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for dealing with class imbalance.

    Args:
        alpha (float): Weighting factor for the rare class
        gamma (float): Focusing parameter that reduces the loss contribution from easy examples
        reduction (str): 'mean', 'sum', or 'none'
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Calculate focal loss.

        Args:
            inputs (Tensor): Predicted probabilities from the model (after sigmoid) - shape [N, 1]
            targets (Tensor): Ground truth labels - shape [N, 1]

        Returns:
            Tensor: Computed focal loss
        """
        # Flatten to ensure we're working with [N] tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # Binary cross entropy
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")

        # Calculate focal weights
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = torch.pow(1 - pt, self.gamma)

        # Apply alpha for class imbalance
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)

        # Calculate final focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


def vae_loss(recon_x, x, mu, logvar, kld_weight=0.5):
    """
    VAE loss function combining reconstruction loss and KL divergence.

    Args:
        recon_x (Tensor): Reconstructed images
        x (Tensor): Original images
        mu (Tensor): Mean of the latent distribution
        logvar (Tensor): Log variance of the latent distribution
        kld_weight (float): Weight for the KL divergence term

    Returns:
        tuple: (total_loss, reconstruction_loss, kl_divergence)
    """
    # Reconstruction loss (mean squared error)
    recon_loss = F.mse_loss(recon_x, x, reduction="sum") / x.size(0)

    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)

    # Total loss
    total_loss = recon_loss + kld_weight * kld_loss

    return total_loss, recon_loss, kld_loss


def gan_losses():
    """
    Returns loss functions for GAN training.

    Returns:
        tuple: (adversarial_loss, auxiliary_loss)
    """
    # Binary cross entropy loss
    adversarial_loss = nn.BCELoss()

    # Mean squared error for features matching
    l2_loss = nn.MSELoss()

    return adversarial_loss, l2_loss


def anomaly_score(real, reconstructed, discriminator=None):
    """
    Calculate anomaly score based on reconstruction error and optional feature matching.

    Args:
        real (Tensor): Original images
        reconstructed (Tensor): Reconstructed images
        discriminator (nn.Module, optional): Discriminator model for feature matching

    Returns:
        Tensor: Anomaly scores for each sample
    """
    # Reconstruction error (L2 distance)
    recon_error = torch.mean((real - reconstructed) ** 2, dim=[1, 2, 3])

    if discriminator is not None:
        # Feature matching error
        _, real_features = discriminator(real)
        _, recon_features = discriminator(reconstructed)
        feature_error = torch.mean((real_features - recon_features) ** 2, dim=1)

        # Combined anomaly score (weighted sum)
        anomaly_scores = 0.7 * recon_error + 0.3 * feature_error
    else:
        anomaly_scores = recon_error

    return anomaly_scores


# Example usage
if __name__ == "__main__":
    # Create example tensors
    recon_x = torch.randn(4, 1, 128, 128)
    x = torch.randn(4, 1, 128, 128)
    mu = torch.randn(4, 128)
    logvar = torch.randn(4, 128)

    # Test VAE loss
    total_loss, recon_loss, kld_loss = vae_loss(recon_x, x, mu, logvar)
    print(
        f"VAE total loss: {total_loss.item()}, recon loss: {recon_loss.item()}, KLD: {kld_loss.item()}"
    )

    # Test Focal Loss
    inputs = torch.sigmoid(torch.randn(4, 1))
    targets = torch.randint(0, 2, (4, 1)).float()
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss_value = focal_loss(inputs, targets)
    print(f"Focal loss: {loss_value.item()}")

    # Test anomaly score
    scores = anomaly_score(x, recon_x)
    print(f"Anomaly scores: {scores}")
