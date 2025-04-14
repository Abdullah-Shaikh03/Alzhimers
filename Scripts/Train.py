import os
import time
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm

from DataLoader import get_dataloaders
from Model import get_model
from Losses import vae_loss, gan_losses, FocalLoss


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models for medical imaging")
    parser.add_argument('--config', type=str,
                        default='Config.yaml', help='Path to config file')
    parser.add_argument('--model_type', type=str,
                        help='Override model type from config (vae or gan)')
    parser.add_argument('--epochs', type=int,
                        help='Override max epochs from config')
    parser.add_argument('--batch_size', type=int,
                        help='Override batch size from config')
    parser.add_argument('--run_id', type=str, default=None,
                        help='Run identifier for logging')
    return parser.parse_args()


def setup_logging(config, args):
    """Set up logging directories and tensorboard writer"""
    # Create run id based on timestamp if not provided
    if args.run_id is None:
        run_id = f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        run_id = args.run_id

    # Create logging directory
    log_dir = Path(config['log_dir']) / run_id
    model_dir = Path(config['model_dir']) / run_id

    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Set up tensorboard writer
    writer = SummaryWriter(log_dir)

    return writer, log_dir, model_dir, run_id


def train_vae(model, train_loader, val_loader, optimizer, scheduler, config, writer, model_dir):
    """Train a Variational Autoencoder model"""
    device = torch.device(config['device'])
    model = model.to(device)

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(config['max_epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kld_loss = 0

        # Create progress bar for the training loop
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} [Train]")

        for batch_idx, (data, _) in enumerate(train_pbar):
            data = data.to(device)

            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss, recon_loss, kld_loss = vae_loss(recon_batch, data, mu, logvar,
                                                  kld_weight=config['kld_weight'])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_kld_loss += kld_loss.item()

            # Update progress bar description with current loss
            train_pbar.set_postfix(loss=f"{loss.item():.4f}",
                                   recon=f"{recon_loss.item():.4f}",
                                   kld=f"{kld_loss.item():.4f}")

        avg_train_loss = train_loss / len(train_loader)
        avg_train_recon = train_recon_loss / len(train_loader)
        avg_train_kld = train_kld_loss / len(train_loader)

        # Log training metrics
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('ReconLoss/train', avg_train_recon, epoch)
        writer.add_scalar('KLDLoss/train', avg_train_kld, epoch)

        # Validation phase
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_kld_loss = 0

        # Create progress bar for validation loop
        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} [Val]")

        with torch.no_grad():
            for data, _ in val_pbar:
                data = data.to(device)
                recon_batch, mu, logvar = model(data)
                loss, recon_loss, kld_loss = vae_loss(recon_batch, data, mu, logvar,
                                                      kld_weight=config['kld_weight'])

                val_loss += loss.item()
                val_recon_loss += recon_loss.item()
                val_kld_loss += kld_loss.item()

                # Update validation progress bar
                val_pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_val_loss = val_loss / len(val_loader)
        avg_val_recon = val_recon_loss / len(val_loader)
        avg_val_kld = val_kld_loss / len(val_loader)

        # Log validation metrics
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        writer.add_scalar('ReconLoss/val', avg_val_recon, epoch)
        writer.add_scalar('KLDLoss/val', avg_val_kld, epoch)

        print(f"Epoch {epoch+1}/{config['max_epochs']} complete. "
              f"Train loss: {avg_train_loss:.6f}, Val loss: {avg_val_loss:.6f}")

        # Update learning rate scheduler
        scheduler.step(avg_val_loss)

        # Save model checkpoint if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, model_dir / f"best_model.pt")
            print(
                f"Model saved at epoch {epoch+1} with validation loss {best_val_loss:.6f}")
        else:
            early_stopping_counter += 1

        # Save checkpoint every N epochs
        if epoch % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, model_dir / f"checkpoint_epoch_{epoch+1}.pt")

        # Early stopping
        if early_stopping_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_val_loss,
    }, model_dir / "final_model.pt")

    print("Training completed.")
    return model


def train_gan(model, train_loader, val_loader, config, writer, model_dir):
    """Train a GAN-based anomaly detection model"""
    device = torch.device(config['device'])
    model = model.to(device)

    # Optimizers
    optimizer_G = optim.Adam(model.generator.parameters(),
                             lr=config['gan_lr'],
                             betas=(config['gan_beta1'], 0.999))

    optimizer_D = optim.Adam(model.discriminator.parameters(),
                             lr=config['gan_lr'],
                             betas=(config['gan_beta1'], 0.999))

    # Loss functions
    adversarial_loss, l2_loss = gan_losses()

    # Labels for real and fake
    real_label = 1.0
    fake_label = 0.0

    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(config['max_epochs']):
        # Training phase
        model.train()
        train_loss_D = 0
        train_loss_G = 0

        # Create progress bar for training loop
        train_pbar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} [Train]")

        for batch_idx, (data, _) in enumerate(train_pbar):
            data = data.to(device)
            batch_size = data.size(0)

            # -----------------
            # Train Discriminator
            # -----------------
            optimizer_D.zero_grad()

            # Real batch
            real_target = torch.full(
                (batch_size, 1), real_label, device=device)
            real_pred, real_features = model.discriminator(data)
            d_loss_real = adversarial_loss(real_pred, real_target)

            # Fake batch
            z = torch.randn(batch_size, model.latent_dim, device=device)
            fake_data = model.generator(z)
            fake_target = torch.full(
                (batch_size, 1), fake_label, device=device)
            fake_pred, fake_features = model.discriminator(fake_data.detach())
            d_loss_fake = adversarial_loss(fake_pred, fake_target)

            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_D.step()

            # -----------------
            # Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate fake data again for generator training
            fake_pred, fake_features = model.discriminator(fake_data)

            # Feature matching loss
            _, real_features = model.discriminator(data)
            feature_matching_loss = l2_loss(
                fake_features, real_features.detach())

            # Generator loss for fooling discriminator
            g_loss = adversarial_loss(
                fake_pred, real_target) + feature_matching_loss
            g_loss.backward()
            optimizer_G.step()

            train_loss_D += d_loss.item()
            train_loss_G += g_loss.item()

            # Update progress bar with current losses
            train_pbar.set_postfix(
                D_loss=f"{d_loss.item():.4f}", G_loss=f"{g_loss.item():.4f}")

        avg_train_loss_D = train_loss_D / len(train_loader)
        avg_train_loss_G = train_loss_G / len(train_loader)

        # Log training metrics
        writer.add_scalar('Loss/train_D', avg_train_loss_D, epoch)
        writer.add_scalar('Loss/train_G', avg_train_loss_G, epoch)

        # Validation phase
        model.eval()
        val_loss = 0

        # Create progress bar for validation loop
        val_pbar = tqdm(
            val_loader, desc=f"Epoch {epoch+1}/{config['max_epochs']} [Val]")

        with torch.no_grad():
            for data, _ in val_pbar:
                data = data.to(device)
                batch_size = data.size(0)

                # Encode-decode the real data
                encoded_z = model.encode(data)
                reconstructed = model.generator(encoded_z)

                # Calculate reconstruction loss as validation metric
                recon_loss = nn.MSELoss()(reconstructed, data).item()
                val_loss += recon_loss

                # Update validation progress bar
                val_pbar.set_postfix(recon_loss=f"{recon_loss:.4f}")

        avg_val_loss = val_loss / len(val_loader)
        writer.add_scalar('Loss/val_recon', avg_val_loss, epoch)

        print(f"Epoch {epoch+1}/{config['max_epochs']} complete. "
              f"Train D loss: {avg_train_loss_D:.6f}, G loss: {avg_train_loss_G:.6f}, "
              f"Val recon loss: {avg_val_loss:.6f}")

        # Save model checkpoint if it's the best so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stopping_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss': best_val_loss,
            }, model_dir / f"best_model.pt")
            print(
                f"Model saved at epoch {epoch+1} with validation loss {best_val_loss:.6f}")
        else:
            early_stopping_counter += 1

        # Save checkpoint every N epochs
        if epoch % config['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'loss': avg_val_loss,
            }, model_dir / f"checkpoint_epoch_{epoch+1}.pt")

        # Early stopping
        if early_stopping_counter >= config['early_stopping_patience']:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss': avg_val_loss,
    }, model_dir / "final_model.pt")

    print("Training completed.")
    return model


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command-line arguments if provided
    if args.model_type:
        config['model_type'] = args.model_type
    if args.epochs:
        config['max_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size

    # Set random seed for reproducibility
    torch.manual_seed(config['random_seed'])
    np.random.seed(config['random_seed'])
    if torch.cuda.is_available() and config['device'] == 'cuda':
        torch.cuda.manual_seed(config['random_seed'])
        torch.backends.cudnn.deterministic = True

    # Setup logging
    writer, log_dir, model_dir, run_id = setup_logging(config, args)

    # Save the config for this run
    with open(log_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Get data loaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Get model
    model = get_model(config)
    print(f"Model type: {config['model_type']}")

    # Train model based on type
    if config['model_type'].lower() == 'vae':
        optimizer = optim.AdamW(model.parameters(),
                                lr=float(config['learning_rate']),
                                weight_decay=float(config['weight_decay']))

        scheduler = ReduceLROnPlateau(optimizer,
                                      mode='min',
                                      factor=config['scheduler_factor'],
                                      patience=config['scheduler_patience'],
                                      verbose=True)

        model = train_vae(model, train_loader, val_loader,
                          optimizer, scheduler, config, writer, model_dir)

    elif config['model_type'].lower() == 'gan':
        model = train_gan(model, train_loader, val_loader,
                          config, writer, model_dir)

    else:
        raise ValueError(f"Unknown model type: {config['model_type']}")

    # Close tensorboard writer
    writer.close()
    print(f"Training completed. Logs saved to {log_dir}")


if __name__ == "__main__":
    main()
