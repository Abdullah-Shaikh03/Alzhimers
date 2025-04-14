# Visualize.py

import os
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
from matplotlib.colors import LinearSegmentedColormap
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from DataLoader import get_dataloaders
from Model import get_model


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize model results and features")
    parser.add_argument(
        "--config", type=str, default="Config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--results_dir", type=str, default="Results", help="Path to analysis results"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--grad_cam", action="store_true", help="Generate Grad-CAM visualizations"
    )
    parser.add_argument(
        "--filters", action="store_true", help="Visualize first layer filters"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10,
        help="Number of samples for visualizations",
    )
    return parser.parse_args()


def load_model_and_results(model_path, results_dir, config):
    """Load trained model and analysis results"""
    device = torch.device(config["device"])

    # Load model
    model = get_model(config)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # Load results
    metrics_df = pd.read_csv(Path(results_dir) / "metrics.csv")
    class_metrics_df = pd.read_csv(Path(results_dir) / "class_metrics.csv")
    confusion_mat = np.load(Path(results_dir) / "confusion_matrix.npy")

    try:
        roc_data = np.load(Path(results_dir) / "roc_data.npz")
        pr_data = np.load(Path(results_dir) / "pr_data.npz")
    except FileNotFoundError:
        roc_data = None
        pr_data = None

    return model, metrics_df, class_metrics_df, confusion_mat, roc_data, pr_data


def visualize_conv_filters(model, output_dir, config):
    """Visualize filters from the first convolutional layer"""
    print("Visualizing first layer filters...")

    if config["model_type"].lower() == "vae":
        # For VAE, first layer is in the encoder
        first_conv_layer = model.encoder[0].conv
    else:  # GAN
        # For GAN, first layer is in the discriminator
        first_conv_layer = model.discriminator.feature_extractor[0][0]

    # Get filters
    filters = first_conv_layer.weight.data.cpu().numpy()
    n_filters = filters.shape[0]

    # Plot filters
    plt.figure(figsize=(20, 10))
    for i in range(min(n_filters, 64)):  # Show up to 64 filters
        plt.subplot(8, 8, i + 1)
        plt.imshow(filters[i, 0], cmap="viridis")
        plt.axis("off")

    plt.suptitle(
        f"First Layer Convolutional Filters ({config['model_type'].upper()})",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "conv_filters.png", dpi=300)
    plt.close()

    print(f"Filter visualization saved to {output_dir}/conv_filters.png")


def visualize_reconstructions(model, dataloader, output_dir, config, sample_size=10):
    """Visualize original and reconstructed samples"""
    print("Visualizing reconstructions...")
    device = torch.device(config["device"])

    # Get samples
    samples = []
    labels = []
    count = 0

    with torch.no_grad():
        for data, label in dataloader:
            samples.append(data)
            labels.append(label)
            count += data.size(0)
            if count >= sample_size:
                break

    samples = torch.cat(samples, dim=0)[:sample_size].to(device)
    labels = torch.cat(labels, dim=0)[:sample_size].cpu().numpy()

    # Get reconstructions
    if config["model_type"].lower() == "vae":
        recon_samples, _, _ = model(samples)
    else:  # GAN
        encoded_z = model.encode(samples)
        recon_samples = model.generator(encoded_z)

    # Convert to numpy for visualization
    samples = samples.cpu().numpy()
    recon_samples = recon_samples.cpu().numpy()

    # Denormalize images from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    recon_samples = (recon_samples + 1) / 2

    # Plot original vs reconstructed
    plt.figure(figsize=(20, 4))
    for i in range(min(sample_size, 10)):
        # Original
        plt.subplot(2, 10, i + 1)
        plt.imshow(samples[i, 0], cmap="gray")
        plt.title(dataloader.dataset.idx_to_class.get(labels[i], str(labels[i])))
        plt.axis("off")

        # Reconstructed
        plt.subplot(2, 10, i + 11)
        plt.imshow(recon_samples[i, 0], cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

    plt.suptitle(
        f"Original vs Reconstructed Images ({config['model_type'].upper()})",
        fontsize=16,
    )
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "reconstructions.png", dpi=300)
    plt.close()

    print(f"Reconstruction visualization saved to {output_dir}/reconstructions.png")


def visualize_roc_curve(roc_data, output_dir):
    """Visualize ROC curve"""
    if roc_data is None:
        print("No ROC data found. Skipping ROC curve visualization.")
        return

    print("Visualizing ROC curve...")

    plt.figure(figsize=(10, 8))
    plt.plot(
        roc_data["fpr"],
        roc_data["tpr"],
        "b-",
        label=f"ROC curve (AUC = {roc_data['auc']:.3f})",
    )
    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.savefig(Path(output_dir) / "roc_curve.png", dpi=300)
    plt.close()

    print(f"ROC curve saved to {output_dir}/roc_curve.png")


def visualize_pr_curve(pr_data, output_dir):
    """Visualize Precision-Recall curve"""
    if pr_data is None:
        print("No PR data found. Skipping PR curve visualization.")
        return

    print("Visualizing Precision-Recall curve...")

    plt.figure(figsize=(10, 8))
    plt.plot(
        pr_data["recall"],
        pr_data["precision"],
        "b-",
        label=f"PR curve (AP = {pr_data['auc']:.3f})",
    )
    plt.axhline(
        y=sum(pr_data["precision"]) / len(pr_data["precision"]),
        color="r",
        linestyle="--",
        label="No skill",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)

    plt.savefig(Path(output_dir) / "pr_curve.png", dpi=300)
    plt.close()

    print(f"PR curve saved to {output_dir}/pr_curve.png")


def visualize_confusion_matrix(confusion_mat, output_dir):
    """Visualize confusion matrix"""
    print("Visualizing confusion matrix...")

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    plt.savefig(Path(output_dir) / "confusion_matrix.png", dpi=300)
    plt.close()

    print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")


def visualize_class_metrics(class_metrics_df, output_dir):
    """Visualize metrics for each class"""
    print("Visualizing class metrics...")

    metrics = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]

    plt.figure(figsize=(12, 8))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i + 1)
        sns.barplot(x="class", y=metric, data=class_metrics_df)
        plt.title(f"{metric.replace('_', ' ').title()}")
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()

    plt.savefig(Path(output_dir) / "class_metrics.png", dpi=300)
    plt.close()

    print(f"Class metrics visualization saved to {output_dir}/class_metrics.png")


def generate_grad_cam(model, dataloader, output_dir, config, sample_size=5):
    """Generate Grad-CAM visualizations"""
    print("Generating Grad-CAM visualizations...")
    device = torch.device(config["device"])

    if config["model_type"].lower() == "vae":
        # For VAE, use the encoder's last convolutional layer
        target_layer = model.encoder[-1].conv
    else:  # GAN
        # For GAN, use the discriminator's last feature extraction layer
        target_layer = model.discriminator.feature_extractor[-1][0]

    # Create GradCAM
    grad_cam = GradCAM(
        model=model, target_layers=[target_layer], use_cuda=device.type == "cuda"
    )

    # Get samples
    samples = []
    labels = []
    count = 0

    for data, label in dataloader:
        samples.extend(data)
        labels.extend(label)
        count += len(data)
        if count >= sample_size:
            break

    samples = samples[:sample_size]
    labels = labels[:sample_size]

    plt.figure(figsize=(15, 5 * len(samples)))

    for i, (input_tensor, label) in enumerate(zip(samples, labels)):
        # Prepare input
        input_tensor = input_tensor.unsqueeze(0).to(device)

        # Generate the heatmap
        target_categories = [ClassifierOutputTarget(label.item())]
        grayscale_cam = grad_cam(input_tensor=input_tensor, targets=target_categories)[
            0, :
        ]

        # Convert input tensor to numpy image
        input_image = input_tensor.squeeze().cpu().numpy()
        # Normalize to [0, 1] for visualization
        input_image = (input_image + 1) / 2

        # Convert grayscale to RGB for heatmap overlay
        rgb_img = np.stack([input_image, input_image, input_image], axis=2)

        # Overlay heatmap on image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Plot
        plt.subplot(len(samples), 2, 2 * i + 1)
        plt.imshow(input_image, cmap="gray")
        plt.title(
            f"Original Image - Class: {dataloader.dataset.idx_to_class.get(label.item(), str(label.item()))}"
        )
        plt.axis("off")

        plt.subplot(len(samples), 2, 2 * i + 2)
        plt.imshow(visualization)
        plt.title("Grad-CAM Visualization")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(Path(output_dir) / "grad_cam.png", dpi=300)
    plt.close()

    print(f"Grad-CAM visualizations saved to {output_dir}/grad_cam.png")


def visualize_latent_space(model, dataloader, output_dir, config):
    """Visualize 2D projection of latent space"""
    print("Visualizing latent space...")
    device = torch.device(config["device"])

    # Collect latent representations and labels
    latent_vectors = []
    labels = []

    with torch.no_grad():
        for data, label in dataloader:
            data = data.to(device)

            if config["model_type"].lower() == "vae":
                mu, _ = model.encode(data)
                latent_vectors.append(mu.cpu().numpy())
            else:  # GAN
                z = model.encode(data)
                latent_vectors.append(z.cpu().numpy())

            labels.append(label.cpu().numpy())

    # Concatenate all batches
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    labels = np.concatenate(labels, axis=0)

    # Apply dimensionality reduction if latent_dim > 2
    if latent_vectors.shape[1] > 2:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        latent_vectors = pca.fit_transform(latent_vectors)

    # Plot
    plt.figure(figsize=(10, 8))

    classes = np.unique(labels)
    cmap = plt.get_cmap("viridis", len(classes))

    for i, label in enumerate(classes):
        mask = labels == label
        plt.scatter(
            latent_vectors[mask, 0],
            latent_vectors[mask, 1],
            c=[cmap(i)],
            label=dataloader.dataset.idx_to_class.get(label, str(label)),
            alpha=0.7,
        )

    plt.xlabel("Latent Dimension 1")
    plt.ylabel("Latent Dimension 2")
    plt.title(f"Latent Space Visualization ({config['model_type'].upper()})")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.savefig(Path(output_dir) / "latent_space.png", dpi=300)
    plt.close()

    print(f"Latent space visualization saved to {output_dir}/latent_space.png")


def main():
    args = parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get data loaders (test dataset for visualizations)
    _, _, test_loader = get_dataloaders(config)

    # Load model and results
    model, metrics_df, class_metrics_df, confusion_mat, roc_data, pr_data = (
        load_model_and_results(args.model_path, args.results_dir, config)
    )

    # Visualize reconstructions
    visualize_reconstructions(model, test_loader, output_dir, config, args.sample_size)

    # Visualize latent space
    visualize_latent_space(model, test_loader, output_dir, config)

    # Visualization of first layer filters
    if args.filters:
        visualize_conv_filters(model, output_dir, config)

    # Grad-CAM visualization
    if args.grad_cam:
        generate_grad_cam(model, test_loader, output_dir, config, sample_size=5)

    # Visualize metrics
    visualize_confusion_matrix(confusion_mat, output_dir)
    visualize_class_metrics(class_metrics_df, output_dir)
    visualize_roc_curve(roc_data, output_dir)
    visualize_pr_curve(pr_data, output_dir)

    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main()
