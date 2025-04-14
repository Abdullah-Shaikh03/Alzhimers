# Medical Image Anomaly Detection

This project implements anomaly detection for medical brain MRI images using unsupervised deep learning approaches. The framework supports both Variational Autoencoders (VAE) and Generative Adversarial Networks (GAN) for anomaly detection.

## Project Structure

```
.
└── Anomaly/
    ├── Dataset                # Contains Dataset/
    │   ├── AD                 # Alzheimer's Disease
    │   ├── CN                 # Cognitively Normal (control)
    │   ├── EMCI               # Early Mild Cognitive Impairment
    │   └── LMCI               # Late Mild Cognitive Impairment
    ├── Logs                   # contains the training logs/
    │   ├── run_001
    │   └── ......
    ├── Model                  # Contain trained models
    ├── Scripts                # All the required Scripts/
    │   ├── Analysis.py        # Save test matrices after the training
    │   ├── DataLoader.py      # Load dataset
    │   ├── Model.py           # Contains Model architecture (VAE and GAN)
    │   ├── Losses.py          # Focal Loss implementation
    │   ├── Config.yaml        # Contains hyperparameters (batch size, LR, optimizer, epochs, etc.)
    │   ├── Train.py           # Training Script with adaptive LR scheduler and AdamW optimizer
    │   └── Visualize.py       # Output first conv layer filters, gradCAM, AUC, ROC, confusion matrix, etc.
    ├── .gitignore
    └── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abdullah-shaikh03/Alzhimer.git
cd medical-anomaly-detection
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset Preparation

The project expects brain MRI images organized in the following structure:
```
Dataset/
├── AD/      # Alzheimer's Disease images
├── CN/      # Cognitively Normal (control) images
├── EMCI/    # Early Mild Cognitive Impairment images
└── LMCI/    # Late Mild Cognitive Impairment images
```

Supported image formats:
- NIfTI files (.nii, .nii.gz)
- Standard image formats (.png, .jpg)

## Configuration

Project settings are controlled through the `Config.yaml` file. Key parameters include:

- `model_type`: Model architecture to use ('vae' or 'gan')
- `device`: Computing device ('cuda' or 'cpu')
- `batch_size`: Batch size for training
- `image_size`: Target image size for processing
- `learning_rate`: Learning rate for optimization
- `max_epochs`: Maximum training epochs
- `anomaly_threshold`: Threshold for anomaly classification

## Training

Train a model using:

```bash
python Scripts/Train.py --config Scripts/Config.yaml [--model_type vae/gan] [--epochs N] [--batch_size N] [--run_id my_run]
```

Options:
- `--config`: Path to configuration file
- `--model_type`: Override model type from config ('vae' or 'gan')
- `--epochs`: Override max epochs from config
- `--batch_size`: Override batch size from config
- `--run_id`: Custom identifier for this training run

Training progress is logged to the `Logs` directory and TensorBoard.

## Analysis

Analyze model performance using:

```bash
python Scripts/Analysis.py --config Scripts/Config.yaml --model_path Model/run_001/best_model.pt --output_dir Results/run_001 --normal_class CN
```

Options:
- `--config`: Path to configuration file
- `--model_path`: Path to trained model checkpoint
- `--output_dir`: Directory to save analysis results
- `--normal_class`: Class to treat as normal (default: 'CN')

## Visualization

Generate visualizations using:

```bash
python Scripts/Visualize.py --config Scripts/Config.yaml --model_path Model/run_001/best_model.pt --results_dir Results/run_001 --output_dir Visualizations/run_001 --grad_cam --filters
```

Options:
- `--config`: Path to configuration file
- `--model_path`: Path to trained model checkpoint
- `--results_dir`: Path to analysis results
- `--output_dir`: Directory to save visualizations
- `--grad_cam`: Generate Grad-CAM visualizations
- `--filters`: Visualize first layer filters
- `--sample_size`: Number of samples for visualizations

## Model Architectures

### Variational Autoencoder (VAE)

The VAE implementation includes:
- Convolutional encoder and decoder networks
- KL divergence regularization
- Customizable latent dimension

### Generative Adversarial Network (GAN)

The GAN implementation includes:
- Generator and discriminator networks
- Feature matching for improved training stability
- Latent space encoding for anomaly detection

## Requirements

- Python 3.8+
- PyTorch 1.8+
- torchvision
- nibabel (for NIfTI file support)
- scikit-learn
- matplotlib
- seaborn
- pandas
- numpy
- PyYAML
- tqdm
- tensorboard
- pytorch-grad-cam (for Grad-CAM visualizations)
- opencv-python

## Citation

If you use this code for your research, please cite:

```
@misc{medical-anomaly-detection,
  author = {Your Name},
  title = {Medical Image Anomaly Detection},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/yourusername/medical-anomaly-detection}}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.