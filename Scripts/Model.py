# Model.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Basic convolutional block with batch normalization and optional dropout"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bn=True, dropout=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x, 0.2)
        x = self.bn(x)
        x = self.dropout(x)
        return x


class VAE(nn.Module):
    """Variational Autoencoder for anomaly detection in medical images"""

    def __init__(self, input_channels=1, image_size=128, latent_dim=128, filters=[32, 64, 128, 256]):
        super(VAE, self).__init__()

        self.input_channels = input_channels
        self.latent_dim = latent_dim

        # Calculate output size of encoder
        self.encoder_output_size = image_size // (2 ** len(filters))
        self.final_filters = filters[-1]

        # Encoder
        encoder_layers = []
        in_channels = input_channels
        for f in filters:
            encoder_layers.append(
                ConvBlock(in_channels, f, kernel_size=4, stride=2,
                          padding=1, use_bn=True, dropout=0.1)
            )
            in_channels = f

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent representation
        flattened_dim = self.final_filters * \
            self.encoder_output_size * self.encoder_output_size
        self.fc_mu = nn.Linear(flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(flattened_dim, latent_dim)

        # Decoder input layers
        self.decoder_input = nn.Linear(latent_dim, flattened_dim)

        # Decoder
        decoder_layers = []
        reversed_filters = list(reversed(filters))

        for i in range(len(reversed_filters)-1):
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        reversed_filters[i],
                        reversed_filters[i+1],
                        kernel_size=4,
                        stride=2,
                        padding=1
                    ),
                    nn.BatchNorm2d(reversed_filters[i+1]),
                    nn.LeakyReLU(0.2)
                )
            )

        # Final layer
        decoder_layers.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    reversed_filters[-1],
                    input_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                nn.Tanh()  # Output in range [-1, 1]
            )
        )

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        x = x.view(-1, self.final_filters, self.encoder_output_size,
                   self.encoder_output_size)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def generate(self, num_samples=1):
        """Generate new samples by sampling from latent space"""
        z = torch.randn(num_samples, self.latent_dim).to(
            next(self.parameters()).device)
        samples = self.decode(z)
        return samples


class Generator(nn.Module):
    """Generator for GAN-based anomaly detection"""

    def __init__(self, latent_dim=100, output_channels=1, image_size=128, filters=[256, 128, 64, 32]):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        # Initial size for convolution layers
        initial_size = image_size // (2 ** (len(filters) - 1))

        # Initial projection and reshape
        self.init_size = initial_size
        self.l1 = nn.Linear(
            latent_dim, filters[0] * initial_size * initial_size)

        # Transposed convolution layers
        layers = []
        for i in range(len(filters) - 1):
            layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        filters[i], filters[i+1],
                        kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(filters[i+1]),
                    nn.LeakyReLU(0.2, inplace=True)
                )
            )

        # Final layer to get to the desired output channels
        layers.append(
            nn.Sequential(
                nn.Conv2d(filters[-1], output_channels,
                          kernel_size=3, padding=1),
                nn.Tanh()  # Output in range [-1, 1]
            )
        )

        self.model = nn.Sequential(*layers)

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        out = self.model(out)
        return out


class Discriminator(nn.Module):
    """Discriminator for GAN-based anomaly detection"""

    def __init__(self, input_channels=1, image_size=128, filters=[32, 64, 128, 256]):
        super(Discriminator, self).__init__()

        # Feature extraction layers
        layers = []
        in_channels = input_channels
        for f in filters:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, f, kernel_size=4,
                              stride=2, padding=1),
                    # No BN on first layer
                    nn.BatchNorm2d(
                        f) if in_channels != input_channels else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Dropout2d(0.2)
                )
            )
            in_channels = f

        self.feature_extractor = nn.Sequential(*layers)

        # Calculate output size
        ds_size = image_size // (2 ** len(filters))

        # Classification layer
        self.classifier = nn.Sequential(
            nn.Linear(filters[-1] * ds_size * ds_size, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        features = self.feature_extractor(img)
        features = torch.flatten(features, start_dim=1)
        validity = self.classifier(features)
        return validity, features


class GAN(nn.Module):
    """Combined GAN model for anomaly detection"""

    def __init__(self, latent_dim=100, image_channels=1, image_size=128):
        super(GAN, self).__init__()

        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, image_channels, image_size)
        self.discriminator = Discriminator(image_channels, image_size)

    def generate(self, num_samples=1):
        """Generate samples from random noise"""
        device = next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.generator(z)

    def encode(self, x):
        """Finds the latent vector that best reconstructs x"""
        device = next(self.parameters()).device
        z = torch.randn(x.size(0), self.latent_dim,
                        requires_grad=True, device=device)
        z_optimizer = torch.optim.Adam([z], lr=0.1)

        self.generator.eval()
        self.discriminator.eval()

        for i in range(100):  # Optimization steps
            z_optimizer.zero_grad()
            x_fake = self.generator(z)
            loss = F.mse_loss(x_fake, x)
            loss.backward()
            z_optimizer.step()

        return z.detach()

    def reconstruct(self, x):
        """Reconstruct input by finding optimal latent representation"""
        z = self.encode(x)
        return self.generator(z)


def get_model(config):
    """Get model based on configuration"""
    model_type = config['model_type'].lower()
    image_size = config['image_size']

    if model_type == 'vae':
        model = VAE(
            input_channels=config.get('input_channels', 1),
            image_size=image_size,
            latent_dim=config.get('latent_dim', 128),
            filters=[32, 64, 128, 256]
        )
    elif model_type == 'gan':
        model = GAN(
            latent_dim=config.get('latent_dim', 100),
            image_channels=config.get('input_channels', 1),
            image_size=image_size
        )
    else:
        raise ValueError(
            f"Unknown model type: {model_type}. Expected 'vae' or 'gan'")

    return model


# Example usage
if __name__ == "__main__":
    # Test VAE
    vae = VAE(input_channels=1, image_size=128)
    x = torch.randn(4, 1, 128, 128)
    recon, mu, logvar = vae(x)
    print(f"VAE input shape: {x.shape}, reconstruction shape: {recon.shape}")

    # Test GAN
    gan = GAN(latent_dim=100, image_channels=1, image_size=128)
    z = torch.randn(4, 100)
    gen_img = gan.generator(z)
    validity, features = gan.discriminator(gen_img)
    print(
        f"GAN generated image shape: {gen_img.shape}, discriminator output shape: {validity.shape}")
