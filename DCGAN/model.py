from collections import defaultdict
from typing import Optional
import torch.nn as nn
import torch

import torchvision.utils as vutils
from torch.utils.data import DataLoader
import numpy as np
import imageio
from fastprogress import master_bar, progress_bar

def weights_init(m: nn.Conv2d | nn.ConvTranspose2d |nn.BatchNorm2d):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
class ConvTranspose2dLayer(nn.Module):
    """
    A layer module representing a convolution transpose layer followed by batch normalization and ReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size=4, stride=1, padding=0, output_padding=0, groups: int=1, bias: bool=True) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.seq(input)


class Generator(nn.Module):
    """
    Generator module in GAN, creating images from noise vectors.
    """
    def __init__(self, nz: int, ngf: int, nc: int, num_layers: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            ConvTranspose2dLayer(nz, ngf * (2**(num_layers-2)), 4, 1, 0, bias=False),
            *[ConvTranspose2dLayer(ngf * 2**i, ngf * 2**(i-1), 4, 2, 1, bias=False) for i in range(num_layers-2, 0, -1)],
            nn.Sequential(nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), nn.Sigmoid())
        )

    def forward(self, input: torch.Tensor, lesion_dict: Optional[dict] = None) -> torch.Tensor:
        x = input
        lesion_dict = lesion_dict or defaultdict(list)
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i in lesion_dict:
                x[:, lesion_dict[i]] = 0
        return x

class Discriminator(nn.Module):
    """
    Discriminator module in GAN, distinguishing real images from fake ones generated by the Generator.
    """
    def __init__(self, nc: int, ndf: int, num_layers: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            *[nn.Sequential(
                nn.Conv2d(ndf * 2**i, ndf * 2**(i+1), 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2**(i+1)),
                nn.LeakyReLU(0.2, inplace=True)) for i in range(num_layers-2)],
            nn.Conv2d(ndf * (2**(num_layers - 2)), 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.main(input)
    

def training_step(
    generator: Generator, discriminator: Discriminator, 
    generator_optimizer: torch.optim.Optimizer, discriminator_optimizer: torch.optim.Optimizer, 
    criterion: torch.nn.Module, real_images: torch.Tensor, nz: int,
    real_label: float, fake_label: float, device: torch.device = 'cuda'):
    """
    Performs a single training step for the generator and discriminator.
    """
    ## Train with all-real batch

    real_images = real_images.to(device)
    
    discriminator.zero_grad()
    bs = real_images.shape[0]
    label = torch.full((bs,), real_label, dtype=torch.float, device=real_images.device)
    output = discriminator(real_images)
    err_real = criterion(output.view(-1), label)
    err_real.backward()
    D_x = output.mean().item()

    ## Train with all-fake batch
    noise = torch.randn(bs, nz, 1, 1, device=real_images.device)
    fake = generator(noise)
    label.fill_(fake_label)
    output = discriminator(fake.detach())
    err_fake = criterion(output.view(-1), label)
    err_fake.backward()
    D_G_z1 = output.mean().item()
    discriminator_optimizer.step()
    ## Update generator
    generator.zero_grad()
    label.fill_(real_label)  # fake labels are real for generator cost
    output = discriminator(fake)
    err_g = criterion(output.view(-1), label)
    err_g.backward()
    D_G_z2 = output.mean().item()
    generator_optimizer.step()
    return D_x, D_G_z1, D_G_z2

def train_model(
    generator: Generator, discriminator: Discriminator,
    num_epochs: int, lr: float, data_loader: DataLoader, nz: int, fixed_noise: torch.Tensor,
    save_every: int = 100, device: torch.device = 'cuda'):
    """
    Trains the generator and discriminator for the specified number of epochs.
    """
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    real_label = 1.
    fake_label = 0.

    img_list = []
    iters = 0

    D_x, D_G_z1, D_G_z2 = 0, 0, 0

    mbar = master_bar(range(num_epochs))

    for epoch in mbar:
        mbar.main_bar.comment = f"Epoch {epoch}:"
        for i, (real_images, _) in progress_bar(enumerate(data_loader), leave=False, parent=mbar, total=len(data_loader)):
            mbar.child.comment = f'D(x): {D_x}, D(G(z)): {D_G_z1 + D_G_z2}'
            D_x, D_G_z1, D_G_z2 = training_step(
                generator, discriminator, 
                generator_optimizer, discriminator_optimizer, 
                criterion, real_images, nz, 
                real_label, fake_label
            )
            if iters % save_every == 0:
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                fake = vutils.make_grid(fake, padding=2, normalize=True).numpy()
                fake = np.transpose(fake, (1, 2, 0)) * 255
                fake = fake.astype(np.uint8)
                img_list.append(fake)

                # save img_list as a gif
                imageio.mimsave('dcgan_epoch.gif', img_list, fps=2)
            iters += 1

    return img_list
