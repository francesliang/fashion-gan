#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib
matplotlib.use('PS')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import config as cfg
from dcgan.weights import weights_init
from dcgan.generator import Generator
from dcgan.discriminator import Discriminator


### Seed
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


### Load data
dataset = dset.ImageFolder(root=cfg.dataroot,
    transform=transforms.Compose([
        transforms.Resize(cfg.image_size),
        transforms.CenterCrop(cfg.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
)


dataloader = torch.utils.data.DataLoader(dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.workers)


device = torch.device("cuda:0" if (torch.cuda.is_available() and cfg.ngpu > 0) else "cpu")


### Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(2,5))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(), (1,2,0)))
plt.savefig("training_images.png")


### Create the generator
netG = Generator(cfg.ngpu).to(device)
if (device.type == 'cuda') and (cfg.gpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
netG.apply(weights_init)
print(netG)


### Create the Discriminator
netD = Discriminator(cfg.ngpu).to(device)
if (device.type == 'cuda') and (cfg.gpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))
netD.apply(weights_init)
print(netD)


### Loss functions and optimiser
criterion = nn.BCELoss()

# Create batch of latent vectors that we will use to visualize
#  the progression of the generator
fixed_noise = torch.randn(64, cfg.nz, 1, 1, device=device)

real_label = 1
fake_label = 0

optimiserD = optim.Adam(netD.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))
optimiserG = optim.Adam(netG.parameters(), lr=cfg.lr, betas=(cfg.beta1, 0.999))


### Training
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting Training")
for epoch in range(cfg.num_epochs):
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)
        # Forward pass real batch through D
        output = netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        noise = torch.randn(b_size, cfg.nz, 1, 1, device=device)
        # Generate fake image batch with G
        fake = netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake
        # Update D
        optimiserD.step()


        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimiserG.step()


        # Output training stats
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, cfg.num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        G_losses.append(errG.item())
        D_losses.append(errD.item())


        # Check how the generator is doing by saving G's output on fixed_noise
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1

# Visualise
fig = plt.figure(figsize=(8,8))
plt.axis("off")
for ind, i in enumerate(img_list):
    plt.imshow(np.transpose(i, (1,2,0)))
    plt.savefig("result_batch_{}.png".format(str(ind)))
