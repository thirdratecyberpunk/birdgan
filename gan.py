# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 11:36:51 2021

@author: purple
"""

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
import datetime

parser = argparse.ArgumentParser(description="Train a GAN to generate images of birds.")
parser.add_argument('--seed', type = int, default = 42, help="Value used as the seed for random generators")
parser.add_argument('--epochs', type= int, default = 100, help="Number of epochs to run")

args = parser.parse_args()
# seed for RNG
manual_seed = args.seed
# number of training epochs
num_epochs = args.epochs

print(f"Seed: {manual_seed}")
random.seed(manual_seed)
torch.manual_seed(manual_seed)

# root directory for dataset
data_root = "./dataset"
# number of workers for dataloader
workers = 0
# batch size
batch_size = 128
# spatial size of training images
# images will be forced into this size
image_size = 64
# number of channels (for colour images, this value is 3)
nc = 3
# size of z latent vector (i.e. generator input)
nz = 100
# size of feature maps in generator
ngf = 64
# size of feature maps in discriminator
ndf = 64
# learning rate for optimisers
lr = 0.0002
# beta1 hyperparam for Adam optimisers
beta1 = 0.5
# number of GPUs (0 uses CPU)
ngpu = 1

# loading in dataset

dataset = dset.ImageFolder(root=data_root,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                           ]))

# creating the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size,
                                         shuffle=True, num_workers = workers)

# assigning device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# custom weights initialisation
# according to the DCGAN paper, model weights are randomly initialised from a
# normal distribution with mean=0, stdev=0.02

def weights_init(m):
    """
    Initialises weights for generator and discriminator
    @param m values of weights
    """
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        
# generator maps the latent space vector z to data-space
# creates an RGB image with the same size as the training images
# original example creates a 3*64*64 image
# uses a series of strided 2d convolutional transpose layers,
# which are paired with a 2d batch norm layer (layer that normalises the whole
# batch and a relu activation (returns 0 for negative values, original value for
# positive values)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is the z vector
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # second layer gets smaller
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4 , 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # third layer
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # fourth layer takes you to ngf
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # fifth layer takes you to an image
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()            
        )
        
    def forward(self, input):
        return self.main(input)
    
# create an instance of the generator class
netG = Generator(ngpu).to(device)

# can parallelise the process if there are sufficient GPUs
if (device.type == "cuda" and ngpu > 1):
    netG == nn.DataParallel(netG, list(range(ngpu)))
    
# apply the weight initialisation
netG.apply(weights_init)

# discriminator is a binary classification network
# takes in an image, outputs a scalar probability that the input image is real
# D takes in 3*64*64 image, processes through Conv2d, BatchNorm2d and LeakyReLU
# then outputs final probability via Sigmoid (turns into a probability distribution)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is nc * 64 * 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # second layer
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # third layer
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # fourth layer
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # final layer outputs probability
            nn.Conv2d(ndf * 8 , 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self,input):
        return self.main(input)
    
    
# creating the discriminator
netD = Discriminator(ngpu).to(device)

# handling multi-gpu
if (device.type == "cuda" and ngpu > 1):
    netD == nn.DataParallel(netD, list(range(ngpu)))
    
# initialise the weights
netD.apply(weights_init)

# configuring loss
# using binary cross entropy
# uses logarithmic value based on the probability of 
# the expected classification
criterion = nn.BCELoss()

# ctreate batch of latent vectors for visualising generator progression
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# real and fake category labels
real_label = 1.
fake_label = 0.

# Adam optimisers for G and D
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

# training loop
# progress lists
img_list = []
G_losses = []
D_losses = []
iters = 0

print("Starting training loop")
for epoch in range(num_epochs):
    # iterates over the entire dataset
    for i, data in enumerate(dataloader, 0):
        # updates discriminator
        # train with an all real batch
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # passes real batch through discriminator
        output = netD(real_cpu).view(-1)
        # calculates loss on REAL batch
        errD_real = criterion(output, label)
        # calculates gradients for D
        errD_real.backward()
        D_x = output.mean().item()
        
        # train with an all fake batch
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # generate fake batch with the generator
        fake = netG(noise)
        label.fill_(fake_label)
        # classify the fake batch via the discriminator
        output = netD(fake.detach()).view(-1)
        # calculate the discriminator's loss for the fake batch
        errD_fake = criterion(output, label)
        # calculate gradients for this batch, summed with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # computer discriminator error as the sum of real and fake batches
        errD = errD_real + errD_fake
        # update discriminator based on this error
        optimizerD.step()
        
        # updates generator
        netG.zero_grad()
        label.fill_(real_label)
        # passing all fake batch through discriminator again
        output = netD(fake).view(-1)
        # calculate generator's loss for an output
        errG = criterion(output, label)
        # calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # update G
        optimizerG.step()
        
        # training stats
        if i % 50 == 0:
            print(f"{epoch}/{num_epochs}, {i}/{len(dataloader)}, errD : {errD.item()}, errG : {errG.item()}")
            
        # saving losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        
        # saving G's output on fixed_noise to check how it's doing
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            # disables gradient update to save memory
            with torch.no_grad():
                fake = netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))
            
        iters += 1

# saving model that can be loaded into separate session
torch.save(netG.state_dict(), f"models/generator_seed_{manual_seed}_epochs_{num_epochs}_{datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')}")
torch.save(netD.state_dict(), f"models/discriminator_seed_{manual_seed}_epochs_{num_epochs}_{datetime.datetime.now().strftime('%Y-%m-%d%H:%M:%S')}")


# visualisation of results and loss
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# animation of training results on fixed noise
fig = plt.figure(figsize=(8,8))
plt.axis("off")
ims = [[plt.imshow(np.transpose(i,(1,2,0)), animated=True)] for i in img_list]
ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())

# real vs non-real images
# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

# Plot the fake images from the last epoch
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1],(1,2,0)))
plt.show()