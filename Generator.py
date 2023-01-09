# generator maps the latent space vector z to data-space
# creates an RGB image with the same size as the training images
# original example creates a 3*64*64 image
# uses a series of strided 2d convolutional transpose layers,
# which are paired with a 2d batch norm layer (layer that normalises the whole
# batch and a relu activation (returns 0 for negative values, original value for
# positive values)
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils


class Generator(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nz = nz
        self.nfg = ngf
        self.nc = nc
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