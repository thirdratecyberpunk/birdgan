# code that runs the server

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from Generator import Generator

# TODO: load these from config file for both training AND server
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

# assigning device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# create batch of latent vectors for visualising generator progression
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/bird")
def get_bird():
    # load checkpoint
    CheckPoint = torch.load("checkpoints/seed_42_epochs_1_2023-01-0914:26:25")
    # load generator model
    netG = Generator(ngpu, nz, ngf, nc).to(device)

    #    'epoch': num_epochs,
    # 'g_state_dict': netG.state_dict(),
    # 'd_state_dict': netD.state_dict(),
    # 'optimizerG_state_dict': optimizerG.state_dict(),
    # 'optimizerG_state_dict': optimizerG.state_dict(),
    # 'G_losses': G_losses,
    # 'D_losses': D_losses
    netG.load_state_dict(CheckPoint['g_state_dict'])
    netG.eval()

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    grid = vutils.make_grid(fake, padding=2, normalize=True)
    vutils.save_image(grid, 'grid.jpg')
    return FileResponse('grid.jpg')
