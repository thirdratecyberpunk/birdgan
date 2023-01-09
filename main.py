# server imports
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests

# PyTorch imports
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

# model imports
from Generator import Generator

# misc
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

checkpoint = config['checkpoint']

workers = config['workers']
batch_size = config['batch_size']
image_size = config['image_size']
nc = config['nc']
nz = config['nz']
ngf = config['ngf']
lr = config['lr']
beta1 = config['beta1']
ngpu = config['ngpu']

# assigning device
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)

# load checkpoint
CheckPoint = torch.load(checkpoint)
# load generator model
netG = Generator(ngpu, nz, ngf, nc).to(device)

# 'epoch': num_epochs,
# 'g_state_dict': netG.state_dict(),
# 'd_state_dict': netD.state_dict(),
# 'optimizerG_state_dict': optimizerG.state_dict(),
# 'optimizerG_state_dict': optimizerG.state_dict(),
# 'G_losses': G_losses,
# 'D_losses': D_losses
netG.load_state_dict(CheckPoint['g_state_dict'])
netG.eval()

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# endpoint which returns a Jinja template containing a bird
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    # endpoint to call bird image from
    url = "http://localhost:8000/bird?num_samples=64"
    # pass bird object to template
    return templates.TemplateResponse("home.html", {"request": request, "url": url})

# endpoint which returns an image of a bird sampled from the Generator model
@app.get("/bird")
def get_bird(num_samples: int = 1):
    # create batch of latent vectors for visualising generator progression
    if (num_samples < 1):
        num_samples = 1
    fixed_noise = torch.randn(num_samples, nz, 1, 1, device=device)
    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    grid = vutils.make_grid(fake, padding=2, normalize=True)
    vutils.save_image(grid, 'grid.jpg')
    return FileResponse('grid.jpg')

