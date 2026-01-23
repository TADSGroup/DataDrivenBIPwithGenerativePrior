# %%
import torch
import torch.nn as nn
import numpy as np
import darcySolver as ds
import matplotlib.pyplot as plt
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
rng=np.random.default_rng(seed=101)
import torch.optim as optim
import importlib
import torchvision.transforms as transforms
from torchvision import datasets as datasets
import dataloaders
importlib.reload(dataloaders)
from dataloaders import get_darcy_dataloader_transformers
from os import mkdir

from scipy.ndimage import zoom
import time
import cProfile
from scipy.interpolate import griddata

from scipy.interpolate import RectBivariateSpline

# %%
# print(torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

# %%
def plot_image(image_reshaped):
    plt.imshow(image_reshaped, cmap="gray")
    plt.show()

# %%
def plot_darcy_select_point(points, A_choose):
    plt.figure(figsize=(5, 5))
    sc = plt.scatter(A_choose[:, 0], A_choose[:, 1], c=points, cmap='viridis', s=80)
    plt.colorbar(sc, label='Function Value')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.gca().set_aspect('equal')
    plt.title("Function values at 60 selected points")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.show()

# %%
def plot_darcy_contour(testing_image, grid_num):
    testing_image = testing_image.reshape(grid_num, grid_num)

    plt.imshow(
        testing_image,
        cmap="viridis",
        origin="lower",
        extent=[0, 1, 0, 1],
        vmin=0,  # optionally fix color scale
    )
    plt.colorbar(label="Pressure")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("True Solution u (imshow)")
    plt.show()

# %%
# Hyperparameters for WGANgp

latent_size = 64
batch_size = 100
image_side = 16
y_side = 28
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
class Generator(nn.Module):
    def __init__(self, input_dim=latent_size, output_dim=784):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()  # Normalize output between -1 and 1
        )

    def forward(self, z):
        img = self.model(z)
        return img  # Output shape (batch_size, 784)

class Critic(nn.Module):
    def __init__(self, input_dim=latent_size):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)  # Output a single critic score
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

# %%
# Hyperparameters
latent_dim = latent_size  # Same as MNIST flattened size

# Initialize models
G = Generator(input_dim=latent_dim, output_dim=784).cuda()
D = Critic(input_dim=latent_size).cuda()
# print(G)
G.load_state_dict(torch.load('./'+str(latent_size)+'/G_WGANgp.pth'))
G.eval()


# %%
# Check the generator for generating MNIST images
z = torch.randn(batch_size, latent_size).to(device)
fake_images = G(z)
fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
fake_images = fake_images.cpu().data

fig, axes = plt.subplots(10, 10, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    ax.imshow(fake_images[i].reshape(28, 28), cmap='gray')
    ax.axis('off')
plt.tight_layout()
plt.show()

# %%
# Initialize grid for Darcyflow Solver's Preliminary fields
x_grid=np.linspace(0,1,28)
y_grid=np.linspace(0,1,28)
X,Y=np.meshgrid(x_grid,y_grid)

# %%
# Darcysolver
def constant_rhs(x):
    return x[0]*0+1

# %%
def make_datapoint(
        permeability_xgrid,
        permeability_ygrid,
        permeability_field,
        num_eval=100,
        f_rhs=constant_rhs
        ):

    interpolated_permeability_field=RectBivariateSpline(
        permeability_xgrid,
        permeability_ygrid,
        permeability_field,
        kx=2,
        ky=2
        )

    permeability_function=lambda x:interpolated_permeability_field(x[0],x[1],grid=False)

    num_cells=100
    darcy=ds.DarcySolver(num_x_cells=num_cells,num_y_cells=num_cells)


    sol=darcy.solve(
        permeability=permeability_function,
        f_rhs=f_rhs
        )
    
    x_obs=np.linspace(0,1,num_eval)
    y_obs=np.linspace(0,1,num_eval)
    Xobs,Yobs=np.meshgrid(x_obs,y_obs)
    sol_observed=sol(Xobs,Yobs,grid=False)
    return sol_observed

# %%
def Darcy_sol(unresized_images,xgrid, ygrid, grid_size):
    zoom_factor = grid_size / 28
    resized_images = np.array([zoom(img, zoom_factor, order=3) for img in unresized_images])
    params="""
    n=100000
    num_kernel_points=40
    kernel = matern_three_half(0.5)
    eval_points = 100
    """
    
    permeability_fields = resized_images
    # print(permeability_fields.shape)
    observations=[
        make_datapoint(
        xgrid,
        ygrid,
        field,
        num_eval=grid_size
        ) for field in permeability_fields #tqdm(permeability_fields)
    ]

    X_data_observed=np.array(observations)
    return X_data_observed

# %%
mnist = datasets.MNIST(root='./data', train=True, download=True,
                       transform=transforms.ToTensor())

# Get a single image and label
img, label = mnist[7]

# Display the image
plot_image(img.squeeze())

# Check an example MNIST images go through DarcySolver and random picking map A
num_points = 300
A = np.random.rand(num_points, 2)
G_start_time = time.time()
Gx_d = img
G_end_time = time.time()
xgrid=np.linspace(0,1,image_side)
ygrid=np.linspace(0,1,image_side)
xgrid_y = np.linspace(0,1,y_side)
ygrid_y = np.linspace(0,1,y_side)
Gx_d = Gx_d.cpu().detach().numpy()
Gx_d = Gx_d.reshape(-1, 28, 28)
G_testing_image = Gx_d.reshape(28, 28)
print(np.max(G_testing_image))
plot_image(G_testing_image)
print("Min:", np.min(Gx_d))
print("Max:", np.max(Gx_d))
Gx_d = np.exp(Gx_d)
plot_image(Gx_d.reshape(28, 28))
print(f"Total time taken for G: {G_end_time - G_start_time:.5f} seconds")

D_start_time = time.time()
DGx_d = Darcy_sol(Gx_d,xgrid_y, ygrid_y, 28)
print(DGx_d.shape)
DGx_d = DGx_d.reshape(y_side,y_side)

D_end_time = time.time()
print(f"Total time taken for D: {D_end_time - D_start_time:.5f} seconds")
print(DGx_d.shape)
plot_darcy_contour(DGx_d, y_side)
interp = RectBivariateSpline(xgrid_y, ygrid_y, DGx_d)
y = np.array([interp(x, y)[0][0] for x, y in zip(A[:, 0], A[:, 1])])

y = torch.from_numpy(y).to(device)
C = np.eye(latent_dim)
C = torch.tensor(torch.from_numpy(C), dtype=torch.float).to(device)
C_ch = torch.linalg.cholesky(C).to(device)
sigma = 0.2*torch.std(y) 
sigma = torch.tensor(sigma, dtype=torch.float).to(device)
y = y + (sigma*torch.randn(y.shape).to(device))
y_test = y.cpu().detach().numpy()
plot_darcy_select_point(y_test,A)

# %%
# %%
# Define loglikelihood function
def Phi(x):
    x = torch.transpose(x, 0, 1)
    x = G(x)
    x = x.cpu().detach().numpy()
    x = x.reshape(-1, 28, 28)
    x = np.exp((x+1)/2)
    x = Darcy_sol(x,xgrid, ygrid,image_side)
    x = x.reshape(image_side,image_side)
    interp = RectBivariateSpline(xgrid, ygrid, x)
    interped = np.array([interp(x, y)[0][0] for x, y in zip(A[:, 0], A[:, 1])])
    x = torch.tensor(torch.from_numpy(interped), dtype=torch.float).to(device)
    return ((1/(2*(sigma**2))) * (torch.norm(y - x)**2))  

# %%
beta = torch.tensor(0.1).to(device)
# Parameters for MALA
folder = './MCMC_sample/6_20_02STD_250k/'
num_samples = 250000
num_burn = 0.2*num_samples
num_effect = int(num_samples - num_burn)
# Initialize the chain
current_sample = torch.tensor(torch.randn(latent_dim,1), dtype=torch.float) 
# Initial guess for 'x'
samples_MALA = torch.zeros((num_effect,latent_dim,1))
sample_Phi = torch.zeros((num_effect,1))
# [current_sample]
samples_MALA[0] = current_sample
alpha_sum = 0
current_sample = current_sample.to(device)
u = torch.tensor(current_sample, dtype=torch.float)
Phi_u = Phi(u)
# MCMC iteration
for i in tqdm(range(num_samples)):
    xi_n = torch.randn(latent_dim,1,dtype=torch.float).to(device)
    proposal = beta*current_sample+(torch.sqrt(1-beta**2)*C_ch@xi_n)
    u = torch.tensor(current_sample, dtype=torch.float)
    v = torch.tensor(proposal, dtype=torch.float)
    Phi_v = Phi(v)
    alpha = min(1, torch.exp(Phi_u-Phi_v))
    a_n = torch.rand(1)
    a_n = a_n.to(device)
    if a_n <= alpha:
        current_sample = proposal
        Phi_u= Phi_v
    alpha_sum = alpha_sum + alpha
    if i >= num_burn:
        samples_MALA[int(i-num_burn-1)] = current_sample
        sample_Phi[int(i-num_burn-1)] = Phi_u
        if i % 200000 == 0:
            torch.save(samples_MALA, folder+str(i)+'_MCMC_sample.pt')
            torch.save(A, folder+str(i)+'_A.pt')
            torch.save(Gx_d, folder+str(i)+'_Gx_d.pt')
            torch.save(beta, folder+str(i)+'_beta.pt')
            torch.save(sample_Phi, folder+str(i)+'_sample_Phi.pt')

    if i % 100==0:
        alpha_avg = alpha_sum/100
        alpha_sum = 0
        if i < num_burn:
            if alpha_avg >0.4:
                beta = beta/2
            if alpha_avg < 0.2: 
                beta = beta + ((1-beta)/2)
    if i % 1000==0:
        print('iteration:', i)
        print("Alpha_avg: ",alpha_avg)
        print("Beta: ",beta)
    
print(beta)
torch.save(samples_MALA, folder+'MCMC_sample.pt')
torch.save(A, folder+'A.pt')
torch.save(Gx_d, folder+'Gx_d.pt')
torch.save(beta, folder+'beta.pt')
torch.save(sample_Phi, folder+'sample_Phi.pt')