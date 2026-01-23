# %%
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from inf_train_gen import inf_train_gener
from models2D import Generator, Discriminator
from training2D import Trainer

# Config hyperparameters for training WGANgp
configs = {
    "dataset": "darcy_flow",
    "momentum": 0,
    "gp_weight": 5,
    "batch_size": 512,
    "latent_dim": 2,
    "num_epochs": 256,
    "penalty_type": "monge",
    "num_datapoints": 10000,
    "monotone_penalty": 0.0000,
    "critic_iterations": 5,
    "full_critic_train": 20,
    "critic_learning_rate": 0.01,
    "gradient_penalty_type": "two_sided",
    "gen_hidden_layer_sizes": [
        128, 128, 128
    ],
    "generator_learning_rate": 0.01,
    "discriminator_hidden_layer_sizes": [
        128, 128, 128
    ],
    "device":'cuda'
}

# %%
device = torch.device(configs['device'])

def plot_sample(samples, title=""):
    plt.hist2d(samples[:, 0], samples[:, 1], bins=100)
    if title:
        plt.title(title)
    plt.show()

# Getting data for each step of sequential training 
def mixing(x, y, t):
    return (1 - t) * x + t * y



# %%
# initializing input x and target y
n = configs["num_datapoints"]
x_init = np.random.randn(n, 2).astype(np.float32)
y_target = inf_train_gener("pinwheel", batch_size=n) #.astype(np.float32)
plot_sample(y_target, title="Target Distribution")

# Make a list for generators of seqential training
num_steps = 1
delta_t = 1/num_steps
z = x_init.copy()
generators = []



# %%
alpha_list = [0.2, 0.4, 0.6, 0.8, 1.0]

i = 0
for t in alpha_list:
    print(f"\n=== Training G_{i+1} for alpha={t:.2f} ===")
    # initialize generator and discriminator for each step of sequential training
    generator = Generator(
        condition_dim=0,
        sample_dim=configs['latent_dim'],
        hidden_layer_sizes=configs['gen_hidden_layer_sizes'],
        sample_latent=torch.tensor(z, dtype=torch.float32)
    )

    discriminator = Discriminator(
        condition_dim=0,
        sample_dim=3,
        hidden_layer_sizes=configs['discriminator_hidden_layer_sizes']
    )
    print(generator)
    print(discriminator)
    G_opt = optim.SGD(generator.parameters(), lr=configs['generator_learning_rate'], momentum=configs['momentum'])
    D_opt = optim.SGD(discriminator.parameters(), lr=configs['critic_learning_rate'], momentum=configs['momentum'])
    y_mixed = mixing(x_init, y_target, t)

    z_tensor = torch.tensor(z, dtype=torch.float32)
    y_tensor = torch.tensor(y_mixed, dtype=torch.float32)

    dataset = TensorDataset(z_tensor, y_tensor)
    data_loader = DataLoader(dataset, batch_size=configs['batch_size'], shuffle=True)
    # Train the WGANgp for eacn step of sequential training
    trainer = Trainer(
        generator, discriminator, G_opt, D_opt,
        device=configs['device'],
        print_every=50,
        gp_weight=configs['gp_weight'],
        monotone_penalty=configs['monotone_penalty'],
        penalty_type=configs['penalty_type'],
        gradient_penalty_type=configs['gradient_penalty_type'],
        full_critic_train=configs['full_critic_train'],
        batch_size=configs['batch_size']
    )
    epoch_num = configs['num_epochs']
    trainer.train(data_loader, epoch_num)

    X_input = z_tensor.to(device) 
    generator.eval()
    with torch.no_grad():
        z = generator(X_input).cpu().numpy()

    generators.append(generator)
    plot_sample(z, title=f"Generated After G_{i+1}")
    plot_sample(y_mixed, title=f"Generated After G_{i+1}")
    i = i+1


# %%
# Save the generator sequence
for k in range(int(5)):
    torch.save(generators[k].state_dict(), './generator_for_WGANGP-10000_'+str(k)+'.pth')
    print(k)

# %%



