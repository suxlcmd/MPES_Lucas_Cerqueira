import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .generator import Generator
from .discriminator import Discriminator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_bce_loss(pred, target, real_labels_flag):
    loss = nn.BCELoss(reduction='none')(pred, target)
    weights = torch.ones_like(target)
    is_fraud_mask = (target < 0.5) & real_labels_flag 
    weights[is_fraud_mask] = 15.0 
    return (loss * weights).mean()

class STEPGANTrainer:
    def __init__(self, input_dim, config):
        self.config = config
        self.input_dim = input_dim
        self.latent_dim = config['latent_dim']
        self.G = Generator(input_dim, self.latent_dim).to(device)
        self.D = Discriminator(input_dim).to(device)
        self.logs = []
        
    def train_local_epoch(self, client_id, round_num, dados_X, dados_y):
        self.G.train()
        self.D.train()
        opt_G = optim.Adam(self.G.parameters(), lr=self.config['lr_g'], weight_decay=1e-4)
        opt_D = optim.Adam(self.D.parameters(), lr=self.config['lr_d'], weight_decay=1e-4)
        
        batch_size = self.config['batch']
        dataset = TensorDataset(torch.FloatTensor(dados_X).to(device), torch.FloatTensor(dados_y).to(device))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        loss_g_avg, loss_d_avg = 0.0, 0.0
        
        for i, (real_data, real_y) in enumerate(loader):
            batch_s = real_data.size(0)
            real_labels = torch.where(real_y.unsqueeze(1) == 0, torch.tensor(0.9).to(device), torch.tensor(0.1).to(device))
            fake_labels = torch.zeros(batch_s, 1).to(device) + 0.1
            
            opt_D.zero_grad()
            out_real = self.D(real_data)
            loss_D_real = custom_bce_loss(out_real, real_labels, torch.ones_like(real_labels, dtype=torch.bool))
            
            z = torch.randn(batch_s, self.latent_dim).to(device)
            fake_data = self.G(z)
            out_fake = self.D(fake_data.detach())
            loss_D_fake = custom_bce_loss(out_fake, fake_labels, torch.zeros_like(fake_labels, dtype=torch.bool))
            
            loss_D = loss_D_real + loss_D_fake
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(self.D.parameters(), max_norm=1.0)
            opt_D.step()
            
            if i % 2 == 0:
                opt_G.zero_grad()
                out_fake_g = self.D(fake_data)
                loss_G = nn.BCELoss()(out_fake_g, torch.ones(batch_s, 1).to(device) * 0.9)
                loss_G.backward()
                torch.nn.utils.clip_grad_norm_(self.G.parameters(), max_norm=1.0)
                opt_G.step()
                loss_g_avg = loss_G.item()
            
            loss_d_avg = loss_D.item()
            
        self.logs.append({"client": client_id, "round": round_num, "loss_g": loss_g_avg, "loss_d": loss_d_avg})
        return self.D.state_dict()