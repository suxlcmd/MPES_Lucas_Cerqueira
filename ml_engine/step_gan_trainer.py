"""
Treinamento local do STEP-GAN (Camada 3), idêntico ao modelo final validado no Teste 4 (v5) da PoC:

- K geradores condicionados por limiares, G_k(z, τ), um por degrau da saída do Discriminador;
- Discriminador graduado: normal real -> 0.9, fraude real -> 0.1 (com peso maior) e amostra do degrau τ -> τ;
- os estados de D e dos K geradores formam o "estado global" que é agregado via SMPC a cada rodada.
"""

import copy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .discriminator import Discriminator
from .generator import Generator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_STEPS = ((0.10, 0.35), (0.35, 0.60), (0.60, 0.85))


@dataclass
class STEPGANConfig:
    latent_dim: int = 100
    batch_size: int = 512
    lr_g: float = 2e-4
    lr_d: float = 1e-4
    weight_decay: float = 1e-4
    local_epochs: int = 7
    fraud_weight: float = 15.0
    normal_target: float = 0.9
    anomaly_target: float = 0.1
    steps: tuple = field(default_factory=lambda: DEFAULT_STEPS)
    cond_dim: int = 16
    graded_discriminator: bool = True
    graded_margin: float = 0.0

    def to_dict(self):
        dados = asdict(self)
        dados["steps"] = [list(degrau) for degrau in self.steps]
        return dados

    @classmethod
    def from_dict(cls, dados):
        dados = dict(dados)
        dados["steps"] = tuple(tuple(degrau) for degrau in dados.get("steps", DEFAULT_STEPS))
        return cls(**{k: v for k, v in dados.items() if k in cls.__dataclass_fields__})


def weighted_bce(pred, target, real_mask, fraud_weight):
    """BCE com peso maior para fraudes reais (alvo < 0.5 em amostras reais)."""
    loss = nn.BCELoss(reduction="none")(pred, target)
    weights = torch.ones_like(target)
    weights[(target < 0.5) & real_mask] = fraud_weight
    return (loss * weights).mean()


def _cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


class STEPGANTrainer:
    """Constrói, treina localmente e avalia o STEP-GAN a partir de um estado global {'D': ..., 'G': [...]}."""

    def __init__(self, input_dim, config=None):
        self.input_dim = int(input_dim)
        self.config = config or STEPGANConfig()

    # ----- construção -----
    def build_discriminator(self, state=None):
        D = Discriminator(self.input_dim).to(device)
        if state is not None:
            D.load_state_dict(state)
        return D

    def build_generator(self, state=None):
        G = Generator(self.input_dim, self.config.latent_dim, self.config.cond_dim).to(device)
        if state is not None:
            G.load_state_dict(state)
        return G

    def init_global_state(self, seed):
        torch.manual_seed(seed)
        return {
            "D": _cpu_state(Discriminator(self.input_dim)),
            "G": [
                _cpu_state(Generator(self.input_dim, self.config.latent_dim, self.config.cond_dim))
                for _ in self.config.steps
            ],
        }

    def sample_tau(self, k, n):
        low, high = self.config.steps[k]
        return torch.rand(n, 1, device=device) * (high - low) + low

    # ----- treino local -----
    def train_local(self, global_state, X, y, seed):
        """Treina um cliente a partir do estado global e devolve (estado local, estatísticas de perda)."""
        cfg = self.config
        K = len(cfg.steps)
        torch.manual_seed(seed)

        D = self.build_discriminator(global_state["D"])
        generators = [self.build_generator(g) for g in global_state["G"]]
        D.train()
        for G in generators:
            G.train()

        opt_D = optim.Adam(D.parameters(), lr=cfg.lr_d, betas=(0.5, 0.999), weight_decay=cfg.weight_decay)
        opts_G = [
            optim.Adam(G.parameters(), lr=cfg.lr_g, betas=(0.5, 0.999), weight_decay=cfg.weight_decay)
            for G in generators
        ]
        loader = DataLoader(
            TensorDataset(torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)),
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

        losses_d, losses_g = [], []
        for _ in range(cfg.local_epochs):
            for i, (real_x, real_y) in enumerate(loader):
                real_x, real_y = real_x.to(device), real_y.to(device)
                batch = real_x.size(0)
                if batch < 2:
                    continue  # BatchNorm dos geradores exige pelo menos 2 amostras
                n_k = max(batch // K, 2)

                real_target = torch.where(
                    real_y.unsqueeze(1) == 0,
                    torch.tensor(cfg.normal_target, device=device),
                    torch.tensor(cfg.anomaly_target, device=device),
                )
                taus = [self.sample_tau(k, n_k) for k in range(K)]
                fakes = [
                    G(torch.randn(n_k, cfg.latent_dim, device=device), tau)
                    for G, tau in zip(generators, taus)
                ]

                # Discriminador graduado: amostra do degrau τ -> τ (ou 0.1 no modo não graduado)
                opt_D.zero_grad()
                loss_real = weighted_bce(
                    D(real_x), real_target, torch.ones_like(real_target, dtype=torch.bool), cfg.fraud_weight
                )
                fake_x = torch.cat([f.detach() for f in fakes])
                if cfg.graded_discriminator:
                    fake_target = torch.cat(taus).sub(cfg.graded_margin).clamp(min=cfg.anomaly_target)
                else:
                    fake_target = torch.full((fake_x.size(0), 1), cfg.anomaly_target, device=device)
                loss_fake = weighted_bce(
                    D(fake_x), fake_target, torch.zeros_like(fake_target, dtype=torch.bool), cfg.fraud_weight
                )
                loss_d = loss_real + loss_fake
                loss_d.backward()
                torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
                opt_D.step()
                losses_d.append(loss_d.item())

                # Geradores (a cada 2 passos do D): condicionamento D(G_k(z, τ)) -> τ
                if i % 2 == 0:
                    for G, opt_G, fake, tau in zip(generators, opts_G, fakes, taus):
                        opt_G.zero_grad()
                        loss_g = nn.BCELoss()(D(fake), tau)
                        loss_g.backward()
                        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
                        opt_G.step()
                        losses_g.append(loss_g.item())

        state = {"D": _cpu_state(D), "G": [_cpu_state(G) for G in generators]}
        stats = {
            "loss_d": float(np.mean(losses_d)) if losses_d else float("nan"),
            "loss_g": float(np.mean(losses_g)) if losses_g else float("nan"),
        }
        return state, stats

    # ----- inferência -----
    def anomaly_scores(self, state, X, batch_size=65536):
        """Escore de anomalia em [0, 1]: 1 - D(x). Não depende do lote avaliado."""
        D = self.build_discriminator(state["D"] if "D" in state else state)
        D.eval()
        saida = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                lote = torch.as_tensor(X[i:i + batch_size], dtype=torch.float32, device=device)
                saida.append(D(lote).cpu().numpy().ravel())
        return 1.0 - np.concatenate(saida) if saida else np.array([])

    def conditioning_adherence(self, state, n=1000, seed=0):
        """Para cada gerador, compara o limiar pedido τ com a saída obtida D(G_k(z, τ))."""
        torch.manual_seed(seed)
        D = self.build_discriminator(state["D"])
        D.eval()
        linhas = []
        with torch.no_grad():
            for k, g_state in enumerate(state["G"]):
                G = self.build_generator(g_state)
                G.eval()
                tau = self.sample_tau(k, n)
                saida = D(G(torch.randn(n, self.config.latent_dim, device=device), tau))
                tau_np, d_np = tau.cpu().numpy().ravel(), saida.cpu().numpy().ravel()
                corr = float(np.corrcoef(tau_np, d_np)[0, 1]) if d_np.std() > 0 else float("nan")
                low, high = self.config.steps[k]
                linhas.append({
                    "Gerador": f"G{k + 1}",
                    "Degrau": f"[{low:.2f}, {high:.2f}]",
                    "D médio": float(d_np.mean()),
                    "Erro absoluto médio": float(np.abs(d_np - tau_np).mean()),
                    "Correlação τ×D": corr,
                })
        return linhas


def copy_state(state):
    return copy.deepcopy(state)
