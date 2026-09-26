"""
Checkpoint completo do modelo (Camada 2): pesos do STEP-GAN, pré-processador, limiar calibrado,
métricas e metadados de rastreabilidade num único arquivo .pt.

O arquivo contém apenas tensores e tipos primitivos, e é lido com torch.load(weights_only=True):
um checkpoint adulterado no armazenamento não consegue executar código ao ser carregado.
"""

import io
from dataclasses import dataclass, field

import torch

from ml_engine.step_gan_trainer import STEPGANConfig, STEPGANTrainer
from orchestration.data_preprocessor import DataPreprocessor

BUNDLE_FORMAT = "mpes-stepgan-bundle"
BUNDLE_VERSION = 1


@dataclass
class ModelBundle:
    state: dict
    config: dict
    preprocessor: dict
    threshold: float
    threshold_strategy: str
    reference: dict
    metrics: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    adherence: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def input_dim(self):
        return len(self.preprocessor["feature_names"])

    @property
    def feature_names(self):
        return self.preprocessor["feature_names"]

    def build_trainer(self):
        return STEPGANTrainer(self.input_dim, STEPGANConfig.from_dict(self.config))

    def build_preprocessor(self):
        return DataPreprocessor.from_dict(self.preprocessor)

    def to_bytes(self):
        buffer = io.BytesIO()
        torch.save(
            {
                "format": BUNDLE_FORMAT,
                "version": BUNDLE_VERSION,
                "state": self.state,
                "config": self.config,
                "preprocessor": self.preprocessor,
                "threshold": float(self.threshold),
                "threshold_strategy": self.threshold_strategy,
                "reference": self.reference,
                "metrics": self.metrics,
                "history": self.history,
                "adherence": self.adherence,
                "metadata": self.metadata,
            },
            buffer,
        )
        return buffer.getvalue()

    @classmethod
    def from_bytes(cls, data):
        conteudo = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        if not isinstance(conteudo, dict) or conteudo.get("format") != BUNDLE_FORMAT:
            raise ValueError("O arquivo não é um checkpoint MPES STEP-GAN válido.")
        if conteudo.get("version", 0) > BUNDLE_VERSION:
            raise ValueError("O checkpoint foi gerado por uma versão mais nova da aplicação.")
        campos = {k: conteudo[k] for k in cls.__dataclass_fields__ if k in conteudo}
        return cls(**campos)
