from privacy.secret_sharing import SecretSharing
from privacy.smpc_aggregator import SMPCAggregator

class FederatedCoordinator:
    def __init__(self, num_clients):
        self.num_clients = num_clients
        self.smpc_aggregator = SMPCAggregator()
        self.secret_sharing = SecretSharing()
        self.canal_smpc = {i: [] for i in range(num_clients)}
        
    def run_federated_round(self, round_num, trainer, dados_clientes_X, dados_clientes_y):
        self.canal_smpc = {i: [] for i in range(self.num_clients)}
        
        for i in range(self.num_clients):
            pesos_D = trainer.train_local_epoch(f"Cliente_{i+1}", round_num, dados_clientes_X[i], dados_clientes_y[i])
            shares_gerados = self.secret_sharing.gerar_shares_smpc_modulares(pesos_D, self.num_clients)
            for id_destino in range(self.num_clients):
                self.canal_smpc[id_destino].append(shares_gerados[id_destino])
                
        shares_para_agregar = [self.canal_smpc[i] for i in range(self.num_clients)]
        pesos_globais_D = self.smpc_aggregator.agregar_shares(shares_para_agregar)
        trainer.D.load_state_dict(pesos_globais_D)
        
        return pesos_globais_D