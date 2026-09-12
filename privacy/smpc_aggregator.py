import torch
from .secret_sharing import MODULUS, fixed_point_to_float

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SMPCAggregator:
    def agregar_shares(self, shares_dos_clientes):
        num_clientes = len(shares_dos_clientes)
        if num_clientes == 0: return None
        
        pesos_agregados = {}
        chaves = shares_dos_clientes[0][0].keys()
        
        for key in chaves:
            soma_total_int = 0
            for cliente_shares in shares_dos_clientes:
                for share in cliente_shares:
                    if key in share:
                        soma_total_int = torch.remainder(soma_total_int + share[key], MODULUS)
                        
            mask_negative = soma_total_int > (MODULUS // 2)
            soma_total_int = torch.where(mask_negative, soma_total_int - MODULUS, soma_total_int)
            soma_total_float = fixed_point_to_float(soma_total_int)
            pesos_agregados[key] = (soma_total_float / num_clientes).to(device)
            
        return pesos_agregados