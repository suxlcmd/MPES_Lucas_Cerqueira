import torch

PRECISION_BITS = 24
SCALING_FACTOR = 2 ** PRECISION_BITS
MODULUS = 2147483647 

def float_to_fixed_point(tensor):
    return (tensor * SCALING_FACTOR).to(torch.int64)

def fixed_point_to_float(tensor):
    return tensor.to(torch.float32) / SCALING_FACTOR

class SecretSharing:
    def gerar_shares_smpc_modulares(self, state_dict, num_shares):
        shares = [{} for _ in range(num_shares)]
        for key, tensor in state_dict.items():
            tensor_int = float_to_fixed_point(tensor.cpu())
            tensor_int = torch.remainder(tensor_int, MODULUS)
            
            random_tensors = [torch.randint(0, MODULUS, tensor_int.size(), dtype=torch.int64) for _ in range(num_shares - 1)]
            soma_ruidos = sum(random_tensors) if random_tensors else 0
            ultima_share = torch.remainder(tensor_int - soma_ruidos, MODULUS)
            
            for i in range(num_shares - 1):
                shares[i][key] = random_tensors[i]
            shares[-1][key] = ultima_share
        return shares