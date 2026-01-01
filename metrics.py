import torch
import math

def psnr(x, x_hat):
    mse = torch.mean((x - x_hat) ** 2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))
