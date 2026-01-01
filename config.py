import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMAGE_SIZE = 128
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3

LATENT_DIM = 256

DATA_DIR = "./data"
MODEL_PATH = "./compression_autoencoder.pth"
