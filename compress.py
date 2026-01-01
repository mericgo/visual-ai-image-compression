import torch
import matplotlib.pyplot as plt
from model import CompressionAutoEncoder
from dataset import get_loader
from config import DEVICE, MODEL_PATH

model = CompressionAutoEncoder().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

loader = get_loader(False)
images, _ = next(iter(loader))
images = images.to(DEVICE)

with torch.no_grad():
    recon, latent = model(images)

print("Latent vector size:", latent.shape)

plt.figure(figsize=(10,4))
for i in range(5):
    plt.subplot(2,5,i+1)
    plt.imshow(images[i].permute(1,2,0).cpu())
    plt.axis("off")

    plt.subplot(2,5,i+6)
    plt.imshow(recon[i].permute(1,2,0).cpu())
    plt.axis("off")

plt.show()
