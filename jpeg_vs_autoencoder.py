import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.datasets import CIFAR10

from model import CompressionAutoEncoder
from config import DEVICE, LATENT_DIM, IMAGE_SIZE
from metrics import psnr

# -----------------------------
# Load trained autoencoder
# -----------------------------
model = CompressionAutoEncoder().to(DEVICE)
model.load_state_dict(
    torch.load("compression_autoencoder.pth", map_location=DEVICE)
)
model.eval()

# -----------------------------
# Dataset
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

dataset = CIFAR10(
    root="./data",
    train=False,
    download=False,
    transform=transform
)

images = [dataset[i][0] for i in range(5)]

# -----------------------------
# JPEG compression helper
# -----------------------------
def jpeg_compress(img_tensor, quality=20):
    img = transforms.ToPILImage()(img_tensor.cpu())
    img.save("temp.jpg", format="JPEG", quality=quality)
    jpeg_img = Image.open("temp.jpg")
    return transforms.ToTensor()(jpeg_img)

# -----------------------------
# Visualization
# -----------------------------
plt.figure(figsize=(12, 6))

for i, img in enumerate(images):
    img = img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        recon, z = model(img)
        recon = recon.cpu().squeeze(0)

    jpeg_img = jpeg_compress(img.cpu().squeeze(0))

    psnr_ae = psnr(img.cpu().squeeze(0), recon)
    psnr_jpeg = psnr(img.cpu().squeeze(0), jpeg_img)

    # Original
    plt.subplot(3, len(images), i + 1)
    plt.imshow(img.cpu().squeeze(0).permute(1, 2, 0))
    plt.title("Original")
    plt.axis("off")

    # JPEG
    plt.subplot(3, len(images), i + 1 + len(images))
    plt.imshow(jpeg_img.permute(1, 2, 0))
    plt.title(f"JPEG\nPSNR={psnr_jpeg:.2f}")
    plt.axis("off")

    # Autoencoder
    plt.subplot(3, len(images), i + 1 + 2 * len(images))
    plt.imshow(recon.permute(1, 2, 0))
    plt.title(f"AE\nPSNR={psnr_ae:.2f}")
    plt.axis("off")

plt.tight_layout()
plt.show()
