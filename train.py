import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_loader
from model import CompressionAutoEncoder
from config import DEVICE, EPOCHS, LR, MODEL_PATH
from tqdm import tqdm

loader = get_loader(True)
model = CompressionAutoEncoder().to(DEVICE)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    total_loss = 0
    for imgs, _ in tqdm(loader):
        imgs = imgs.to(DEVICE)

        recon, _ = model(imgs)
        loss = criterion(recon, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)
print("✅ Compression model saved")
