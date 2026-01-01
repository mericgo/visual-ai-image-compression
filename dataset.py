from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from config import IMAGE_SIZE, BATCH_SIZE, DATA_DIR

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

def get_loader(train=True):
    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=train,
        download=True,
        transform=transform
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=train,
        num_workers=0  # ⚠️ Windows fix
    )
