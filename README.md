# Visual AI – Image Compression with Autoencoders

This project implements an image compression system using a convolutional autoencoder trained on the CIFAR-10 dataset. The goal is to learn compact latent representations of images and compare deep learning–based compression performance against traditional JPEG compression.

## 🚀 Project Overview

- Images are compressed into a low-dimensional latent space using a CNN-based autoencoder.
- The reconstructed images are evaluated and compared with JPEG-compressed images.
- Performance is measured using **MSE (Mean Squared Error)** and **PSNR (Peak Signal-to-Noise Ratio)**.

## 🧠 Model Architecture

The autoencoder consists of:
- **Encoder:** Convolutional layers that downsample the input image into a compact latent vector.
- **Latent Space:** Controls the compression rate.
- **Decoder:** Transposed convolutions that reconstruct the image from the latent vector.

## 📊 Dataset

- **CIFAR-10**
- Image size: 32×32 RGB
- 60,000 images (50,000 training / 10,000 test)

## ⚙️ Technologies Used

- Python 3
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- TQDM

## 🏋️ Training

Train the autoencoder using:

```bash
python train.py


To compare JPEG compression with the autoencoder:

python jpeg_vs_autoencoder.py


This script:

Compresses images using JPEG

Compresses images using the trained autoencoder

Compares reconstruction quality using MSE and PSNR




visual-ai-image-compression/
│
├── config.py
├── dataset.py
├── model.py
├── train.py
├── jpeg_vs_autoencoder.py
├── utils.py
├── requirements.txt
└── README.md


📈 Results

The autoencoder demonstrates competitive performance compared to JPEG compression at similar compression levels, showing the potential of deep learning–based approaches for image compression tasks.



✨ Author

Meriç Yıldırım
