📦 Visual AI Image Compression

Autoencoder vs JPEG Comparison

📌 Project Overview

This project explores deep learning–based image compression using a Convolutional Autoencoder, and compares its performance against traditional JPEG compression.

The goal is to analyze how well a neural network can compress and reconstruct images while preserving visual quality, using PSNR (Peak Signal-to-Noise Ratio) as the evaluation metric.

The project is implemented with PyTorch and trained/tested on the CIFAR-10 dataset.

🧠 Key Concepts

Convolutional Autoencoders

Learned image compression

Latent space representation

JPEG compression baseline

PSNR quality metric

Visual comparison of reconstructions

visual-ai-image-compression/
│
├── config.py                    # Global configuration (device, image size, latent dim)
├── model.py                     # CompressionAutoEncoder architecture
├── train.py                     # Autoencoder training script
├── jpeg_vs_autoencoder.py       # JPEG vs Autoencoder comparison & visualization
├── metrics.py                   # PSNR metric
├── compression_autoencoder.pth  # Trained model weights
├── data/                        # CIFAR-10 dataset
└── README.md



🏗️ Autoencoder Architecture

Encoder

3× Convolution layers (stride=2)

ReLU activations

Fully connected layer → Latent vector (compression)

Decoder

Fully connected layer

Transposed convolutions

Sigmoid output for normalized image reconstruction

The latent vector represents the compressed image representation.

📊 Compression Comparison

The project compares:

Method	Description
JPEG	Standard lossy image compression (PIL)
Autoencoder	Learned compression via latent space

Evaluation metric:

PSNR (Peak Signal-to-Noise Ratio)

Higher PSNR → better reconstruction quality.

🖼️ Example Output

Each experiment displays:

Original Image

JPEG-compressed image (with PSNR)

Autoencoder-reconstructed image (with PSNR)

This allows both numerical and visual comparison.

🚀 How to Run
1️⃣ Create Virtual Environment (Optional)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

2️⃣ Install Dependencies
pip install torch torchvision matplotlib pillow

3️⃣ Train the Autoencoder
python train.py


This will generate:

compression_autoencoder.pth

4️⃣ Compare JPEG vs Autoencoder
python jpeg_vs_autoencoder.py

⚙️ Configuration

Edit config.py to experiment with:

LATENT_DIM = 128      # Compression strength
IMAGE_SIZE = 128
DEVICE = "cuda" or "cpu"


Lower latent dimensions = higher compression, lower quality.

💡 Results & Insights

Autoencoders can outperform JPEG in certain visual details

Learned compression adapts to dataset characteristics

Latent space size strongly affects reconstruction quality

Visual artifacts differ significantly between JPEG and neural compression

🧪 Future Improvements

Compression ratio vs PSNR plots

SSIM metric support

Larger datasets (ImageNet)

Variational Autoencoder (VAE)

Learned entropy coding

🧑‍💻 Author

Meriç Yıldırım
Visual AI • Deep Learning • Computer Vision

⭐ Why This Project Matters

This project demonstrates:

Practical understanding of deep learning fundamentals

Ability to compare classical vs learned methods

Hands-on experience with computer vision pipelines

Research-oriented thinking