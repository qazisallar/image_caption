# Image Captioning with Transformer-Based Models

This repository implements an end-to-end image captioning system using a Vision Transformer (ViT) encoder and a Transformer decoder. It supports training and inference on the Flickr30k dataset.

# Features
Vision Transformer (ViT) encoder for image feature extraction
Transformer decoder for sequence generation
Custom PyTorch Dataset class
Inference with beam search or greedy decoding
Logging and experiment tracking using Weights & Biases (wandb)
Modular codebase for easy extension

# Installation
1. Clone the repository
```bash
git clone https://github.com/yourusername/image-captioning-transformer.git
cd image-captioning-transformer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```
3. Login to Weights & Biases (optional)
```bash
wandb login
```
# Dataset
This project uses the Flickr30k dataset. The dataset.py script will automatically download and preprocess it.

# Training

To train the model:
```bash
python train.py --epochs 10 --batch_size 32 --lr 1e-4
```
Use --help for more options:
```bash
python train.py --help
```

# Inference
Generate captions for images:
```bash
python predict.py --image_path path/to/image.jpg
```
# Configuration Options

All key hyperparameters can be set via command-line arguments, including:

- --epochs
- --batch_size
- --lr (learning rate)
- --model_save_path
- --num_beams (for beam search in inference)

# Example
Sample command to generate captions using beam search:

python predict.py --image_path sample.jpg --num_beams 5

# Requirements
See requirements.txt for all dependencies​

