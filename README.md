# MedShapeNet19-PSPooling

Benchmark-ready 3D anatomical shape classification using MedShapeNet19 with PSPooling. This repository implements a fully precomputed graph pooling operator for 3D meshes, designed for efficient GNN-based classification and autoencoder architectures.

## Features

- MedShapeNet19 dataset included as a JSON file with dataset links and metadata
- Script to download and organize the dataset with labels (`download_dataset.py`)
- Precomputed, non-trainable pooling and unpooling for 3D meshes
- PSPooling-inspired architecture for shape classification
- Trained models provided for immediate evaluation

## Repository Structure

- `MedShapeNet19/` — Dataset definition (`MedShapeNet19.json`) and download script (`download_dataset.py`)  
- `precomputation/` — Precompute pooling/unpooling weights  
- `source/` — Core GNN and PSPooling implementations  
- `trained_models/` — Pretrained models for evaluation  
- `requirements.txt` — Packages and their versions required to run this project
- `LICENSE` — License information
## Dataset

Before training, the `dataset/` directory should be populated with anatomical mesh data in `.stl` format, accompanied by precomputed files:

- `edge_index_*.npy`
- `weights_*.npy`
- `reverse_weights_*.npy`
- `_iter*.obj`

These can be generated using the `precompute.py` script
Each sample corresponds to a specific anatomical structure (e.g., femur, liver, trachea).

## Scripts
- `mesh_autoencoder.py` — Pretrains an autoencoder with adjustable layer configuration. Pretrained model can then be used for any suitable downstream task.
- `mesh_classifier.py` — Trains a classifier with adjustable layer configuration
