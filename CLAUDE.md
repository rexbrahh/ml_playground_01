# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is an ML playground repository for experimenting with machine learning libraries and frameworks, particularly focused on Hugging Face transformers and text processing.

## Development Setup

### Virtual Environment
```bash
# Activate the virtual environment
source venv/bin/activate

# Deactivate when done
deactivate
```

### Dependencies
The project uses Hugging Face transformers with PyTorch backend:
- `transformers` - Main library for pre-trained models
- `torch` - PyTorch backend
- `datasets` - For dataset loading and processing

### Running Scripts
```bash
# Run the tokenization demo
python load_and_tokenize.py
```

## Project Structure

- `load_and_tokenize.py` - Demonstrates text tokenization using various pre-trained models (BERT, GPT-2, DistilBERT)
- `trainer.py` - Main training pipeline for fine-tuning 4-bit quantized models with LoRA
- `model_loader.py` - Utility to load 4-bit quantized models from Hugging Face
- `inference.py` - Run inference with fine-tuned models
- `evaluate_model.py` - Evaluate model improvement before/after fine-tuning
- `VISUALIZATION_GUIDE.md` - Complete guide for monitoring training progress
- `venv/` - Python virtual environment with ML dependencies

## Training Visualization & Monitoring

The training pipeline supports multiple visualization options:

### TensorBoard (Built-in)
```bash
python trainer.py --enable_tensorboard --validation_split 0.1
tensorboard --logdir=./fine_tuned_model/logs
```

### Weights & Biases  
```bash
pip install wandb
python trainer.py --enable_wandb --project_name "my-experiments"
```

### Model Evaluation
```bash
# Compare base vs fine-tuned model performance
python evaluate_model.py --base_model microsoft/DialoGPT-small --fine_tuned_model ./fine_tuned_model
```

See `VISUALIZATION_GUIDE.md` for detailed instructions.