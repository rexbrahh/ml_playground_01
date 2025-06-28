#!/usr/bin/env python3
"""
Script to load 4-bit quantized models from Hugging Face.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_quantized_model(model_name="mistralai/Mistral-7B-v0.1", cache_dir="./models"):
    """
    Load a 4-bit quantized model from Hugging Face.

    Args:
        model_name (str): Name or path of the model on Hugging Face Hub
        cache_dir (str): Directory to store downloaded models

    Returns:
        tuple: (model, tokenizer)
    """
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Loading 4-bit quantized model: {model_name}")

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # Normal Float 4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True  # Use nested quantization for more memory efficiency
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Load model with quantization configuration
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",  # Automatically distribute across available GPUs
        quantization_config=quantization_config,
        cache_dir=cache_dir,
        trust_remote_code=True
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Successfully loaded model and tokenizer")
    print(f"Model size: {model.get_memory_footprint() / (1024 ** 3):.2f} GB")

    return model, tokenizer


def list_available_models():
    """
    List some popular 8-12B models suitable for 4-bit quantization.
    """
    models = [
        # 7-8B Models
        "meta-llama/Llama-2-7b-hf",  # 7B parameters
        "mistralai/Mistral-7B-v0.1",  # 7B parameters
        "tiiuae/falcon-7b",  # 7B parameters
        "mosaicml/mpt-7b",  # 7B parameters

        # 11-13B Models
        "meta-llama/Llama-2-13b-hf",  # 13B parameters
        "tiiuae/falcon-11b",  # 11B parameters
        "google/gemma-7b",  # 7B parameters
        "google/gemma-2-9b",  # 9B parameters
        "google/gemma-2-9b-it",  # 9B instruction-tuned
        "google/gemma-1.1-7b-it",  # 7B instruction-tuned
        "microsoft/phi-2",  # ~2.7B but high performance
    ]

    print("Available models for 4-bit quantization:")
    for model in models:
        print(f" - {model}")

    print("\nNote: You can use any 7B-13B model from Hugging Face with this loader.")
    return models


def main():
    """
    Example usage of the model loader.
    """
    # List available models
    list_available_models()

    # Choose a model
    model_name = "mistralai/Mistral-7B-v0.1"  # ~7B parameters

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("WARNING: CUDA not available. Quantized models perform best with GPU acceleration.")

    # Load model and tokenizer
    model, tokenizer = load_quantized_model(model_name)

    # Test with a simple prompt
    prompt = "The best way to learn machine learning is to"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            **inputs, 
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nPrompt: {prompt}")
    print(f"Response: {response}")


if __name__ == "__main__":
    main()
