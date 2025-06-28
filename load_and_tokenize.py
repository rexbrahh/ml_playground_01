#!/usr/bin/env python3
"""
Script to load and tokenize text using Hugging Face transformers.
Demonstrates basic text tokenization with various pre-trained models.
"""

from transformers import AutoTokenizer, AutoModel
import torch

def load_and_tokenize_text(text, model_name="bert-base-uncased"):
    """
    Load a tokenizer and model, then tokenize the input text.
    
    Args:
        text (str): Text to tokenize
        model_name (str): Name of the pre-trained model to use
    
    Returns:
        dict: Dictionary containing tokenized results and model info
    """
    print(f"Loading tokenizer and model: {model_name}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    print(f"Original text: {text}")
    print(f"Text length: {len(text)} characters")
    
    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    print(f"Tokens: {tokens}")
    print(f"Number of tokens: {len(tokens)}")
    
    # Encode the text (convert to input IDs)
    encoded = tokenizer.encode(text, return_tensors="pt")
    print(f"Encoded input IDs: {encoded}")
    print(f"Input IDs shape: {encoded.shape}")
    
    # Decode back to text
    decoded = tokenizer.decode(encoded[0])
    print(f"Decoded text: {decoded}")
    
    # Get attention mask and token type IDs if available
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    print(f"Full tokenizer output keys: {inputs.keys()}")
    
    # Get model embeddings
    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        print(f"Hidden state shape: {last_hidden_state.shape}")
    
    return {
        "original_text": text,
        "tokens": tokens,
        "encoded": encoded,
        "decoded": decoded,
        "inputs": inputs,
        "hidden_state_shape": last_hidden_state.shape,
        "model_name": model_name
    }

def compare_tokenizers(text, model_names=None):
    """
    Compare tokenization across different models.
    
    Args:
        text (str): Text to tokenize
        model_names (list): List of model names to compare
    """
    if model_names is None:
        model_names = [
            "bert-base-uncased",
            "gpt2",
            "distilbert-base-uncased"
        ]
    
    print(f"\n{'='*60}")
    print("COMPARING TOKENIZERS")
    print(f"{'='*60}")
    
    for model_name in model_names:
        print(f"\n{'-'*40}")
        print(f"Model: {model_name}")
        print(f"{'-'*40}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            tokens = tokenizer.tokenize(text)
            encoded = tokenizer.encode(text)
            
            print(f"Tokens ({len(tokens)}): {tokens}")
            print(f"Token IDs: {encoded}")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")

def main():
    """Main function to demonstrate tokenization."""
    
    # Sample texts to tokenize
    sample_texts = [
        "Hello, how are you today?",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world of artificial intelligence.",
        "This is a longer sentence that contains multiple clauses and should demonstrate how tokenizers handle complex text with punctuation, numbers like 123, and various linguistic patterns."
    ]
    
    print("HUGGING FACE TRANSFORMERS - TEXT TOKENIZATION DEMO")
    print("="*60)
    
    # Tokenize each sample text
    for i, text in enumerate(sample_texts, 1):
        print(f"\n{'#'*60}")
        print(f"EXAMPLE {i}")
        print(f"{'#'*60}")
        
        try:
            result = load_and_tokenize_text(text)
            print(f"✓ Successfully processed: {result['model_name']}")
            
        except Exception as e:
            print(f"✗ Error processing text: {str(e)}")
    
    # Compare different tokenizers on one text
    comparison_text = "The artificial intelligence revolution is here!"
    compare_tokenizers(comparison_text)

if __name__ == "__main__":
    main()