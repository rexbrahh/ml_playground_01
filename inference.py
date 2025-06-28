#!/usr/bin/env python3
"""
Inference script for fine-tuned 4-bit quantized models.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned 4-bit model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="./fine_tuned_model",
        help="Path to the fine-tuned model or model name on Hugging Face Hub"
    )
    parser.add_argument(
        "--base_model", 
        type=str, 
        default=None,
        help="Base model name if using a LoRA adapter"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Write a short poem about machine learning.",
        help="Prompt for the model"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=256,
        help="Maximum number of new tokens to generate"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.95,
        help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--use_lora", 
        action="store_true",
        help="Whether to load a LoRA adapter"
    )
    return parser.parse_args()


def generate_text(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7, top_p=0.95):
    """
    Generate text based on a prompt.

    Args:
        model: The model to use for generation
        tokenizer: The tokenizer to use for tokenization
        prompt (str): The prompt to generate from
        max_new_tokens (int): Maximum number of new tokens to generate
        temperature (float): Temperature for sampling
        top_p (float): Top-p sampling parameter

    Returns:
        str: Generated text
    """
    # Format the prompt based on the model's expected format
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    # Tokenize the prompt
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    # Generate text
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract just the response part (remove the instruction part)
    response = generated_text.split("### Response:")[-1].strip()

    return response


def main():
    """
    Run inference with a fine-tuned model.
    """
    args = parse_args()

    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("WARNING: CUDA not available. Inference will be slower.")

    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # If using LoRA, load the base model and adapter
    if args.use_lora:
        if args.base_model is None:
            # Try to determine base model from the adapter config
            try:
                peft_config = PeftConfig.from_pretrained(args.model_path)
                base_model_name = peft_config.base_model_name_or_path
                print(f"Using base model: {base_model_name}")
            except Exception as e:
                print(f"Error loading PEFT config: {e}")
                print("Please specify the base model using --base_model")
                return
        else:
            base_model_name = args.base_model

        # Load base model
        print(f"Loading base model: {base_model_name}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )

        # Load LoRA adapter
        print(f"Loading LoRA adapter: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        # Load the complete model directly
        print(f"Loading model: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            trust_remote_code=True
        )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate text
    print(f"\nPrompt: {args.prompt}")
    print("\nGenerating response...")

    response = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p
    )

    print(f"\nResponse:\n{response}")


if __name__ == "__main__":
    main()
