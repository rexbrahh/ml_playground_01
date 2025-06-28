#!/usr/bin/env python3
"""
Training pipeline for fine-tuning 4-bit quantized models from Hugging Face.
"""

import os
import argparse
import torch
import numpy as np
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType
)
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_target_modules_for_model(model_name, model):
    """
    Dynamically determine the target modules for LoRA based on model architecture.

    Args:
        model_name (str): Name or path of the model
        model: The loaded model object

    Returns:
        list: List of target module names for LoRA
    """
    # Default target modules for common architectures
    if any(name in model_name.lower() for name in ["llama", "mistral", "vicuna"]):
        # Llama, Mistral, and Vicuna models
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    elif "falcon" in model_name.lower():
        # Falcon models
        return ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
    elif "mpt" in model_name.lower():
        # MPT models
        return ["Wqkv", "out_proj", "fc1", "fc2"]
    elif "gemma" in model_name.lower():
        # Gemma models
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "gpt-neox" in model_name.lower() or "pythia" in model_name.lower():
        # GPT-NeoX based models
        return ["query_key_value", "dense"]
    elif "gpt2" in model_name.lower():
        # GPT-2 based models
        return ["c_attn", "c_proj", "c_fc"]
    else:
        # For unknown architectures, try to detect attention modules automatically
        logger.info("Model architecture not explicitly supported, attempting to detect attention modules...")

        # Get all named modules
        named_modules = list(model.named_modules())

        # Look for common attention module patterns
        target_modules = []

        # Check for query, key, value projection patterns
        for name, _ in named_modules:
            if any(pattern in name for pattern in ["q_proj", "k_proj", "v_proj", "query", "key", "value"]):
                if name.split(".")[-1] not in target_modules:
                    target_modules.append(name.split(".")[-1])

        # Check for output projection patterns
        for name, _ in named_modules:
            if any(pattern in name for pattern in ["o_proj", "out_proj", "output"]):
                if name.split(".")[-1] not in target_modules:
                    target_modules.append(name.split(".")[-1])

        # If we couldn't find attention modules, fall back to linear layers
        if not target_modules:
            logger.warning("Could not detect specific attention modules, falling back to all linear layers")
            for name, module in named_modules:
                if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
                    target_modules.append(name.split(".")[-1])

            # Deduplicate
            target_modules = list(set(target_modules))

        logger.info(f"Automatically detected target modules: {target_modules}")
        return target_modules


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Fine-tune a 4-bit quantized model using LoRA")
    parser.add_argument(
        "--model_name", 
        type=str, 
        default="mistralai/Mistral-7B-v0.1",
        help="Model name or path on Hugging Face Hub"
    )
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        default="databricks/databricks-dolly-15k",
        help="Dataset name or path on Hugging Face Hub"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./fine_tuned_model",
        help="Directory to save the fine-tuned model"
    )
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default="./models",
        help="Directory to store downloaded models"
    )
    parser.add_argument(
        "--lora_r", 
        type=int, 
        default=16,
        help="LoRA attention dimension"
    )
    parser.add_argument(
        "--lora_alpha", 
        type=int, 
        default=32,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", 
        type=float, 
        default=0.05,
        help="LoRA dropout probability"
    )
    parser.add_argument(
        "--learning_rate", 
        type=float, 
        default=2e-4,
        help="Learning rate for training"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=4,
        help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", 
        type=int, 
        default=3,
        help="Number of epochs for training"
    )
    parser.add_argument(
        "--max_seq_length", 
        type=int, 
        default=512,
        help="Maximum sequence length for training"
    )
    parser.add_argument(
        "--gradient_accumulation_steps", 
        type=int, 
        default=8,
        help="Number of steps for gradient accumulation"
    )
    # Memory optimization options
    parser.add_argument(
        "--use_gradient_checkpointing", 
        action="store_true",
        help="Enable gradient checkpointing to save memory"
    )
    parser.add_argument(
        "--use_dynamic_padding", 
        action="store_true",
        default=True,
        help="Use dynamic padding instead of max_length padding to save memory"
    )
    parser.add_argument(
        "--precision", 
        type=str, 
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Precision for training (fp32, fp16, or bf16)"
    )
    parser.add_argument(
        "--target_modules", 
        type=str, 
        default=None,
        help="Comma-separated list of target modules for LoRA (if not provided, will be auto-detected)"
    )
    # Visualization and monitoring options
    parser.add_argument(
        "--enable_tensorboard", 
        action="store_true",
        help="Enable TensorBoard logging for training visualization"
    )
    parser.add_argument(
        "--enable_wandb", 
        action="store_true",
        help="Enable Weights & Biases logging for training visualization"
    )
    parser.add_argument(
        "--project_name", 
        type=str, 
        default="llm-fine-tuning",
        help="Project name for wandb logging"
    )
    parser.add_argument(
        "--validation_split", 
        type=float, 
        default=0.1,
        help="Fraction of data to use for validation (0.0 to disable)"
    )
    parser.add_argument(
        "--eval_steps", 
        type=int, 
        default=100,
        help="Evaluate model every N steps"
    )
    return parser.parse_args()


def prepare_dataset(dataset_name, tokenizer, max_seq_length=512, use_dynamic_padding=True, validation_split=0.1):
    """
    Prepare dataset for training.

    Args:
        dataset_name (str): Name of the dataset on Hugging Face Hub
        tokenizer: Tokenizer to use for tokenization
        max_seq_length (int): Maximum sequence length
        use_dynamic_padding (bool): Whether to use dynamic padding to save memory
        validation_split (float): Fraction of data to use for validation

    Returns:
        tokenized_dataset: Tokenized dataset ready for training (with optional validation split)
    """
    try:
        # Load dataset
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name)

        if "train" not in dataset:
            raise ValueError(f"Dataset {dataset_name} does not contain a 'train' split")

        # For demonstration, we'll use the 'databricks-dolly-15k' dataset
        # which has columns: instruction, context, response, category

        def format_prompt(example):
            # Format based on the model's expected prompt structure
            # This is a simple template - adjust based on your model and dataset
            instruction = example.get("instruction", "")
            context = example.get("context", "")
            response = example.get("response", "")

            # Skip empty examples
            if not instruction or not response:
                return {"formatted_text": ""}

            # Create a prompt format similar to instruction tuning
            if context:
                prompt = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

            return {"formatted_text": prompt}

        # Apply formatting to each example
        logger.info("Formatting dataset examples...")
        formatted_dataset = dataset.map(format_prompt, remove_columns=dataset["train"].column_names)

        # Filter out empty examples
        formatted_dataset = formatted_dataset.filter(lambda x: len(x["formatted_text"]) > 0)

        # Tokenize the dataset
        logger.info("Tokenizing dataset...")

        # Use different padding strategies based on the use_dynamic_padding flag
        padding_strategy = "longest" if use_dynamic_padding else "max_length"
        logger.info(f"Using padding strategy: {padding_strategy}")

        def tokenize_function(examples):
            return tokenizer(
                examples["formatted_text"],
                padding=padding_strategy,  # Use dynamic padding if requested
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt"
            )

        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["formatted_text"]
        )

        # Split dataset into train/validation if requested
        if validation_split > 0.0:
            logger.info(f"Splitting dataset with validation ratio: {validation_split}")
            train_test_split = tokenized_dataset["train"].train_test_split(test_size=validation_split, seed=42)
            tokenized_dataset["train"] = train_test_split["train"]
            tokenized_dataset["validation"] = train_test_split["test"]
            logger.info(f"Dataset prepared with {len(tokenized_dataset['train'])} training examples and {len(tokenized_dataset['validation'])} validation examples")
        else:
            logger.info(f"Dataset prepared with {len(tokenized_dataset['train'])} training examples (no validation split)")
        
        return tokenized_dataset

    except Exception as e:
        logger.error(f"Error preparing dataset: {str(e)}")
        raise


def train_model(args):
    """
    Train a 4-bit quantized model using LoRA.

    Args:
        args: Command line arguments
    """
    try:
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.cache_dir, exist_ok=True)

        # Check if CUDA is available
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
        else:
            logger.warning("CUDA not available. Training will be extremely slow.")
            if not input("Do you want to continue without GPU? (y/n): ").lower().startswith('y'):
                logger.info("Training aborted.")
                return

        logger.info(f"Loading model: {args.model_name}")

        # Set compute dtype based on precision argument
        if args.precision == "fp32":
            compute_dtype = torch.float32
        elif args.precision == "fp16":
            compute_dtype = torch.float16
        else:  # bf16
            compute_dtype = torch.bfloat16

        logger.info(f"Using {args.precision} precision for training")

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True
        )

        # Load tokenizer
        try:
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name,
                cache_dir=args.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {str(e)}")
            raise

        # Set padding token if not set
        if tokenizer.pad_token is None:
            logger.info("Padding token not found, using EOS token as padding token")
            tokenizer.pad_token = tokenizer.eos_token

        # Load model with quantization
        try:
            logger.info("Loading model with 4-bit quantization...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                device_map="auto",
                quantization_config=quantization_config,
                cache_dir=args.cache_dir,
                trust_remote_code=True
            )
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

        # Enable gradient checkpointing if requested
        if args.use_gradient_checkpointing:
            logger.info("Enabling gradient checkpointing to save memory")
            model.gradient_checkpointing_enable()

        # Prepare model for k-bit training
        model = prepare_model_for_kbit_training(model)

        # Determine target modules for LoRA
        if args.target_modules:
            # Use user-provided target modules
            target_modules = args.target_modules.split(",")
            logger.info(f"Using user-provided target modules: {target_modules}")
        else:
            # Auto-detect target modules based on model architecture
            logger.info("Auto-detecting target modules based on model architecture...")
            target_modules = get_target_modules_for_model(args.model_name, model)
            logger.info(f"Detected target modules: {target_modules}")

        # Define LoRA configuration
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        # Create PEFT (Parameter-Efficient Fine-Tuning) model
        try:
            logger.info("Applying LoRA adapters to model...")
            model = get_peft_model(model, lora_config)
        except Exception as e:
            logger.error(f"Failed to apply LoRA adapters: {str(e)}")
            raise

        # Print trainable parameters info
        model.print_trainable_parameters()

        # Prepare dataset
        tokenized_dataset = prepare_dataset(
            args.dataset_name, 
            tokenizer, 
            max_seq_length=args.max_seq_length,
            use_dynamic_padding=args.use_dynamic_padding,
            validation_split=args.validation_split
        )

        # Create appropriate data collator based on dynamic padding setting
        if args.use_dynamic_padding:
            logger.info("Using DataCollatorForSeq2Seq with dynamic padding")
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=tokenizer,
                padding=True,
                return_tensors="pt"
            )
        else:
            logger.info("Using DataCollatorForLanguageModeling with max_length padding")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False  # We're doing causal language modeling, not masked
            )

        # Set up precision flags for training
        fp16 = args.precision == "fp16"
        bf16 = args.precision == "bf16"

        # Set up reporting based on user preferences
        report_to = []
        if args.enable_tensorboard:
            report_to.append("tensorboard")
            logger.info("TensorBoard logging enabled - view with: tensorboard --logdir=./runs")
        if args.enable_wandb:
            report_to.append("wandb")
            logger.info(f"Weights & Biases logging enabled for project: {args.project_name}")
        
        if not report_to:
            report_to = ["none"]
            logger.info("No visualization tools enabled - training will run without logging")

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=0.01,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=10,
            logging_dir=f"{args.output_dir}/logs",  # Directory for TensorBoard logs
            eval_strategy="steps" if args.validation_split > 0.0 else "no",
            eval_steps=args.eval_steps if args.validation_split > 0.0 else None,
            save_strategy="epoch",
            save_total_limit=2,
            fp16=fp16,
            bf16=bf16,
            gradient_checkpointing=args.use_gradient_checkpointing,
            report_to=report_to,
            run_name=f"{args.model_name.split('/')[-1]}-{args.dataset_name.split('/')[-1]}",
            load_best_model_at_end=True if args.validation_split > 0.0 else False,
            metric_for_best_model="eval_loss" if args.validation_split > 0.0 else None,
            greater_is_better=False if args.validation_split > 0.0 else None,
        )

        # Create Trainer
        from transformers import Trainer

        # Set up validation dataset if available
        eval_dataset = tokenized_dataset.get("validation") if args.validation_split > 0.0 else None

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )

        # Train model
        logger.info("Starting training...")
        trainer.train()

        # Save the final model
        logger.info(f"Saving fine-tuned model to {args.output_dir}")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        logger.info("Training complete!")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise


def main():
    args = parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
