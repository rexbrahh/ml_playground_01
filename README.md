# 4-Bit Quantized Model Training Pipeline

This project provides a pipeline for training and fine-tuning large language models (8-12B parameters) using 4-bit quantization. It allows you to run these large models on consumer hardware by significantly reducing memory requirements.

## Features

- Load 4-bit quantized models from Hugging Face Hub
- Fine-tune models using Parameter-Efficient Fine-Tuning (PEFT) with LoRA
- Prepare datasets for instruction tuning
- Run inference with fine-tuned models

## Requirements

- Python 3.8+
- CUDA-compatible GPU with at least 8GB VRAM (16GB+ recommended)
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Loading a Model

To load a 4-bit quantized model from Hugging Face:

```bash
python model_loader.py
```

This will list available models and load a default model (Mistral-7B).

### Training a Model

To fine-tune a model using LoRA:

```bash
python trainer.py \
  --model_name mistralai/Mistral-7B-v0.1 \
  --dataset_name databricks/databricks-dolly-15k \
  --output_dir ./fine_tuned_model \
  --num_epochs 3 \
  --batch_size 4 \
  --gradient_accumulation_steps 8
```

This will fine-tune the model on the Dolly dataset and save the LoRA adapter.

### Running Inference

To generate text with a fine-tuned model:

```bash
python inference.py \
  --model_path ./fine_tuned_model \
  --base_model mistralai/Mistral-7B-v0.1 \
  --use_lora \
  --prompt "Explain quantum computing in simple terms"
```

## Available Models

Some recommended models for 4-bit quantization (8-12B range):

- `meta-llama/Llama-2-7b-hf` - 7B parameters
- `meta-llama/Llama-2-13b-hf` - 13B parameters
- `mistralai/Mistral-7B-v0.1` - 7B parameters
- `tiiuae/falcon-7b` - 7B parameters
- `tiiuae/falcon-11b` - 11B parameters
- `google/gemma-7b` - 7B parameters
- `google/gemma-2-9b` - 9B parameters

## Memory Requirements

Approximate VRAM requirements for 4-bit quantized models:

- 7B models: ~6-8GB VRAM
- 13B models: ~10-12GB VRAM

Using LoRA significantly reduces memory requirements during fine-tuning.

## Tips

- If you encounter CUDA out-of-memory errors, try:
  - Reducing batch size
  - Increasing gradient accumulation steps
  - Using a smaller model
  - Reducing max sequence length
- For better performance, use a model that's already instruction-tuned
- You may need a Hugging Face token for some models (like Llama-2)

## License

This project is provided under the MIT License. Note that the models themselves may have their own licenses.
