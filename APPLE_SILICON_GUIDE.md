# Apple Silicon Training Guide 🍎

**Optimized for M1/M2/M3/M4 Max chips with Metal Performance Shaders**

This guide shows you how to train models efficiently on Apple Silicon without CUDA dependencies.

---

## 🚀 Quick Start

```bash
# Use the Apple Silicon optimized trainer
python trainer_apple_silicon.py \
  --model_name microsoft/DialoGPT-small \
  --enable_tensorboard \
  --validation_split 0.1 \
  --num_epochs 3
```

---

## 🎯 Recommended Models for Apple Silicon

### **Small Models** (Good for M1/M2 base)
- `microsoft/DialoGPT-small` - 117M params ⭐ **Recommended**
- `distilgpt2` - 82M params
- `microsoft/DialoGPT-medium` - 345M params

### **Medium Models** (Good for M1/M2 Pro)
- `gpt2` - 124M params ⭐ **Popular choice**
- `microsoft/DialoGPT-large` - 762M params
- `EleutherAI/gpt-neo-125M` - 125M params

### **Large Models** (Good for M1/M2/M3/M4 Max)
- `EleutherAI/gpt-neo-1.3B` - 1.3B params ⭐ **Best for Max chips**
- `EleutherAI/gpt-neo-2.7B` - 2.7B params (M3/M4 Max only)

---

## ⚡ Key Optimizations for Apple Silicon

### **1. No Quantization**
- Removed bitsandbytes dependency (CUDA-only)
- Uses native PyTorch fp32/fp16 precision
- Better compatibility with Metal Performance Shaders

### **2. Metal Performance Shaders (MPS)**
```python
# Automatically detects and uses MPS
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
```

### **3. Optimized LoRA Settings**
- Lower rank (`r=8`) for efficiency
- Conservative target modules
- Reduced memory footprint

### **4. Efficient Data Processing**
- Smaller dataset subsets (5K examples max)
- Dynamic padding
- Optimized batch sizes

---

## 🔧 Training Examples

### **Basic Training** (M1/M2)
```bash
python trainer_apple_silicon.py \
  --model_name microsoft/DialoGPT-small \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --max_seq_length 256 \
  --num_epochs 3
```

### **Advanced Training** (M1/M2/M3/M4 Max)
```bash
python trainer_apple_silicon.py \
  --model_name EleutherAI/gpt-neo-1.3B \
  --batch_size 2 \
  --gradient_accumulation_steps 8 \
  --max_seq_length 512 \
  --learning_rate 2e-4 \
  --enable_tensorboard \
  --validation_split 0.1
```

### **With Visualization**
```bash
python trainer_apple_silicon.py \
  --model_name gpt2 \
  --enable_tensorboard \
  --enable_wandb \
  --project_name "apple-silicon-experiments" \
  --validation_split 0.1 \
  --eval_steps 25
```

---

## 📊 Performance Expectations

| Chip | Model Size | Training Speed | Memory Usage |
|------|------------|----------------|--------------|
| M1 | DialoGPT-small | ~3-5 min/epoch | ~4-6 GB |
| M1 Pro | GPT-2 | ~5-8 min/epoch | ~8-12 GB |
| M1 Max | GPT-Neo-1.3B | ~15-20 min/epoch | ~16-24 GB |
| M2/M3 Max | GPT-Neo-2.7B | ~25-35 min/epoch | ~24-32 GB |

---

## 🛠️ Troubleshooting

### **MPS Not Available**
```bash
# Check MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

### **Out of Memory**
- Reduce `--batch_size` to 1
- Increase `--gradient_accumulation_steps`
- Use smaller `--max_seq_length`
- Choose a smaller model

### **Slow Training**
- Ensure MPS is being used (check logs)
- Close other memory-intensive applications
- Use smaller datasets for testing

---

## ✅ Verification Commands

### **Test Apple Silicon Trainer**
```bash
# Quick test run
python trainer_apple_silicon.py \
  --model_name microsoft/DialoGPT-small \
  --num_epochs 1 \
  --max_seq_length 128 \
  --batch_size 1
```

### **Check Model Performance**
```bash
# Evaluate trained model
python evaluate_model.py \
  --base_model microsoft/DialoGPT-small \
  --fine_tuned_model ./fine_tuned_model_apple
```

### **Monitor Training**
```bash
# Start TensorBoard
tensorboard --logdir=./fine_tuned_model_apple/logs
```

---

## 🎯 Best Practices for Apple Silicon

### **Memory Management**
- Start with small models and scale up
- Monitor Activity Monitor during training
- Close unnecessary applications

### **Model Selection**
- **M1/M2**: Stick to models < 500M parameters
- **M1/M2 Pro**: Can handle up to 1B parameters  
- **M1/M2/M3/M4 Max**: Can handle 1-3B parameters

### **Training Strategy**
- Use validation splits to monitor progress
- Enable TensorBoard for real-time monitoring
- Start with fewer epochs (1-3) for testing

---

## 🚫 What NOT to Use

- ❌ `bitsandbytes` (CUDA-only)
- ❌ 4-bit/8-bit quantization
- ❌ `device_map="auto"` (use MPS directly)
- ❌ Models larger than your available memory
- ❌ Very large batch sizes

---

## 🎊 Success Indicators

Your Apple Silicon training is working well when you see:

- ✅ "Using Metal Performance Shaders (MPS)" in logs
- ✅ Decreasing loss curves in TensorBoard
- ✅ Memory usage stable (not continuously growing)
- ✅ Training completes without crashes

Happy training on Apple Silicon! 🚀🍎