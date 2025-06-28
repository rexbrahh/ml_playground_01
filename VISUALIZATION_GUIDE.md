# Training Visualization & Monitoring Guide

This guide shows you how to visualize training progress and measure model improvement during fine-tuning.

## 🎯 Overview

You now have **4 ways** to monitor training progress:

1. **TensorBoard** - Real-time training visualization
2. **Weights & Biases** - Professional ML experiment tracking  
3. **Validation Loss** - Track model performance during training
4. **Evaluation Script** - Before/after model comparison

---

## 📊 Option 1: TensorBoard (Recommended for Beginners)

TensorBoard provides real-time visualization of training metrics.

### Setup & Usage

```bash
# Install TensorBoard (if not already installed)
pip install tensorboard

# Train with TensorBoard enabled
python trainer.py \
  --model_name microsoft/DialoGPT-small \
  --dataset_name databricks/databricks-dolly-15k \
  --enable_tensorboard \
  --validation_split 0.1 \
  --num_epochs 3

# Open TensorBoard in a new terminal (while training is running)
tensorboard --logdir=./fine_tuned_model/logs
```

### What You'll See

- **Training Loss**: Should decrease over time
- **Validation Loss**: Should decrease (if higher than training loss, model is overfitting)
- **Learning Rate**: Shows the learning rate schedule
- **GPU Memory**: Memory usage over time

**Access**: Open browser to `http://localhost:6006`

---

## 🚀 Option 2: Weights & Biases (Best for Advanced Users)

W&B provides professional experiment tracking with beautiful dashboards.

### Setup & Usage

```bash
# Install W&B
pip install wandb

# Login to W&B (creates free account)
wandb login

# Train with W&B enabled
python trainer.py \
  --model_name microsoft/DialoGPT-small \
  --dataset_name databricks/databricks-dolly-15k \
  --enable_wandb \
  --project_name "my-llm-experiments" \
  --validation_split 0.1 \
  --num_epochs 3
```

### What You'll See

- **Real-time metrics** in web dashboard
- **System monitoring** (GPU, CPU, memory usage)
- **Experiment comparison** across different runs
- **Model versioning** and artifact tracking

**Access**: Automatically opens browser to your W&B dashboard

---

## 📈 Option 3: Validation Loss Tracking

Monitor validation loss to detect overfitting and model improvement.

### Usage

```bash
# Enable validation split and evaluation
python trainer.py \
  --model_name microsoft/DialoGPT-small \
  --dataset_name databricks/databricks-dolly-15k \
  --validation_split 0.1 \
  --eval_steps 50 \
  --enable_tensorboard
```

### What to Look For

- **Validation loss decreasing**: Model is improving ✅
- **Validation loss increasing while training loss decreases**: Overfitting ⚠️
- **Both losses plateauing**: Training might be complete ✅

---

## 🎯 Option 4: Model Evaluation (Before/After Comparison)

Measure actual model improvement with the evaluation script.

### Before Training

```bash
# Evaluate base model performance
python evaluate_model.py \
  --base_model microsoft/DialoGPT-small \
  --output_file baseline_evaluation.json
```

### After Training

```bash
# Compare base vs fine-tuned model
python evaluate_model.py \
  --base_model microsoft/DialoGPT-small \
  --fine_tuned_model ./fine_tuned_model \
  --output_file final_evaluation.json
```

### What You'll Get

```json
{
  "base_model": {"average_score": 0.45},
  "fine_tuned_model": {"average_score": 0.72},
  "improvement": {
    "absolute": 0.27,
    "percentage": 60.0,
    "analysis": "Improved"
  }
}
```

---

## 🔥 Complete Training Example with All Monitoring

Here's how to run training with comprehensive monitoring:

```bash
# Full training with all visualization options
python trainer.py \
  --model_name microsoft/DialoGPT-small \
  --dataset_name databricks/databricks-dolly-15k \
  --output_dir ./my_fine_tuned_model \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --validation_split 0.1 \
  --eval_steps 25 \
  --enable_tensorboard \
  --enable_wandb \
  --project_name "my-llm-project" \
  --use_dynamic_padding \
  --precision bf16

# In another terminal, start TensorBoard
tensorboard --logdir=./my_fine_tuned_model/logs

# After training, evaluate improvement
python evaluate_model.py \
  --base_model microsoft/DialoGPT-small \
  --fine_tuned_model ./my_fine_tuned_model \
  --output_file improvement_report.json
```

---

## 📋 Key Metrics to Monitor

### During Training
- **Training Loss**: Should steadily decrease
- **Validation Loss**: Should decrease but may fluctuate
- **Learning Rate**: Follows the schedule (cosine decay)
- **GPU Memory**: Should be stable (no memory leaks)

### Signs of Good Training
- ✅ Training loss consistently decreasing
- ✅ Validation loss generally decreasing  
- ✅ Gap between train/val loss not too large
- ✅ No NaN values or sudden spikes

### Warning Signs
- ⚠️ Validation loss increasing while training loss decreases (overfitting)
- ⚠️ Loss values becoming NaN (learning rate too high)
- ⚠️ Loss plateauing immediately (learning rate too low)
- ⚠️ Memory usage continuously increasing (memory leak)

---

## 🛠️ Troubleshooting

### TensorBoard Not Showing Data
```bash
# Make sure you're pointing to the right directory
tensorboard --logdir=./fine_tuned_model/logs --reload_interval=1
```

### W&B Not Logging
```bash
# Re-login if authentication fails
wandb login --relogin
```

### Evaluation Script Errors
```bash
# Make sure you have the right model paths
ls -la ./fine_tuned_model  # Should contain adapter files
```

---

## 🎊 Success Indicators

Your model is improving when you see:

1. **Decreasing validation loss** in TensorBoard/W&B
2. **Higher evaluation scores** in the evaluation script
3. **Better response quality** in generated text
4. **Stable training** without crashes or NaN values

Happy training! 🚀