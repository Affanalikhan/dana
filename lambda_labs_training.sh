#!/bin/bash
# Lambda Labs Training Script for Business Understanding AI

echo "Starting Business Understanding AI Training on Lambda Labs"

# Update system
sudo apt update

# Install Python dependencies
pip install torch transformers datasets peft accelerate bitsandbytes anthropic

# Set API key (you'll need to set this)
export ANTHROPIC_API_KEY="your-anthropic-api-key-here"

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Run training
python complete_training_pipeline.py --output-dir ./trained_model --base-model meta-llama/Llama-3.1-8B

echo "Training completed!"
echo "Model saved to: ./trained_model/business_understanding_model"

# Zip for download
tar -czf business_understanding_model.tar.gz -C ./trained_model business_understanding_model

echo "Ready for download: business_understanding_model.tar.gz"