#!/bin/bash
# Lambda Labs GPU Training Setup for Neural Business Understanding System

echo "🚀 Setting up Neural Business Understanding System training on Lambda Labs"

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
pip install torch transformers datasets scikit-learn tqdm groq
pip install accelerate bitsandbytes

# Clone repository (replace with your repo)
git clone https://github.com/your-username/neural-business-understanding.git
cd neural-business-understanding

# Set environment variables
export GROQ_API_KEY="your-groq-api-key-here"

# Generate training data
echo "📊 Generating training data..."
python groq_neural_data_generator.py --output-dir ./groq_neural_training_data

# Start training
echo "🏋️ Starting neural model training..."
echo "This will take 4-8 hours on A100 GPU"

python neural_training_pipeline.py \
  --data-dir ./groq_neural_training_data \
  --output-dir ./trained_neural_models \
  --batch-size 16 \
  --epochs 10

# Test trained models
echo "🧪 Testing trained models..."
python -c "
from neural_business_understanding_system import NeuralBusinessUnderstandingSystem
system = NeuralBusinessUnderstandingSystem('./trained_neural_models')
system.load_models()
session_id = system.create_session('How can we reduce customer churn?')
questions = system.get_session_questions(session_id)
print(f'✅ System working! Generated {len(questions)} questions')
"

# Create download package
echo "📦 Creating download package..."
tar -czf trained_neural_models.tar.gz trained_neural_models/

echo "✅ Training complete!"
echo "Download trained_neural_models.tar.gz to use locally"
echo "Total training cost: ~$30-80 (one-time)"
echo "Runtime cost: $0 (local inference)"
