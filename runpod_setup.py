# RunPod GPU Training Setup
# Use this in a RunPod PyTorch template

import subprocess
import os

def setup_training():
    print("🚀 Setting up Neural Business Understanding System on RunPod")
    
    # Install dependencies
    subprocess.run(["pip", "install", "transformers", "datasets", "scikit-learn", "tqdm", "groq"])
    
    # Clone repository
    subprocess.run(["git", "clone", "https://github.com/your-username/neural-business-understanding.git"])
    os.chdir("neural-business-understanding")
    
    # Set API key
    os.environ['GROQ_API_KEY'] = "your-groq-api-key-here"
    
    # Generate training data
    print("📊 Generating training data...")
    subprocess.run(["python", "groq_neural_data_generator.py", "--output-dir", "./groq_neural_training_data"])
    
    # Start training
    print("🏋️ Starting training...")
    subprocess.run([
        "python", "neural_training_pipeline.py",
        "--data-dir", "./groq_neural_training_data",
        "--output-dir", "./trained_neural_models",
        "--batch-size", "16",
        "--epochs", "10"
    ])
    
    print("✅ Training complete!")
    print("Download the trained_neural_models folder")

if __name__ == "__main__":
    setup_training()
