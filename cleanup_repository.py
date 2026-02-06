"""
Repository Cleanup Script

Removes duplicate, outdated, and unnecessary files
Keeps only essential, working components
"""

import os
import shutil

# Files to KEEP (Essential)
KEEP_FILES = {
    # Core Application
    'crisp_dm_app.py',  # Main Streamlit app
    'clarification_with_graph_rag.py',  # Core agent
    'standalone_graph_rag.py',  # Knowledge Graph
    'unified_llm_wrapper.py',  # Multi-provider LLM
    
    # Training System
    'ai_training_generator.py',  # AI training generator
    'train_with_ai_data.py',  # Training script
    'ai_trained_knowledge_graph.json',  # Trained graph
    
    # Testing
    'test_ai_trained_system.py',  # Test AI training
    'test_data_analysis.py',  # Test data analysis
    'test_multi_provider.py',  # Test LLM providers
    
    # Configuration
    '.env',  # API keys
    '.gitignore',  # Git config
    'requirements.txt',  # Dependencies
    
    # Documentation (Essential)
    'README.md',  # Main readme
    'QUICK_START.md',  # Quick start guide
    'AI_TRAINING_SUCCESS.md',  # AI training docs
    'DATA_ANALYSIS_FEATURE.md',  # Data analysis docs
    'GROK_SETUP.md',  # Grok setup
}

# Files to REMOVE (Duplicates/Outdated)
REMOVE_FILES = {
    # Duplicate/Old Apps
    'app.py',  # Old version
    'app_with_graph_rag.py',  # Duplicate
    'streamlit_clarification_app.py',  # Old version
    'streamlit_graph_rag_app.py',  # Old version
    'streamlit_neural_integration.py',  # Not used
    
    # Old Agents
    'business_clarification_agent.py',  # Old version
    'groq_llm.py',  # Replaced by unified_llm_wrapper
    'groq_llm_wrapper.py',  # Old version
    
    # Old Training Files
    'training_data_creator.py',  # Replaced by AI generator
    'train_knowledge_graph.py',  # Old version
    'trained_knowledge_graph.json',  # Old version
    'use_ai_trained_system.py',  # Not needed
    
    # Neural Network Files (Not used in current system)
    'neural_business_understanding_system.py',
    'neural_training_pipeline.py',
    'Neural_Business_Understanding_Training.ipynb',
    'colab_training_complete.py',
    'cloud_neural_training_setup.py',
    'create_demo_neural_data.py',
    'groq_neural_data_generator.py',
    'comprehensive_training_data_generator.py',
    'enhanced_comprehensive_generator.py',
    'run_enhanced_training_generation.py',
    'large_scale_data_generator.py',
    
    # Old Test Files
    'test_trained_system.py',  # Old version
    'test_agent.py',  # Old
    'test_complete_system.py',  # Old
    'test_graph_rag.py',  # Old
    'test_neural_system_demo.py',  # Not used
    'test_business_specific_clarifications.py',  # Old
    'test_contextual_options.py',  # Old
    'test_clickable_options.py',  # Replaced
    'quick_test.py',  # Old
    'final_demo_test.py',  # Old
    
    # Old Documentation
    'CLARIFICATION_SYSTEM_README.md',  # Old
    'CLICKABLE_OPTIONS_GUIDE.md',  # Covered in QUICK_START
    'COMPLETE_SYSTEM_GUIDE.md',  # Old
    'CONTEXTUAL_OPTIONS_FIXED.md',  # Old
    'FINAL_SYSTEM_README.md',  # Old
    'GRAPH_RAG_CLARIFICATION_README.md',  # Old
    'GRAPH_RAG_SETUP.md',  # Old
    'GROK_INTEGRATION_COMPLETE.md',  # Covered in GROK_SETUP
    'IMPLEMENTATION_SUMMARY.md',  # Old
    'MULTI_PROVIDER_GUIDE.md',  # Covered in QUICK_START
    'QUESTION_FORMAT_GUIDE.md',  # Old
    'QUICKSTART_GRAPH_RAG.md',  # Old
    'README_GRAPH_RAG.md',  # Old
    'RUN_SYSTEM.md',  # Old
    'START_GRAPH_RAG.md',  # Old
    'START_HERE.md',  # Old
    'SYSTEM_READY.md',  # Old
    'SYSTEM_STATUS.md',  # Old
    'TRAINING_COMPLETE.md',  # Covered in AI_TRAINING_SUCCESS
    'TRAINING_GUIDE.md',  # Covered in AI_TRAINING_SUCCESS
    
    # Unused Utilities
    'adaptive_engine.py',  # Not used
    'analysis_engine.py',  # Not used
    'context_preservation.py',  # Not used
    'crisp_dm_framework.py',  # Not used
    'error_handling.py',  # Not used
    'example_conversation_templates.py',  # Not used
    'example_graph_rag_demo.py',  # Not used
    'graph_initializer.py',  # Not used
    'graph_rag_pipeline.py',  # Not used
    'hybrid_neural_graph_system.py',  # Not used
    'knowledge_graph_schema.py',  # Not used
    'question_generator.py',  # Not used
    'session_manager.py',  # Not used
    'setup_complete_system.py',  # Not used
    'summary_generator.py',  # Not used
    'visualize_graph.py',  # Not used
    
    # Cloud Setup Files (Not needed for local use)
    'lambda_labs_setup.sh',
    'lambda_labs_training.sh',
    'runpod_setup.py',
    'docker-compose.yml',
    
    # Log Files
    'crisp_dm_errors.log',
}

# Folders to REMOVE
REMOVE_FOLDERS = {
    'demo_neural_training_data',  # Old training data
    'groq_neural_training_data',  # Old training data
    'training_data',  # Old training data
    '.crisp_dm_sessions',  # Old sessions
}

# Folders to KEEP
KEEP_FOLDERS = {
    'ai_training_data',  # AI-generated training data
    '.git',  # Git repository
    '.kiro',  # Kiro settings
    '__pycache__',  # Python cache (will be regenerated)
}


def cleanup_repository():
    """Clean up the repository"""
    print("="*80)
    print("REPOSITORY CLEANUP")
    print("="*80)
    
    removed_files = []
    removed_folders = []
    kept_files = []
    
    # Remove files
    print("\n📁 Removing unnecessary files...")
    for filename in REMOVE_FILES:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                removed_files.append(filename)
                print(f"   ✅ Removed: {filename}")
            except Exception as e:
                print(f"   ❌ Error removing {filename}: {e}")
    
    # Remove folders
    print("\n📁 Removing unnecessary folders...")
    for foldername in REMOVE_FOLDERS:
        if os.path.exists(foldername):
            try:
                shutil.rmtree(foldername)
                removed_folders.append(foldername)
                print(f"   ✅ Removed: {foldername}/")
            except Exception as e:
                print(f"   ❌ Error removing {foldername}: {e}")
    
    # Verify kept files
    print("\n✅ Verifying essential files...")
    for filename in KEEP_FILES:
        if os.path.exists(filename):
            kept_files.append(filename)
            print(f"   ✓ {filename}")
        else:
            print(f"   ⚠️ Missing: {filename}")
    
    # Summary
    print("\n" + "="*80)
    print("CLEANUP SUMMARY")
    print("="*80)
    
    print(f"\n📊 Statistics:")
    print(f"   • Files removed: {len(removed_files)}")
    print(f"   • Folders removed: {len(removed_folders)}")
    print(f"   • Essential files kept: {len(kept_files)}")
    
    print(f"\n✅ Repository Structure:")
    print(f"   • Core Application: crisp_dm_app.py")
    print(f"   • Agent System: clarification_with_graph_rag.py")
    print(f"   • Knowledge Graph: standalone_graph_rag.py")
    print(f"   • LLM Provider: unified_llm_wrapper.py")
    print(f"   • AI Training: ai_training_generator.py")
    print(f"   • Trained Data: ai_trained_knowledge_graph.json")
    
    print(f"\n📚 Documentation:")
    print(f"   • README.md - Main documentation")
    print(f"   • QUICK_START.md - Getting started guide")
    print(f"   • AI_TRAINING_SUCCESS.md - AI training guide")
    print(f"   • DATA_ANALYSIS_FEATURE.md - Data analysis docs")
    print(f"   • GROK_SETUP.md - Grok API setup")
    
    print(f"\n🧪 Testing:")
    print(f"   • test_ai_trained_system.py - Test AI training")
    print(f"   • test_data_analysis.py - Test data analysis")
    print(f"   • test_multi_provider.py - Test LLM providers")
    
    print("\n" + "="*80)
    print("✅ CLEANUP COMPLETE!")
    print("="*80)
    
    print(f"\n🚀 Your repository is now clean and organized!")
    print(f"\n📦 To start using:")
    print(f"   streamlit run crisp_dm_app.py")
    
    return {
        'removed_files': removed_files,
        'removed_folders': removed_folders,
        'kept_files': kept_files
    }


if __name__ == "__main__":
    print("\n⚠️  WARNING: This will delete files permanently!")
    print("Files to remove:", len(REMOVE_FILES))
    print("Folders to remove:", len(REMOVE_FOLDERS))
    
    response = input("\nProceed with cleanup? (yes/no): ")
    
    if response.lower() == 'yes':
        result = cleanup_repository()
    else:
        print("\n❌ Cleanup cancelled")
