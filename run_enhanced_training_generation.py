"""
Run Enhanced Training Data Generation
Execute the comprehensive training data generation with all requirements
"""

import os
import sys
from pathlib import Path
from enhanced_comprehensive_generator import EnhancedComprehensiveGenerator

def main():
    """Run the enhanced training data generation"""
    
    print("🚀 ENHANCED NEURAL TRAINING DATA GENERATION")
    print("="*60)
    
    # Check API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables")
        print("\nPlease add your Groq API key to the .env file:")
        print("GROQ_API_KEY=your_api_key_here")
        return False
    
    print("✅ API key found")
    
    # Initialize generator
    try:
        generator = EnhancedComprehensiveGenerator(api_key)
        print("✅ Generator initialized")
    except Exception as e:
        print(f"❌ Error initializing generator: {e}")
        return False
    
    # Generate dataset
    try:
        print("\n🎯 Starting comprehensive dataset generation...")
        print("This will generate training data covering:")
        print("   • 8 business domains (SaaS, Retail, Finance, Healthcare, etc.)")
        print("   • 66+ total conversations")
        print("   • 1,300+ strategic business questions")
        print("   • Progressive questioning with 5-7 questions per batch")
        print("   • At least 6 of 8 dimension categories covered")
        print("   • Contextual multiple-choice options")
        print("   • Adaptive follow-up questions")
        
        confirm = input("\nProceed with generation? (y/N): ").lower().strip()
        if confirm != 'y':
            print("Generation cancelled.")
            return False
        
        stats = generator.generate_complete_dataset()
        
        # Validate results
        if stats['compliance_score'] >= 90:
            print(f"\n🎉 SUCCESS! Dataset generated with {stats['compliance_score']:.1f}% compliance")
            print("\n📊 Final Statistics:")
            print(f"   Total Conversations: {stats['generation_metadata']['total_conversations']}")
            print(f"   Total Questions: {stats['generation_metadata']['total_questions']}")
            print(f"   Domains Covered: {stats['domain_coverage']['domains_generated']}/8")
            print(f"   Dimensions Covered: {stats['dimension_coverage']['dimensions_covered']}/8")
            print(f"   Average Questions per Conversation: {stats['quality_metrics']['avg_questions_per_conversation']:.1f}")
            
            print(f"\n✅ Requirements Met:")
            for req, status in stats['requirements_compliance'].items():
                print(f"   {req.replace('_', ' ').title()}: {'✅' if status else '❌'}")
            
            print(f"\n📁 Files generated in: ./enhanced_training_data/")
            print("Ready for neural model training!")
            return True
        else:
            print(f"\n⚠️ Warning: Dataset compliance is {stats['compliance_score']:.1f}%")
            print("Some requirements may not be fully met.")
            return False
            
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)