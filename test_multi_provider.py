"""
Test script for multi-provider LLM support
Tests: Groq, Grok (xAI), OpenAI
"""

import os
from dotenv import load_dotenv
from unified_llm_wrapper import UnifiedLLM, get_available_providers

load_dotenv()


def test_provider(provider_name: str):
    """Test a specific provider"""
    print(f"\n{'='*80}")
    print(f"TESTING {provider_name.upper()}")
    print('='*80)
    
    try:
        # Initialize LLM
        print(f"\n1. Initializing {provider_name}...")
        llm = UnifiedLLM(provider=provider_name)
        
        info = llm.get_info()
        print(f"   ✅ Provider: {info['provider']}")
        print(f"   ✅ Model: {info['model']}")
        print(f"   ✅ API Key: {'Set' if info['api_key_set'] else 'Not Set'}")
        
        # Test simple generation
        print(f"\n2. Testing simple generation...")
        response = llm.generate(
            prompt="List 3 key factors for customer retention in one sentence.",
            system_prompt="You are a business analyst.",
            temperature=0.7,
            max_tokens=100
        )
        
        print(f"\n   Response:")
        print(f"   {'-'*76}")
        print(f"   {response}")
        print(f"   {'-'*76}")
        
        # Test chat with history
        print(f"\n3. Testing chat with conversation history...")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is customer churn?"},
            {"role": "assistant", "content": "Customer churn is when customers stop using a service."},
            {"role": "user", "content": "How can we reduce it?"}
        ]
        
        response = llm.chat(messages=messages, temperature=0.7, max_tokens=150)
        
        print(f"\n   Response:")
        print(f"   {'-'*76}")
        print(f"   {response}")
        print(f"   {'-'*76}")
        
        print(f"\n✅ {provider_name.upper()} TEST PASSED!")
        return True
    
    except Exception as e:
        print(f"\n❌ {provider_name.upper()} TEST FAILED!")
        print(f"   Error: {e}")
        return False


def test_clarification_agent():
    """Test clarification agent with current provider"""
    print(f"\n{'='*80}")
    print("TESTING CLARIFICATION AGENT WITH MULTI-PROVIDER SUPPORT")
    print('='*80)
    
    try:
        from clarification_with_graph_rag import GraphEnhancedClarificationAgent
        
        print("\n1. Initializing agent...")
        agent = GraphEnhancedClarificationAgent()
        
        print("\n2. Starting conversation...")
        response = agent.start_conversation(
            business_question="How can we improve customer retention?"
        )
        
        print("\n   Agent Response:")
        print(f"   {'-'*76}")
        print(f"   {response[:500]}...")  # First 500 chars
        print(f"   {'-'*76}")
        
        print("\n✅ CLARIFICATION AGENT TEST PASSED!")
        return True
    
    except Exception as e:
        print(f"\n❌ CLARIFICATION AGENT TEST FAILED!")
        print(f"   Error: {e}")
        return False


def main():
    """Run all tests"""
    print("="*80)
    print("MULTI-PROVIDER LLM SYSTEM TEST")
    print("="*80)
    
    # Check available providers
    print("\n📊 Checking available providers...")
    available = get_available_providers()
    
    if not available:
        print("\n❌ No API keys found!")
        print("\nPlease set at least one API key in .env:")
        print("  • GROQ_API_KEY")
        print("  • GROK_API_KEY")
        print("  • OPENAI_API_KEY")
        return
    
    print(f"\n✅ Available providers: {', '.join(available)}")
    
    # Get current provider
    current_provider = os.getenv("LLM_PROVIDER", "groq")
    print(f"\n🎯 Current provider (from .env): {current_provider.upper()}")
    
    # Test current provider
    print(f"\n{'='*80}")
    print(f"TESTING CURRENT PROVIDER: {current_provider.upper()}")
    print('='*80)
    
    success = test_provider(current_provider)
    
    if success:
        # Test clarification agent
        test_clarification_agent()
    
    # Offer to test other providers
    other_providers = [p for p in available if p != current_provider]
    
    if other_providers:
        print(f"\n{'='*80}")
        print(f"OTHER AVAILABLE PROVIDERS: {', '.join(other_providers).upper()}")
        print('='*80)
        print("\nTo test other providers, update LLM_PROVIDER in .env file:")
        for provider in other_providers:
            print(f"  • LLM_PROVIDER={provider}")
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print('='*80)
    print(f"\n✅ Current Provider: {current_provider.upper()}")
    print(f"✅ Status: {'Working' if success else 'Failed'}")
    print(f"✅ Available Providers: {len(available)}")
    print(f"\n💡 To switch providers, edit .env file:")
    print(f"   LLM_PROVIDER=groq   # Fast, free tier available")
    print(f"   LLM_PROVIDER=grok   # Powerful reasoning (xAI)")
    print(f"   LLM_PROVIDER=openai # GPT models")
    print('='*80)


if __name__ == "__main__":
    main()
