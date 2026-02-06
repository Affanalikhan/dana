"""
Unified LLM Wrapper - Supports Multiple Providers
Providers: Groq, Grok (xAI), OpenAI
"""

import os
from typing import Optional, List, Dict
from dotenv import load_dotenv
import requests

load_dotenv()


class UnifiedLLM:
    """
    Unified interface for multiple LLM providers
    
    Supported Providers:
    - groq: Fast inference with Llama models
    - grok: xAI's Grok models (powerful reasoning)
    - openai: OpenAI GPT models
    """
    
    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize LLM with specified provider
        
        Args:
            provider: 'groq', 'grok', or 'openai' (defaults to env LLM_PROVIDER)
            api_key: API key (defaults to env variable)
            model: Model name (defaults to provider's best model)
        """
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq")
        self.api_key = api_key
        self.model = model
        
        # Initialize based on provider
        if self.provider == "groq":
            self._init_groq()
        elif self.provider == "grok":
            self._init_grok()
        elif self.provider == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
        
        print(f"✅ Initialized {self.provider.upper()} with model: {self.model}")
    
    def _init_groq(self):
        """Initialize Groq"""
        from groq import Groq
        
        self.api_key = self.api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        self.client = Groq(api_key=self.api_key)
        self.model = self.model or "llama-3.1-8b-instant"
    
    def _init_grok(self):
        """Initialize Grok (xAI)"""
        self.api_key = self.api_key or os.getenv("GROK_API_KEY")
        if not self.api_key:
            raise ValueError("GROK_API_KEY not found in environment")
        
        # Grok uses OpenAI-compatible API
        self.base_url = "https://api.x.ai/v1"
        self.model = self.model or "grok-beta"
        self.client = None  # Use requests for Grok
    
    def _init_openai(self):
        """Initialize OpenAI"""
        from openai import OpenAI
        
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = self.model or "gpt-4o-mini"
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Generate text using the configured provider
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Generated text
        """
        if self.provider == "groq":
            return self._generate_groq(prompt, system_prompt, temperature, max_tokens, **kwargs)
        elif self.provider == "grok":
            return self._generate_grok(prompt, system_prompt, temperature, max_tokens, **kwargs)
        elif self.provider == "openai":
            return self._generate_openai(prompt, system_prompt, temperature, max_tokens, **kwargs)
    
    def _generate_groq(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate with Groq"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=kwargs.get('timeout', 20.0)
            )
            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"Groq API error: {e}")
    
    def _generate_grok(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate with Grok (xAI)"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get('timeout', 30.0)
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Grok API error: {e}")
    
    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        **kwargs
    ) -> str:
        """Generate with OpenAI"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        
        except Exception as e:
            raise Exception(f"OpenAI API error: {e}")
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Chat with conversation history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional parameters
        
        Returns:
            Generated response
        """
        if self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=kwargs.get('timeout', 20.0)
            )
            return response.choices[0].message.content
        
        elif self.provider == "grok":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=kwargs.get('timeout', 30.0)
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
    
    def get_info(self) -> Dict:
        """Get provider information"""
        return {
            'provider': self.provider,
            'model': self.model,
            'api_key_set': bool(self.api_key)
        }


# Convenience functions
def create_llm(provider: Optional[str] = None, **kwargs) -> UnifiedLLM:
    """Create LLM instance with specified provider"""
    return UnifiedLLM(provider=provider, **kwargs)


def get_available_providers() -> List[str]:
    """Get list of available providers based on API keys"""
    providers = []
    
    if os.getenv("GROQ_API_KEY"):
        providers.append("groq")
    if os.getenv("GROK_API_KEY"):
        providers.append("grok")
    if os.getenv("OPENAI_API_KEY"):
        providers.append("openai")
    
    return providers


# Test function
if __name__ == "__main__":
    print("="*80)
    print("UNIFIED LLM WRAPPER TEST")
    print("="*80)
    
    # Check available providers
    available = get_available_providers()
    print(f"\n✅ Available providers: {', '.join(available)}")
    
    # Test with default provider
    print(f"\n📊 Testing with default provider...")
    
    try:
        llm = UnifiedLLM()
        
        response = llm.generate(
            prompt="What are the top 3 factors that cause customer churn in SaaS businesses? Be brief.",
            system_prompt="You are a business analyst expert.",
            temperature=0.7,
            max_tokens=200
        )
        
        print(f"\n✅ Response from {llm.provider}:")
        print("-"*80)
        print(response)
        print("-"*80)
        
        print(f"\n✅ Test successful!")
        print(f"Provider: {llm.provider}")
        print(f"Model: {llm.model}")
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have set the API key in .env file")
    
    print("\n" + "="*80)
