"""
AI-Powered Training Data Generator

Uses Groq/Grok API to automatically generate high-quality training examples
for the business clarification system.

Generates:
- 30-40 business problems across domains
- Clarification questions for each problem
- Multiple-choice options for each question
- Domain and concept mappings
"""

import json
import os
from typing import List, Dict
from datetime import datetime
from unified_llm_wrapper import UnifiedLLM
from dotenv import load_dotenv

load_dotenv()


class AITrainingGenerator:
    """
    Generates training data using AI (Groq/Grok)
    """
    
    def __init__(self, provider: str = None, output_dir: str = "ai_training_data"):
        """Initialize AI generator"""
        self.llm = UnifiedLLM(provider=provider)
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.training_examples = []
        
        print(f"✅ AI Training Generator initialized")
        print(f"🤖 Using: {self.llm.provider.upper()}")
        print(f"📁 Output: {output_dir}")
    
    def generate_business_problems(self, num_problems: int = 40) -> List[Dict]:
        """Generate diverse business problems using AI"""
        
        print(f"\n{'='*80}")
        print(f"GENERATING {num_problems} BUSINESS PROBLEMS WITH AI")
        print('='*80)
        
        prompt = f"""Generate {num_problems} diverse business problems that a data scientist would help solve.

Cover these domains (distribute evenly):
1. Customer Retention & Churn (8-10 problems)
2. Revenue Growth & Optimization (8-10 problems)
3. Customer Acquisition & Marketing (6-8 problems)
4. Product Strategy & Development (6-8 problems)
5. Customer Segmentation & Analysis (4-6 problems)
6. Operational Efficiency (4-6 problems)

For each problem, provide:
- business_problem: Clear, specific business question
- domain: One of the domains above (use snake_case)
- concepts: 3-5 key concepts (lowercase, single words)
- industry: SaaS, E-commerce, Retail, Finance, etc.

Return as JSON array:
[
  {{
    "business_problem": "How can we reduce customer churn in our SaaS product?",
    "domain": "customer_retention",
    "concepts": ["churn", "retention", "saas", "customers"],
    "industry": "SaaS"
  }},
  ...
]

Make problems realistic, specific, and diverse. Return ONLY the JSON array."""

        print("\n🤖 Generating business problems with AI...")
        print("⏳ This may take 30-60 seconds...")
        
        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are an expert business analyst who understands data science applications across industries.",
                temperature=0.8,
                max_tokens=4000
            )
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            problems = json.loads(response)
            
            print(f"✅ Generated {len(problems)} business problems")
            return problems
        
        except Exception as e:
            print(f"⚠️ AI generation failed: {e}")
            print("📝 Using fallback problems...")
            return self._get_fallback_problems()
    
    def generate_clarification_questions(self, business_problem: str, domain: str, concepts: List[str]) -> List[Dict]:
        """Generate clarification questions for a business problem"""
        
        prompt = f"""For this business problem, generate 3-4 clarification questions that a data scientist would ask.

Business Problem: "{business_problem}"
Domain: {domain}
Key Concepts: {', '.join(concepts)}

For each question, provide:
- question: The clarification question (specific, actionable)
- options: 5-6 multiple-choice options (realistic, covers common scenarios, includes "Other")
- category: metrics, current_situation, goals, constraints, stakeholders, or data
- priority: critical, high, or medium

Return as JSON array:
[
  {{
    "question": "What is your current monthly churn rate?",
    "options": [
      "Less than 2% (excellent)",
      "2-5% (good)",
      "5-10% (needs improvement)",
      "More than 10% (critical)",
      "Not measured yet",
      "Other (please specify)"
    ],
    "category": "metrics",
    "priority": "critical"
  }},
  ...
]

Make questions specific to the problem. Options should be realistic and actionable. Return ONLY the JSON array."""

        try:
            response = self.llm.generate(
                prompt=prompt,
                system_prompt="You are an expert data scientist conducting business understanding interviews.",
                temperature=0.7,
                max_tokens=2000
            )
            
            # Parse JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            questions = json.loads(response)
            return questions
        
        except Exception as e:
            print(f"⚠️ Question generation failed for: {business_problem[:50]}...")
            return self._get_fallback_questions(domain)
    
    def generate_complete_training_data(self, num_problems: int = 40):
        """Generate complete training dataset"""
        
        print(f"\n{'='*80}")
        print(f"GENERATING COMPLETE TRAINING DATASET")
        print('='*80)
        
        # Step 1: Generate business problems
        problems = self.generate_business_problems(num_problems)
        
        # Step 2: Generate questions for each problem
        print(f"\n{'='*80}")
        print(f"GENERATING CLARIFICATION QUESTIONS")
        print('='*80)
        
        for idx, problem in enumerate(problems, 1):
            print(f"\n[{idx}/{len(problems)}] Generating questions for:")
            print(f"    {problem['business_problem'][:70]}...")
            
            questions = self.generate_clarification_questions(
                problem['business_problem'],
                problem['domain'],
                problem['concepts']
            )
            
            # Add to training examples
            self.training_examples.append({
                'id': f"ai_train_{idx:03d}",
                'business_problem': problem['business_problem'],
                'domain': problem['domain'],
                'concepts': problem['concepts'],
                'industry': problem.get('industry', 'General'),
                'clarification_questions': questions,
                'successful_outcome': True,
                'notes': f"AI-generated training example using {self.llm.provider}",
                'created_at': datetime.now().isoformat()
            })
            
            print(f"    ✅ Generated {len(questions)} questions")
        
        print(f"\n✅ Complete! Generated {len(self.training_examples)} training examples")
    
    def save_training_data(self):
        """Save generated training data"""
        filepath = os.path.join(self.output_dir, 'ai_generated_training.json')
        
        data = {
            'metadata': {
                'total_examples': len(self.training_examples),
                'generated_by': self.llm.provider,
                'model': self.llm.model,
                'created_at': datetime.now().isoformat(),
                'version': '1.0'
            },
            'training_examples': self.training_examples
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Training data saved to: {filepath}")
        return filepath
    
    def generate_statistics(self):
        """Generate statistics about generated data"""
        print(f"\n{'='*80}")
        print("TRAINING DATA STATISTICS")
        print('='*80)
        
        # Count by domain
        domains = {}
        industries = {}
        concepts_set = set()
        total_questions = 0
        
        for ex in self.training_examples:
            domain = ex['domain']
            domains[domain] = domains.get(domain, 0) + 1
            
            industry = ex.get('industry', 'General')
            industries[industry] = industries.get(industry, 0) + 1
            
            concepts_set.update(ex['concepts'])
            total_questions += len(ex['clarification_questions'])
        
        print(f"\n📊 Overview:")
        print(f"   • Total Examples: {len(self.training_examples)}")
        print(f"   • Total Questions: {total_questions}")
        print(f"   • Unique Concepts: {len(concepts_set)}")
        print(f"   • Domains Covered: {len(domains)}")
        print(f"   • Industries: {len(industries)}")
        
        print(f"\n📈 By Domain:")
        for domain, count in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {domain}: {count} examples")
        
        print(f"\n🏢 By Industry:")
        for industry, count in sorted(industries.items(), key=lambda x: x[1], reverse=True):
            print(f"   • {industry}: {count} examples")
        
        print(f"\n💡 Top Concepts:")
        concept_counts = {}
        for ex in self.training_examples:
            for concept in ex['concepts']:
                concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        for concept, count in sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:20]:
            print(f"   • {concept}: {count} times")
    
    def _get_fallback_problems(self) -> List[Dict]:
        """Fallback problems if AI generation fails"""
        return [
            {
                "business_problem": "How can we reduce customer churn in our SaaS product?",
                "domain": "customer_retention",
                "concepts": ["churn", "retention", "saas"],
                "industry": "SaaS"
            },
            {
                "business_problem": "How can we increase monthly recurring revenue?",
                "domain": "revenue_growth",
                "concepts": ["mrr", "revenue", "growth"],
                "industry": "SaaS"
            },
            {
                "business_problem": "How can we reduce customer acquisition cost?",
                "domain": "customer_acquisition",
                "concepts": ["cac", "acquisition", "marketing"],
                "industry": "SaaS"
            }
        ]
    
    def _get_fallback_questions(self, domain: str) -> List[Dict]:
        """Fallback questions if AI generation fails"""
        return [
            {
                "question": "What is your current situation?",
                "options": [
                    "Just starting",
                    "Have some data",
                    "Well established",
                    "Facing challenges",
                    "Looking to optimize",
                    "Other (please specify)"
                ],
                "category": "current_situation",
                "priority": "high"
            }
        ]


def main():
    """Main generation function"""
    print("="*80)
    print("AI-POWERED TRAINING DATA GENERATOR")
    print("="*80)
    
    # Initialize generator
    print("\n🚀 Initializing AI generator...")
    generator = AITrainingGenerator()
    
    # Generate training data
    print("\n💡 This will generate 30-40 high-quality training examples")
    print("⏳ Estimated time: 2-3 minutes")
    
    input("\nPress Enter to start generation...")
    
    # Generate
    generator.generate_complete_training_data(num_problems=40)
    
    # Save
    filepath = generator.save_training_data()
    
    # Statistics
    generator.generate_statistics()
    
    print(f"\n{'='*80}")
    print("✅ AI TRAINING DATA GENERATION COMPLETE!")
    print('='*80)
    print(f"\n📁 File: {filepath}")
    print(f"\n🚀 Next Steps:")
    print(f"   1. Review the generated data (optional)")
    print(f"   2. Train the system: python train_knowledge_graph.py")
    print(f"   3. Test improvements: python test_trained_system.py")
    print(f"   4. Use in app: streamlit run crisp_dm_app.py")
    print('='*80)


if __name__ == "__main__":
    main()
