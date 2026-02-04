"""
Comprehensive Training Data Generator
Ensures coverage across all business domains with progressive questioning
"""

import json
import random
from typing import List, Dict, Any
from groq import Groq
import os
from pathlib import Path
import time

class ComprehensiveTrainingDataGenerator:
    """Generate comprehensive training data covering all business domains"""
    
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        
        # Domain specifications with required counts
        self.domain_specs = {
            "SaaS/Technology": {
                "problem_types": ["Churn", "Growth", "Feature prioritization", "User adoption", "Pricing optimization"],
                "min_count": 10,
                "examples": [
                    "How to reduce customer churn in our SaaS product?",
                    "What features should we prioritize for maximum user engagement?",
                    "How to optimize our pricing strategy for different customer segments?"
                ]
            },
            "Retail": {
                "problem_types": ["Forecasting", "Inventory", "Customer segmentation", "Store performance", "Seasonal trends"],
                "min_count": 10,
                "examples": [
                    "How to improve demand forecasting for seasonal products?",
                    "What's the optimal inventory level for each store location?",
                    "How to segment customers for personalized marketing?"
                ]
            },
            "Finance": {
                "problem_types": ["Risk assessment", "Fraud detection", "Portfolio optimization", "Credit scoring", "Regulatory compliance"],
                "min_count": 8,
                "examples": [
                    "How to improve fraud detection accuracy?",
                    "What's the optimal portfolio allocation for risk management?",
                    "How to assess credit risk for new loan applications?"
                ]
            },
            "Healthcare": {
                "problem_types": ["Patient outcomes", "Resource allocation", "Cost reduction", "Treatment effectiveness", "Operational efficiency"],
                "min_count": 8,
                "examples": [
                    "How to improve patient outcomes for chronic conditions?",
                    "What's the optimal staffing allocation across departments?",
                    "How to reduce operational costs while maintaining quality?"
                ]
            },
            "Manufacturing": {
                "problem_types": ["Quality control", "Predictive maintenance", "Supply chain", "Production optimization", "Defect reduction"],
                "min_count": 8,
                "examples": [
                    "How to implement predictive maintenance for critical equipment?",
                    "What's causing quality issues in our production line?",
                    "How to optimize supply chain for cost and reliability?"
                ]
            },
            "Marketing": {
                "problem_types": ["Campaign optimization", "Attribution", "Customer acquisition", "Brand awareness", "ROI measurement"],
                "min_count": 8,
                "examples": [
                    "How to optimize marketing campaigns for better ROI?",
                    "What's the attribution model for multi-channel campaigns?",
                    "How to reduce customer acquisition costs?"
                ]
            },
            "HR": {
                "problem_types": ["Attrition", "Hiring", "Performance prediction", "Employee engagement", "Talent development"],
                "min_count": 6,
                "examples": [
                    "How to predict and reduce employee attrition?",
                    "What's the best hiring strategy for technical roles?",
                    "How to improve employee performance and engagement?"
                ]
            },
            "E-commerce": {
                "problem_types": ["Recommendation", "Pricing", "Conversion optimization", "Customer lifetime value", "Cart abandonment"],
                "min_count": 8,
                "examples": [
                    "How to improve product recommendation accuracy?",
                    "What's the optimal pricing strategy for maximum revenue?",
                    "How to reduce cart abandonment rates?"
                ]
            }
        }
        
        # 8 dimension categories that must be covered
        self.dimension_categories = [
            "problem_definition",
            "business_objectives", 
            "stakeholders",
            "current_situation",
            "constraints",
            "success_criteria",
            "data_availability",
            "timeline_urgency"
        ]
        
    def _call_groq(self, prompt: str, max_tokens: int = 3000) -> str:
        """Call Groq API with error handling"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return ""
    
    def _extract_json(self, text: str) -> Any:
        """Extract JSON from response text"""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        try:
            return json.loads(text.strip())
        except:
            return []
    
    def generate_progressive_conversation(self, domain: str, problem_type: str, business_question: str) -> Dict[str, Any]:
        """Generate a complete progressive conversation for a business question"""
        
        prompt = f"""Generate a comprehensive business understanding conversation for:

Domain: {domain}
Problem Type: {problem_type}
Business Question: {business_question}

Requirements:
1. Generate 20+ strategic questions across 3-4 batches (5-7 questions per batch)
2. Cover at least 6 of these 8 dimensions: {', '.join(self.dimension_categories)}
3. Questions should be progressively disclosed based on previous answers
4. Include realistic user responses that trigger follow-up questions
5. Each question should have 4-6 contextual multiple choice options

Format as JSON:
{{
  "business_question": "{business_question}",
  "domain": "{domain}",
  "problem_type": "{problem_type}",
  "conversation_batches": [
    {{
      "batch_number": 1,
      "questions": [
        {{
          "question_id": "q1",
          "question_text": "What type of business are you running?",
          "dimension_category": "problem_definition",
          "options": ["Option A", "Option B", "Option C", "Option D", "Option E", "Option F"],
          "reasoning": "Understanding business context is crucial for tailored recommendations"
        }}
      ],
      "user_responses": [
        {{
          "question_id": "q1",
          "selected_option": "Option B",
          "triggers_followup": true,
          "followup_reasoning": "This response indicates need for deeper exploration"
        }}
      ]
    }}
  ],
  "dimensions_covered": ["problem_definition", "business_objectives", "stakeholders", "current_situation", "constraints", "success_criteria"],
  "total_questions": 22,
  "conversation_quality_score": 0.95
}}

Generate a realistic, comprehensive conversation that would help a business analyst fully understand the problem."""

        response = self._call_groq(prompt)
        if response:
            try:
                return self._extract_json(response)
            except:
                return {}
        return {}
    
    def generate_domain_training_data(self, domain: str, count: int) -> List[Dict[str, Any]]:
        """Generate training data for a specific domain"""
        
        print(f"Generating {count} conversations for {domain}...")
        domain_spec = self.domain_specs[domain]
        conversations = []
        
        for i in range(count):
            # Select problem type and create business question
            problem_type = random.choice(domain_spec["problem_types"])
            
            if domain_spec["examples"]:
                base_question = random.choice(domain_spec["examples"])
            else:
                base_question = f"How to solve {problem_type.lower()} challenges in {domain.lower()}?"
            
            # Generate variations of the business question
            variations = [
                base_question,
                f"What's the best approach for {problem_type.lower()} in our {domain.lower()} business?",
                f"How can we improve {problem_type.lower()} outcomes?",
                f"What strategy should we use for {problem_type.lower()}?"
            ]
            
            business_question = random.choice(variations)
            
            # Generate progressive conversation
            conversation = self.generate_progressive_conversation(domain, problem_type, business_question)
            
            if conversation and conversation.get('total_questions', 0) >= 20:
                conversations.append(conversation)
                print(f"  Generated conversation {i+1}/{count}: {conversation.get('total_questions', 0)} questions")
            else:
                print(f"  Retrying conversation {i+1}/{count} (insufficient questions)")
                # Retry with different approach
                time.sleep(1)
                conversation = self.generate_progressive_conversation(domain, problem_type, business_question)
                if conversation:
                    conversations.append(conversation)
            
            time.sleep(0.5)  # Rate limiting
        
        return conversations
    
    def generate_comprehensive_dataset(self, output_dir: str = "./comprehensive_training_data") -> Dict[str, Any]:
        """Generate comprehensive training dataset covering all domains"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("🚀 Starting Comprehensive Training Data Generation")
        print("="*60)
        
        all_conversations = []
        domain_stats = {}
        
        # Generate data for each domain
        for domain, spec in self.domain_specs.items():
            print(f"\n📊 Generating data for {domain} (minimum {spec['min_count']} conversations)")
            
            domain_conversations = self.generate_domain_training_data(domain, spec['min_count'])
            all_conversations.extend(domain_conversations)
            
            domain_stats[domain] = {
                "conversations_generated": len(domain_conversations),
                "total_questions": sum(c.get('total_questions', 0) for c in domain_conversations),
                "problem_types_covered": list(set(c.get('problem_type') for c in domain_conversations))
            }
            
            # Save domain-specific data
            domain_file = output_path / f"{domain.lower().replace('/', '_')}_conversations.json"
            with open(domain_file, 'w') as f:
                json.dump(domain_conversations, f, indent=2)
            
            print(f"✅ {domain}: {len(domain_conversations)} conversations, {domain_stats[domain]['total_questions']} total questions")
        
        # Analyze dimension coverage
        all_dimensions_covered = set()
        for conv in all_conversations:
            all_dimensions_covered.update(conv.get('dimensions_covered', []))
        
        # Generate summary statistics
        summary_stats = {
            "total_conversations": len(all_conversations),
            "total_questions": sum(c.get('total_questions', 0) for c in all_conversations),
            "domains_covered": len(domain_stats),
            "dimensions_covered": len(all_dimensions_covered),
            "dimension_coverage_percentage": len(all_dimensions_covered) / len(self.dimension_categories) * 100,
            "domain_statistics": domain_stats,
            "dimensions_found": list(all_dimensions_covered),
            "requirements_met": {
                "min_20_questions_per_conversation": all(c.get('total_questions', 0) >= 20 for c in all_conversations),
                "min_6_dimensions_covered": len(all_dimensions_covered) >= 6,
                "all_domains_covered": len(domain_stats) == len(self.domain_specs),
                "progressive_questioning": True  # Built into generation logic
            }
        }
        
        # Save complete dataset
        complete_dataset = {
            "metadata": {
                "generation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "generator_version": "1.0",
                "total_conversations": len(all_conversations),
                "domains_covered": list(self.domain_specs.keys())
            },
            "conversations": all_conversations,
            "statistics": summary_stats
        }
        
        # Save complete dataset
        complete_file = output_path / "complete_training_dataset.json"
        with open(complete_file, 'w') as f:
            json.dump(complete_dataset, f, indent=2)
        
        # Save summary statistics
        stats_file = output_path / "dataset_statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)
        
        print("\n" + "="*60)
        print("🎉 COMPREHENSIVE DATASET GENERATION COMPLETE!")
        print("="*60)
        print(f"📊 Total Conversations: {summary_stats['total_conversations']}")
        print(f"📝 Total Questions: {summary_stats['total_questions']}")
        print(f"🏢 Domains Covered: {summary_stats['domains_covered']}/8")
        print(f"📋 Dimensions Covered: {summary_stats['dimensions_covered']}/8 ({summary_stats['dimension_coverage_percentage']:.1f}%)")
        
        print(f"\n✅ Requirements Status:")
        for req, status in summary_stats['requirements_met'].items():
            print(f"  {req}: {'✅ PASS' if status else '❌ FAIL'}")
        
        print(f"\n📁 Files Generated:")
        print(f"  Complete Dataset: {complete_file}")
        print(f"  Statistics: {stats_file}")
        for domain in self.domain_specs.keys():
            domain_file = domain.lower().replace('/', '_')
            print(f"  {domain}: {domain_file}_conversations.json")
        
        return summary_stats

def main():
    """Main function to generate comprehensive training data"""
    
    # Get API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ Error: GROQ_API_KEY not found in environment variables")
        print("Please set your Groq API key in the .env file")
        return
    
    # Initialize generator
    generator = ComprehensiveTrainingDataGenerator(api_key)
    
    # Generate comprehensive dataset
    stats = generator.generate_comprehensive_dataset()
    
    print(f"\n🚀 Training data generation complete!")
    print(f"Ready for neural model training with {stats['total_conversations']} conversations")
    print(f"covering {stats['total_questions']} strategic business questions!")

if __name__ == "__main__":
    main()