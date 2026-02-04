"""
Neural Training Data Generator using Groq API
Alternative to Anthropic API for generating training data for the 6-model neural system
"""

import os
import json
import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from pathlib import Path
import uuid
from datetime import datetime

# Import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: Groq not available. Install with: pip install groq")

load_dotenv()

@dataclass
class PatternEncodingExample:
    """Training example for pattern encoder (contrastive learning)"""
    anchor_problem: str
    positive_problem: str  # Similar problem
    negative_problem: str  # Dissimilar problem
    similarity_score: float

@dataclass
class DomainClassificationExample:
    """Training example for domain classifier"""
    problem_text: str
    domain_labels: List[str]  # Multi-label
    domain_scores: List[float]

@dataclass
class IntentExtractionExample:
    """Training example for intent extractor"""
    problem_text: str
    primary_intent: str
    urgency_level: str
    scope_level: str
    confidence_scores: Dict[str, float]

@dataclass
class QuestionGenerationExample:
    """Training example for question generator"""
    problem_context: str
    domain: str
    intent: str
    generated_questions: List[Dict[str, Any]]
    quality_score: float

@dataclass
class QuestionRankingExample:
    """Training example for question ranker"""
    problem_context: str
    questions: List[str]
    information_gain_scores: List[float]
    optimal_ranking: List[int]

@dataclass
class ClarificationExample:
    """Training example for clarification trigger"""
    question_text: str
    answer_text: str
    vagueness_score: float
    completeness_score: float
    confidence_score: float
    needs_clarification: bool
    clarification_questions: List[str]

class GroqNeuralDataGenerator:
    """
    Generates training data for neural business understanding system using Groq API
    """
    
    def __init__(self, api_key: str = None):
        if not GROQ_AVAILABLE:
            raise ImportError("Groq package not available. Install with: pip install groq")
        
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        
        # Business domains for neural system
        self.domains = [
            {
                "name": "customer_retention",
                "description": "Customer churn reduction, retention strategies, loyalty programs",
                "keywords": ["churn", "retention", "loyalty", "attrition", "customer lifetime value"],
                "sample_problems": [
                    "How can we reduce customer churn in our SaaS product?",
                    "Why are enterprise customers leaving after 6 months?",
                    "What retention strategies work best for different customer segments?"
                ]
            },
            {
                "name": "sales_forecasting", 
                "description": "Revenue prediction, sales pipeline analysis, demand forecasting",
                "keywords": ["forecast", "prediction", "revenue", "sales", "pipeline"],
                "sample_problems": [
                    "How can we improve our quarterly sales forecasting accuracy?",
                    "What factors most influence our monthly revenue predictions?",
                    "How do we forecast sales for a new product launch?"
                ]
            },
            {
                "name": "pricing_optimization",
                "description": "Price elasticity, revenue optimization, competitive pricing", 
                "keywords": ["pricing", "elasticity", "optimization", "revenue", "margin"],
                "sample_problems": [
                    "What's the optimal price point for our new product?",
                    "How price-sensitive are our customers?",
                    "Should we implement dynamic pricing?"
                ]
            },
            {
                "name": "customer_segmentation",
                "description": "Customer clustering, persona development, targeted marketing",
                "keywords": ["segmentation", "clustering", "personas", "targeting"],
                "sample_problems": [
                    "How should we segment our customer base?",
                    "What are the key characteristics of our most valuable customers?",
                    "How do we create actionable customer personas?"
                ]
            },
            {
                "name": "demand_forecasting",
                "description": "Inventory planning, supply chain optimization, stockout prevention",
                "keywords": ["demand", "inventory", "supply chain", "stockout"],
                "sample_problems": [
                    "How much inventory should we stock for Q4?",
                    "What drives demand variability in our products?",
                    "How do we prevent stockouts without overordering?"
                ]
            },
            {
                "name": "marketing_attribution",
                "description": "Channel attribution, ROI measurement, budget allocation",
                "keywords": ["attribution", "roi", "marketing", "channels"],
                "sample_problems": [
                    "Which marketing channels drive the most conversions?",
                    "How should we allocate our marketing budget?",
                    "What's the true ROI of our paid search campaigns?"
                ]
            },
            {
                "name": "process_optimization",
                "description": "Workflow improvement, bottleneck analysis, efficiency gains",
                "keywords": ["process", "optimization", "efficiency", "bottleneck"],
                "sample_problems": [
                    "How can we improve our order fulfillment process?",
                    "Where are the bottlenecks in our production line?",
                    "What processes should we automate first?"
                ]
            },
            {
                "name": "fraud_detection",
                "description": "Transaction monitoring, anomaly detection, risk assessment",
                "keywords": ["fraud", "anomaly", "detection", "risk"],
                "sample_problems": [
                    "How do we detect fraudulent transactions in real-time?",
                    "What patterns indicate potential fraud?",
                    "How do we reduce false positive rates?"
                ]
            }
        ]
        
        # Intent categories
        self.intents = [
            "understand_problem", "optimize_process", "predict_outcome",
            "reduce_cost", "increase_revenue", "improve_quality",
            "manage_risk", "enhance_experience", "automate_task", "strategic_planning"
        ]
        
        self.urgency_levels = ["low", "medium", "high", "critical"]
        self.scope_levels = ["individual", "team", "department", "company", "industry"]
    
    def _call_groq(self, prompt: str, max_tokens: int = 4000) -> str:
        """Make API call to Groq"""
        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # Updated to current available model
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Groq API error: {e}")
            return ""
    
    def _extract_json(self, text: str) -> Any:
        """Extract JSON from text response"""
        text = text.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return []
    
    def generate_pattern_encoding_examples(self, num_examples: int = 2000) -> List[PatternEncodingExample]:
        """Generate examples for Model 1: Pattern Encoder"""
        
        print(f"Generating {num_examples} pattern encoding examples using Groq...")
        examples = []
        
        batch_size = 20  # Smaller batches for Groq
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_size_actual = batch_end - batch_start
            
            prompt = f"""Generate {batch_size_actual} sets of business problems for contrastive learning.

For each set, provide:
1. anchor_problem: A business problem
2. positive_problem: A similar business problem (same domain/intent)
3. negative_problem: A dissimilar business problem (different domain/intent)  
4. similarity_score: How similar anchor and positive are (0.7-0.95)

Make problems diverse across domains: {[d['name'] for d in self.domains]}

Output as JSON array of objects with: anchor_problem, positive_problem, negative_problem, similarity_score

Example:
[
  {{
    "anchor_problem": "How can we reduce customer churn?",
    "positive_problem": "What strategies improve customer retention?",
    "negative_problem": "How do we optimize our supply chain costs?",
    "similarity_score": 0.85
  }}
]"""

            response = self._call_groq(prompt)
            if response:
                try:
                    batch_data = self._extract_json(response)
                    for item in batch_data:
                        if all(key in item for key in ['anchor_problem', 'positive_problem', 'negative_problem', 'similarity_score']):
                            examples.append(PatternEncodingExample(
                                anchor_problem=item['anchor_problem'],
                                positive_problem=item['positive_problem'],
                                negative_problem=item['negative_problem'],
                                similarity_score=item['similarity_score']
                            ))
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
            
            print(f"Generated {len(examples)} pattern encoding examples so far...")
            time.sleep(1)  # Rate limiting
        
        print(f"Generated {len(examples)} pattern encoding examples total")
        return examples
    
    def generate_domain_classification_examples(self, num_examples: int = 1500) -> List[DomainClassificationExample]:
        """Generate examples for Model 2: Domain Classifier"""
        
        print(f"Generating {num_examples} domain classification examples...")
        examples = []
        
        batch_size = 15
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_size_actual = batch_end - batch_start
            
            prompt = f"""Generate {batch_size_actual} business problems with multi-label domain classification.

Domains: {[d['name'] for d in self.domains]}

For each problem, provide:
1. problem_text: The business problem
2. domain_labels: List of relevant domains (can be multiple)
3. domain_scores: List of confidence scores (0.0-1.0) for each domain in domain_labels

Some problems should span multiple domains.

Output as JSON array:
[
  {{
    "problem_text": "How can we reduce customer churn while optimizing pricing?",
    "domain_labels": ["customer_retention", "pricing_optimization"],
    "domain_scores": [0.8, 0.7]
  }}
]"""

            response = self._call_groq(prompt)
            if response:
                try:
                    batch_data = self._extract_json(response)
                    for item in batch_data:
                        if all(key in item for key in ['problem_text', 'domain_labels', 'domain_scores']):
                            examples.append(DomainClassificationExample(
                                problem_text=item['problem_text'],
                                domain_labels=item['domain_labels'],
                                domain_scores=item['domain_scores']
                            ))
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
            
            print(f"Generated {len(examples)} domain classification examples so far...")
            time.sleep(1)
        
        print(f"Generated {len(examples)} domain classification examples total")
        return examples
    
    def generate_intent_extraction_examples(self, num_examples: int = 1500) -> List[IntentExtractionExample]:
        """Generate examples for Model 3: Intent Extractor"""
        
        print(f"Generating {num_examples} intent extraction examples...")
        examples = []
        
        batch_size = 15
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_size_actual = batch_end - batch_start
            
            prompt = f"""Generate {batch_size_actual} business problems with intent, urgency, and scope labels.

Intents: {self.intents}
Urgency levels: {self.urgency_levels}
Scope levels: {self.scope_levels}

For each problem, provide:
1. problem_text: The business problem
2. primary_intent: Main intent from the list
3. urgency_level: How urgent the problem is
4. scope_level: Organizational scope affected
5. confidence_scores: Dict with intent_confidence, urgency_confidence, scope_confidence (0.0-1.0)

Output as JSON array:
[
  {{
    "problem_text": "We need to reduce costs urgently across the company",
    "primary_intent": "reduce_cost",
    "urgency_level": "high",
    "scope_level": "company",
    "confidence_scores": {{"intent_confidence": 0.9, "urgency_confidence": 0.8, "scope_confidence": 0.85}}
  }}
]"""

            response = self._call_groq(prompt)
            if response:
                try:
                    batch_data = self._extract_json(response)
                    for item in batch_data:
                        if all(key in item for key in ['problem_text', 'primary_intent', 'urgency_level', 'scope_level', 'confidence_scores']):
                            examples.append(IntentExtractionExample(
                                problem_text=item['problem_text'],
                                primary_intent=item['primary_intent'],
                                urgency_level=item['urgency_level'],
                                scope_level=item['scope_level'],
                                confidence_scores=item['confidence_scores']
                            ))
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
            
            print(f"Generated {len(examples)} intent extraction examples so far...")
            time.sleep(1)
        
        print(f"Generated {len(examples)} intent extraction examples total")
        return examples
    
    def generate_question_generation_examples(self, num_examples: int = 2000) -> List[QuestionGenerationExample]:
        """Generate examples for Model 4: Question Generator"""
        
        print(f"Generating {num_examples} question generation examples...")
        examples = []
        
        batch_size = 10
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_size_actual = batch_end - batch_start
            
            prompt = f"""Generate {batch_size_actual} examples of business problems with strategic questions.

For each example, provide:
1. problem_context: A business problem description
2. domain: One of {[d['name'] for d in self.domains[:4]]}
3. intent: One of {self.intents[:5]}
4. generated_questions: List of 5-6 strategic questions with:
   - question_text: The question
   - category: problem_definition, business_objectives, stakeholders, current_situation, constraints, or success_criteria
   - priority: 0.0-1.0
   - expected_answer_type: text, numeric, boolean, or multiple_choice
   - reasoning: Why this question is important
5. quality_score: Overall quality of questions (0.7-0.95)

Output as JSON array."""

            response = self._call_groq(prompt)
            if response:
                try:
                    batch_data = self._extract_json(response)
                    for item in batch_data:
                        if all(key in item for key in ['problem_context', 'domain', 'intent', 'generated_questions', 'quality_score']):
                            examples.append(QuestionGenerationExample(
                                problem_context=item['problem_context'],
                                domain=item['domain'],
                                intent=item['intent'],
                                generated_questions=item['generated_questions'],
                                quality_score=item['quality_score']
                            ))
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
            
            print(f"Generated {len(examples)} question generation examples so far...")
            time.sleep(1)
        
        print(f"Generated {len(examples)} question generation examples total")
        return examples
    
    def generate_question_ranking_examples(self, num_examples: int = 1000) -> List[QuestionRankingExample]:
        """Generate examples for Model 5: Question Ranker"""
        
        print(f"Generating {num_examples} question ranking examples...")
        examples = []
        
        batch_size = 10
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_size_actual = batch_end - batch_start
            
            prompt = f"""Generate {batch_size_actual} examples of question ranking for business problems.

For each example, provide:
1. problem_context: A business problem
2. questions: List of 6-8 potential questions to ask
3. information_gain_scores: Score for each question (0.0-1.0) based on insight value
4. optimal_ranking: Indices of questions in order of information gain (highest first)

Higher information gain = more valuable insights, strategic importance.

Output as JSON array."""

            response = self._call_groq(prompt)
            if response:
                try:
                    batch_data = self._extract_json(response)
                    for item in batch_data:
                        if all(key in item for key in ['problem_context', 'questions', 'information_gain_scores', 'optimal_ranking']):
                            examples.append(QuestionRankingExample(
                                problem_context=item['problem_context'],
                                questions=item['questions'],
                                information_gain_scores=item['information_gain_scores'],
                                optimal_ranking=item['optimal_ranking']
                            ))
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
            
            print(f"Generated {len(examples)} question ranking examples so far...")
            time.sleep(1)
        
        print(f"Generated {len(examples)} question ranking examples total")
        return examples
    
    def generate_clarification_examples(self, num_examples: int = 2000) -> List[ClarificationExample]:
        """Generate examples for Model 6: Clarification Trigger"""
        
        print(f"Generating {num_examples} clarification examples...")
        examples = []
        
        batch_size = 15
        for batch_start in range(0, num_examples, batch_size):
            batch_end = min(batch_start + batch_size, num_examples)
            batch_size_actual = batch_end - batch_start
            
            prompt = f"""Generate {batch_size_actual} examples of question-answer pairs with clarification analysis.

For each example, provide:
1. question_text: A business question
2. answer_text: A user's answer (vary from clear to vague)
3. vagueness_score: 0.0-1.0 (higher = more vague)
4. completeness_score: 0.0-1.0 (higher = more complete)
5. confidence_score: 0.0-1.0 (higher = more confident)
6. needs_clarification: boolean (true if clarification needed)
7. clarification_questions: List of 0-3 follow-up questions if needed

Include mix of clear, vague, incomplete, and uncertain answers.

Output as JSON array."""

            response = self._call_groq(prompt)
            if response:
                try:
                    batch_data = self._extract_json(response)
                    for item in batch_data:
                        if all(key in item for key in ['question_text', 'answer_text', 'vagueness_score', 'completeness_score', 'confidence_score', 'needs_clarification', 'clarification_questions']):
                            examples.append(ClarificationExample(
                                question_text=item['question_text'],
                                answer_text=item['answer_text'],
                                vagueness_score=item['vagueness_score'],
                                completeness_score=item['completeness_score'],
                                confidence_score=item['confidence_score'],
                                needs_clarification=item['needs_clarification'],
                                clarification_questions=item['clarification_questions']
                            ))
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    continue
            
            print(f"Generated {len(examples)} clarification examples so far...")
            time.sleep(1)
        
        print(f"Generated {len(examples)} clarification examples total")
        return examples
    
    def generate_complete_neural_dataset(self) -> Dict[str, List]:
        """Generate complete dataset for all 6 neural models"""
        
        print("="*80)
        print("GENERATING NEURAL TRAINING DATASET WITH GROQ")
        print("="*80)
        print("Target: 10k+ examples across 6 models")
        print("Estimated cost: $20-50 in API calls")
        print("Estimated time: 2-4 hours")
        
        start_time = time.time()
        
        # Generate examples for each model (smaller scale for Groq)
        dataset = {}
        
        # Model 1: Pattern Encoder
        dataset['pattern_encoding'] = self.generate_pattern_encoding_examples(2000)
        
        # Model 2: Domain Classifier
        dataset['domain_classification'] = self.generate_domain_classification_examples(1500)
        
        # Model 3: Intent Extractor
        dataset['intent_extraction'] = self.generate_intent_extraction_examples(1500)
        
        # Model 4: Question Generator
        dataset['question_generation'] = self.generate_question_generation_examples(2000)
        
        # Model 5: Question Ranker
        dataset['question_ranking'] = self.generate_question_ranking_examples(1000)
        
        # Model 6: Clarification Trigger
        dataset['clarification'] = self.generate_clarification_examples(2000)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate statistics
        total_examples = sum(len(examples) for examples in dataset.values())
        
        print("\n" + "="*80)
        print("DATASET GENERATION COMPLETE!")
        print("="*80)
        print(f"Total time: {duration/3600:.1f} hours ({duration/60:.1f} minutes)")
        print(f"Total examples: {total_examples:,}")
        print("\nBreakdown by model:")
        for model_name, examples in dataset.items():
            print(f"  {model_name}: {len(examples):,} examples")
        
        return dataset
    
    def save_neural_dataset(self, dataset: Dict[str, List], output_dir: str = "./groq_neural_training_data"):
        """Save dataset for neural model training"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\nSaving neural dataset to {output_path}...")
        
        # Save each model's data separately
        for model_name, examples in dataset.items():
            # Convert to dict format
            data = [asdict(example) for example in examples]
            
            # Save as JSON
            json_path = output_path / f"{model_name}_training_data.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            print(f"  Saved {len(examples):,} examples for {model_name}")
        
        # Save metadata
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_examples': sum(len(examples) for examples in dataset.values()),
            'api_used': 'groq',
            'models': {
                model_name: {
                    'num_examples': len(examples),
                    'example_type': type(examples[0]).__name__ if examples else 'Unknown'
                }
                for model_name, examples in dataset.items()
            },
            'domains': [d['name'] for d in self.domains],
            'intents': self.intents
        }
        
        metadata_path = output_path / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\nDataset saved successfully!")
        print(f"Total size: {sum(len(examples) for examples in dataset.values()):,} examples")
        print(f"Location: {output_path}")
        
        return str(output_path)

# Usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate neural training data using Groq API")
    parser.add_argument("--output-dir", default="./groq_neural_training_data", help="Output directory")
    parser.add_argument("--api-key", help="Groq API key (or set GROQ_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Check API key
    api_key = args.api_key or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY required for data generation")
        print("Set environment variable or use --api-key argument")
        exit(1)
    
    # Initialize generator
    generator = GroqNeuralDataGenerator(api_key=api_key)
    
    # Generate complete dataset
    dataset = generator.generate_complete_neural_dataset()
    
    # Save dataset
    output_path = generator.save_neural_dataset(dataset, args.output_dir)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print(f"""
1. Your neural training dataset is ready at: {output_path}

2. Train the neural models:
   python neural_training_pipeline.py --data-dir {output_path}

3. The neural system provides:
   ✅ Pattern recognition across 10k+ business problems
   ✅ Multi-domain classification with confidence scores
   ✅ Intent extraction with urgency and scope
   ✅ Contextual question generation
   ✅ Intelligent question ranking by information gain
   ✅ Automatic clarification detection

4. Expected training time: 8-16 hours on A100 GPU
5. Expected training cost: $30-80 (cloud GPU rental)
6. Expected quality: 90%+ accuracy on business understanding tasks

Cost Analysis:
- Data generation: $20-50 (one-time, using Groq)
- Model training: $30-80 (one-time)  
- Runtime inference: $0 (local)
- Break-even vs API: ~300-500 sessions

This neural approach provides high-quality business understanding
with pattern recognition and contextual intelligence.
""")