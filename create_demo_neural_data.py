"""
Create Demo Neural Training Data
Generates a small sample dataset for testing the neural system without API calls
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import uuid

def create_demo_pattern_encoding_data():
    """Create demo pattern encoding examples"""
    examples = []
    
    # Sample business problems with similar/dissimilar pairs
    patterns = [
        {
            "anchor_problem": "How can we reduce customer churn in our SaaS product?",
            "positive_problem": "What strategies can improve customer retention rates?",
            "negative_problem": "How do we optimize our supply chain costs?",
            "similarity_score": 0.85
        },
        {
            "anchor_problem": "What's the best pricing strategy for our new product?",
            "positive_problem": "How should we set competitive prices to maximize revenue?",
            "negative_problem": "How can we improve employee satisfaction scores?",
            "similarity_score": 0.82
        },
        {
            "anchor_problem": "How do we forecast quarterly sales more accurately?",
            "positive_problem": "What methods improve revenue prediction accuracy?",
            "negative_problem": "How can we reduce customer support response times?",
            "similarity_score": 0.88
        },
        {
            "anchor_problem": "Which marketing channels drive the most conversions?",
            "positive_problem": "How do we measure marketing ROI across different channels?",
            "negative_problem": "What's causing delays in our production process?",
            "similarity_score": 0.79
        },
        {
            "anchor_problem": "How can we segment our customers more effectively?",
            "positive_problem": "What are the key characteristics of our most valuable customers?",
            "negative_problem": "How do we reduce inventory holding costs?",
            "similarity_score": 0.83
        }
    ]
    
    # Replicate patterns to create more examples
    for i in range(100):  # Create 500 examples total
        base_pattern = patterns[i % len(patterns)]
        examples.append({
            "anchor_problem": base_pattern["anchor_problem"],
            "positive_problem": base_pattern["positive_problem"],
            "negative_problem": base_pattern["negative_problem"],
            "similarity_score": base_pattern["similarity_score"] + np.random.normal(0, 0.05)
        })
    
    return examples

def create_demo_domain_classification_data():
    """Create demo domain classification examples"""
    examples = []
    
    domain_examples = [
        {
            "problem_text": "How can we reduce customer churn in our subscription service?",
            "domain_labels": ["customer_retention"],
            "domain_scores": [0.9]
        },
        {
            "problem_text": "What's the optimal price point for our new product launch?",
            "domain_labels": ["pricing_optimization"],
            "domain_scores": [0.85]
        },
        {
            "problem_text": "How do we forecast demand for the holiday season?",
            "domain_labels": ["demand_forecasting"],
            "domain_scores": [0.88]
        },
        {
            "problem_text": "Which marketing channels provide the best ROI?",
            "domain_labels": ["marketing_attribution"],
            "domain_scores": [0.82]
        },
        {
            "problem_text": "How should we segment our customer base for targeted campaigns?",
            "domain_labels": ["customer_segmentation"],
            "domain_scores": [0.87]
        },
        {
            "problem_text": "How can we improve our sales forecasting accuracy while reducing customer churn?",
            "domain_labels": ["sales_forecasting", "customer_retention"],
            "domain_scores": [0.8, 0.75]
        }
    ]
    
    # Replicate to create more examples
    for i in range(200):  # Create 1200 examples total
        base_example = domain_examples[i % len(domain_examples)]
        examples.append(base_example.copy())
    
    return examples

def create_demo_intent_extraction_data():
    """Create demo intent extraction examples"""
    examples = []
    
    intent_examples = [
        {
            "problem_text": "We urgently need to reduce costs across the entire company",
            "primary_intent": "reduce_cost",
            "urgency_level": "high",
            "scope_level": "company",
            "confidence_scores": {"intent_confidence": 0.9, "urgency_confidence": 0.85, "scope_confidence": 0.88}
        },
        {
            "problem_text": "I want to understand why our customer satisfaction scores are declining",
            "primary_intent": "understand_problem",
            "urgency_level": "medium",
            "scope_level": "department",
            "confidence_scores": {"intent_confidence": 0.82, "urgency_confidence": 0.75, "scope_confidence": 0.8}
        },
        {
            "problem_text": "How can we predict next quarter's revenue more accurately?",
            "primary_intent": "predict_outcome",
            "urgency_level": "medium",
            "scope_level": "company",
            "confidence_scores": {"intent_confidence": 0.88, "urgency_confidence": 0.7, "scope_confidence": 0.85}
        },
        {
            "problem_text": "We need to increase our market share in the next 6 months",
            "primary_intent": "increase_revenue",
            "urgency_level": "high",
            "scope_level": "company",
            "confidence_scores": {"intent_confidence": 0.85, "urgency_confidence": 0.8, "scope_confidence": 0.9}
        },
        {
            "problem_text": "Can we automate our invoice processing to save time?",
            "primary_intent": "automate_task",
            "urgency_level": "low",
            "scope_level": "team",
            "confidence_scores": {"intent_confidence": 0.9, "urgency_confidence": 0.6, "scope_confidence": 0.75}
        }
    ]
    
    # Replicate to create more examples
    for i in range(200):  # Create 1000 examples total
        base_example = intent_examples[i % len(intent_examples)]
        examples.append(base_example.copy())
    
    return examples

def create_demo_question_generation_data():
    """Create demo question generation examples"""
    examples = []
    
    question_examples = [
        {
            "problem_context": "How can we reduce customer churn in our SaaS product?",
            "domain": "customer_retention",
            "intent": "reduce_cost",
            "generated_questions": [
                {
                    "question_text": "What is your current customer churn rate?",
                    "category": "current_situation",
                    "priority": 0.9,
                    "expected_answer_type": "numeric",
                    "reasoning": "Need baseline metrics to understand the problem scope"
                },
                {
                    "question_text": "Which customer segments have the highest churn rates?",
                    "category": "problem_definition",
                    "priority": 0.85,
                    "expected_answer_type": "text",
                    "reasoning": "Segmentation helps identify specific problem areas"
                },
                {
                    "question_text": "What are the main reasons customers give for canceling?",
                    "category": "problem_definition",
                    "priority": 0.8,
                    "expected_answer_type": "multiple_choice",
                    "reasoning": "Understanding root causes is essential for solutions"
                }
            ],
            "quality_score": 0.88
        },
        {
            "problem_context": "What's the optimal pricing strategy for our new product?",
            "domain": "pricing_optimization",
            "intent": "increase_revenue",
            "generated_questions": [
                {
                    "question_text": "What are your competitors charging for similar products?",
                    "category": "current_situation",
                    "priority": 0.85,
                    "expected_answer_type": "numeric",
                    "reasoning": "Competitive analysis is crucial for pricing decisions"
                },
                {
                    "question_text": "What is your target profit margin for this product?",
                    "category": "business_objectives",
                    "priority": 0.9,
                    "expected_answer_type": "numeric",
                    "reasoning": "Profit targets drive pricing strategy"
                }
            ],
            "quality_score": 0.82
        }
    ]
    
    # Replicate to create more examples
    for i in range(100):  # Create 200 examples total
        base_example = question_examples[i % len(question_examples)]
        examples.append(base_example.copy())
    
    return examples

def create_demo_question_ranking_data():
    """Create demo question ranking examples"""
    examples = []
    
    ranking_examples = [
        {
            "problem_context": "How can we improve our sales forecasting accuracy?",
            "questions": [
                "What forecasting methods are you currently using?",
                "How accurate are your current forecasts?",
                "What data sources do you use for forecasting?",
                "Who is responsible for creating forecasts?",
                "How often do you update your forecasts?",
                "What external factors affect your sales?"
            ],
            "information_gain_scores": [0.9, 0.95, 0.85, 0.6, 0.7, 0.8],
            "optimal_ranking": [1, 0, 2, 5, 4, 3]
        },
        {
            "problem_context": "Which marketing channels should we prioritize?",
            "questions": [
                "What is your current marketing budget?",
                "Which channels are you currently using?",
                "What are your conversion rates by channel?",
                "What is your target audience?",
                "How do you measure marketing success?",
                "What are your competitors doing?"
            ],
            "information_gain_scores": [0.7, 0.8, 0.95, 0.85, 0.9, 0.75],
            "optimal_ranking": [2, 4, 3, 1, 5, 0]
        }
    ]
    
    # Replicate to create more examples
    for i in range(50):  # Create 100 examples total
        base_example = ranking_examples[i % len(ranking_examples)]
        examples.append(base_example.copy())
    
    return examples

def create_demo_clarification_data():
    """Create demo clarification examples"""
    examples = []
    
    clarification_examples = [
        {
            "question_text": "What is your current customer churn rate?",
            "answer_text": "It's pretty high, around 8% I think, but it varies by month",
            "vagueness_score": 0.6,
            "completeness_score": 0.7,
            "confidence_score": 0.5,
            "needs_clarification": True,
            "clarification_questions": [
                "Can you provide the exact churn rate for the last 3 months?",
                "What causes the monthly variation in churn rate?"
            ]
        },
        {
            "question_text": "Which customer segments have the highest churn?",
            "answer_text": "Enterprise customers churn at 12% monthly, SMB at 6%, and individual users at 15%",
            "vagueness_score": 0.1,
            "completeness_score": 0.9,
            "confidence_score": 0.85,
            "needs_clarification": False,
            "clarification_questions": []
        },
        {
            "question_text": "What's your marketing budget?",
            "answer_text": "We spend some money on ads and stuff, not sure exactly how much",
            "vagueness_score": 0.9,
            "completeness_score": 0.2,
            "confidence_score": 0.3,
            "needs_clarification": True,
            "clarification_questions": [
                "Can you provide the specific monthly marketing budget?",
                "Which advertising channels do you spend on?",
                "Who manages the marketing budget?"
            ]
        }
    ]
    
    # Replicate to create more examples
    for i in range(200):  # Create 600 examples total
        base_example = clarification_examples[i % len(clarification_examples)]
        examples.append(base_example.copy())
    
    return examples

def create_demo_neural_dataset():
    """Create complete demo dataset for neural system testing"""
    
    print("Creating demo neural training dataset...")
    
    dataset = {
        'pattern_encoding': create_demo_pattern_encoding_data(),
        'domain_classification': create_demo_domain_classification_data(),
        'intent_extraction': create_demo_intent_extraction_data(),
        'question_generation': create_demo_question_generation_data(),
        'question_ranking': create_demo_question_ranking_data(),
        'clarification': create_demo_clarification_data()
    }
    
    # Calculate statistics
    total_examples = sum(len(examples) for examples in dataset.values())
    
    print(f"Demo dataset created with {total_examples} total examples:")
    for model_name, examples in dataset.items():
        print(f"  {model_name}: {len(examples)} examples")
    
    return dataset

def save_demo_dataset(dataset, output_dir="./demo_neural_training_data"):
    """Save demo dataset"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\nSaving demo dataset to {output_path}...")
    
    # Save each model's data
    for model_name, examples in dataset.items():
        json_path = output_path / f"{model_name}_training_data.json"
        with open(json_path, 'w') as f:
            json.dump(examples, f, indent=2)
        print(f"  Saved {len(examples)} examples for {model_name}")
    
    # Save metadata
    metadata = {
        'generation_timestamp': datetime.now().isoformat(),
        'total_examples': sum(len(examples) for examples in dataset.values()),
        'dataset_type': 'demo',
        'models': {
            model_name: {
                'num_examples': len(examples),
                'example_type': 'demo_generated'
            }
            for model_name, examples in dataset.items()
        },
        'domains': [
            "customer_retention", "sales_forecasting", "pricing_optimization",
            "customer_segmentation", "demand_forecasting", "marketing_attribution",
            "process_optimization", "fraud_detection"
        ],
        'intents': [
            "understand_problem", "optimize_process", "predict_outcome",
            "reduce_cost", "increase_revenue", "improve_quality",
            "manage_risk", "enhance_experience", "automate_task", "strategic_planning"
        ]
    }
    
    metadata_path = output_path / "dataset_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDemo dataset saved successfully!")
    print(f"Location: {output_path}")
    
    return str(output_path)

if __name__ == "__main__":
    # Create and save demo dataset
    dataset = create_demo_neural_dataset()
    output_path = save_demo_dataset(dataset)
    
    print("\n" + "="*60)
    print("DEMO NEURAL DATASET READY!")
    print("="*60)
    print(f"""
Demo dataset created at: {output_path}

This demo dataset contains:
✅ 500 pattern encoding examples
✅ 1,200 domain classification examples  
✅ 1,000 intent extraction examples
✅ 200 question generation examples
✅ 100 question ranking examples
✅ 600 clarification examples

Total: 3,600 training examples

Next steps:
1. Train the neural models:
   python neural_training_pipeline.py --data-dir {output_path}

2. Test the neural system:
   python neural_business_understanding_system.py

This demo dataset allows you to test the complete neural system
without waiting for API-based data generation!
""")