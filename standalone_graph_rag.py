"""
Standalone Graph RAG System
Works without Neo4j using in-memory graph
Perfect for testing and development
"""

import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from unified_llm_wrapper import UnifiedLLM
import logging

# Make sentence_transformers optional
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("sentence_transformers not available, embeddings disabled")

import numpy as np

logger = logging.getLogger(__name__)
load_dotenv()


class InMemoryKnowledgeGraph:
    """In-memory knowledge graph (no Neo4j required)"""
    
    def __init__(self):
        self.problems = []
        self.questions = []
        self.concepts = {}
        self.domains = {}
        self.relationships = []
        
        # Initialize with basic knowledge
        self._initialize_basic_knowledge()
    
    def _initialize_basic_knowledge(self):
        """Add basic business knowledge"""
        
        # Common concepts
        self.concepts = {
            'churn': {'category': 'customer', 'related': ['retention', 'satisfaction']},
            'retention': {'category': 'customer', 'related': ['churn', 'engagement']},
            'revenue': {'category': 'financial', 'related': ['mrr', 'growth']},
            'cost': {'category': 'financial', 'related': ['cac', 'efficiency']},
            'conversion': {'category': 'marketing', 'related': ['funnel', 'optimization']},
        }
        
        # Common domains
        self.domains = {
            'customer_retention': ['churn', 'retention', 'satisfaction'],
            'revenue_growth': ['revenue', 'mrr', 'growth'],
            'cost_optimization': ['cost', 'cac', 'efficiency'],
            'conversion_optimization': ['conversion', 'funnel', 'optimization']
        }
        
        # Sample problems and questions
        self.problems = [
            {
                'id': 'prob_001',
                'text': 'How can we reduce customer churn?',
                'domains': ['customer_retention'],
                'concepts': ['churn', 'retention'],
                'questions': ['q_001', 'q_002', 'q_003']
            },
            {
                'id': 'prob_002',
                'text': 'How can we increase revenue?',
                'domains': ['revenue_growth'],
                'concepts': ['revenue', 'growth'],
                'questions': ['q_004', 'q_005']
            }
        ]
        
        self.questions = [
            {
                'id': 'q_001',
                'text': 'What is your current churn rate?',
                'category': 'current_situation',
                'priority': 'critical',
                'concepts': ['churn']
            },
            {
                'id': 'q_002',
                'text': 'Which customer segment has highest churn?',
                'category': 'current_situation',
                'priority': 'high',
                'concepts': ['churn', 'segmentation']
            },
            {
                'id': 'q_003',
                'text': 'What is your target churn rate?',
                'category': 'goals',
                'priority': 'critical',
                'concepts': ['churn', 'target']
            },
            {
                'id': 'q_004',
                'text': 'What is your current monthly revenue?',
                'category': 'current_situation',
                'priority': 'critical',
                'concepts': ['revenue']
            },
            {
                'id': 'q_005',
                'text': 'What is your revenue growth target?',
                'category': 'goals',
                'priority': 'high',
                'concepts': ['revenue', 'growth']
            }
        ]
    
    def add_problem(self, problem_id: str, text: str, domains: List[str], concepts: List[str]):
        """Add a problem to the graph"""
        self.problems.append({
            'id': problem_id,
            'text': text,
            'domains': domains,
            'concepts': concepts,
            'questions': []
        })
    
    def find_similar_problems(self, concepts: List[str], domains: List[str]) -> List[Dict]:
        """Find similar problems based on concepts and domains"""
        similar = []
        
        for prob in self.problems:
            # Calculate similarity
            concept_overlap = len(set(concepts) & set(prob['concepts']))
            domain_overlap = len(set(domains) & set(prob['domains']))
            
            if concept_overlap > 0 or domain_overlap > 0:
                similarity = (concept_overlap * 0.6 + domain_overlap * 0.4) / max(len(concepts), 1)
                similar.append({
                    'problem_text': prob['text'],
                    'similarity': min(similarity, 1.0),
                    'questions': [q for q in self.questions if q['id'] in prob.get('questions', [])]
                })
        
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)
    
    def get_questions_for_concepts(self, concepts: List[str]) -> List[Dict]:
        """Get questions related to concepts"""
        relevant = []
        
        for q in self.questions:
            q_concepts = q.get('concepts', [])
            if any(c in q_concepts for c in concepts):
                relevant.append(q)
        
        return relevant


class StandaloneGraphRAG:
    """
    Standalone Graph RAG system
    No Neo4j required - uses in-memory graph
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize standalone system"""
        self.kg = InMemoryKnowledgeGraph()
        self.llm = UnifiedLLM(provider='groq', api_key=api_key or os.getenv("GROQ_API_KEY"))
        
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Embedder initialized")
            except:
                self.embedder = None
                logger.warning("⚠️ Embedder initialization failed")
        else:
            self.embedder = None
            logger.info("ℹ️ Embeddings not available (optional)")
    
    def extract_concepts_and_domains(self, text: str) -> Dict:
        """Extract concepts and domains from text"""
        text_lower = text.lower()
        
        # Detect concepts
        detected_concepts = []
        for concept in self.kg.concepts.keys():
            if concept in text_lower:
                detected_concepts.append(concept)
        
        # Detect domains
        detected_domains = []
        for domain, domain_concepts in self.kg.domains.items():
            if any(c in detected_concepts for c in domain_concepts):
                detected_domains.append(domain)
        
        # Fallback
        if not detected_concepts:
            detected_concepts = ['general']
        if not detected_domains:
            detected_domains = ['general']
        
        return {
            'concepts': detected_concepts,
            'domains': detected_domains
        }
    
    def generate_questions(self, user_input: str, num_questions: int = 5) -> Dict:
        """
        Generate questions using Graph RAG
        
        Returns:
        {
            'questions': [...],
            'graph_insights': {...},
            'problem_id': '...'
        }
        """
        logger.info(f"Generating questions for: {user_input}")
        
        # Step 1: Extract concepts and domains
        extracted = self.extract_concepts_and_domains(user_input)
        concepts = extracted['concepts']
        domains = extracted['domains']
        
        logger.info(f"Detected - Domains: {domains}, Concepts: {concepts}")
        
        # Step 2: Find similar problems
        similar_problems = self.kg.find_similar_problems(concepts, domains)
        
        # Step 3: Get relevant questions from graph
        graph_questions = self.kg.get_questions_for_concepts(concepts)
        
        # Step 4: Build context
        context = self._build_context(user_input, domains, concepts, similar_problems, graph_questions)
        
        # Step 5: Generate with LLM
        generated_questions = self._generate_with_llm(context, num_questions)
        
        # Step 6: Store in graph
        problem_id = f"prob_{len(self.kg.problems) + 1:03d}"
        self.kg.add_problem(problem_id, user_input, domains, concepts)
        
        return {
            'questions': generated_questions,
            'graph_insights': {
                'domains': domains,
                'concepts': concepts,
                'similar_problems': similar_problems
            },
            'problem_id': problem_id
        }
    
    def _build_context(
        self,
        user_input: str,
        domains: List[str],
        concepts: List[str],
        similar_problems: List[Dict],
        graph_questions: List[Dict]
    ) -> str:
        """Build context for LLM"""
        
        context_parts = [
            f"USER PROBLEM: {user_input}",
            "",
            "DETECTED DOMAINS:",
            *[f"  - {d}" for d in domains],
            "",
            "KEY CONCEPTS:",
            *[f"  - {c}" for c in concepts],
            ""
        ]
        
        if similar_problems:
            context_parts.extend([
                "SIMILAR PROBLEMS FROM KNOWLEDGE BASE:",
                *[f"  - {p['problem_text']} (Similarity: {p['similarity']:.0%})" for p in similar_problems[:3]],
                ""
            ])
        
        if graph_questions:
            context_parts.extend([
                "RECOMMENDED QUESTIONS FROM GRAPH:",
                *[f"  - [{q['priority']}] {q['text']}" for q in graph_questions[:5]],
                ""
            ])
        
        return "\n".join(context_parts)
    
    def _generate_with_llm(self, context: str, num_questions: int) -> List[Dict]:
        """Generate questions using LLM"""
        
        prompt = f"""
You are a business analyst helping to understand a business problem.

Based on the context below, generate {num_questions} clarifying questions that will help
gather the most important information to understand and solve this problem.

{context}

Generate questions that:
1. Build on the recommended questions from the knowledge graph
2. Are specific and actionable
3. Follow a logical sequence (current state → goals → constraints → solutions)
4. Cover different aspects (metrics, timeline, stakeholders, resources)

Return as JSON array:
[
    {{
        "question": "question text",
        "category": "current_situation|goals|constraints|stakeholders|metrics",
        "priority": "critical|high|medium",
        "reasoning": "why this question is important"
    }}
]

Return ONLY the JSON array, no other text.
"""
        
        try:
            response = self.llm.generate(prompt)
            
            # Try to parse JSON
            import json
            
            # Clean response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            questions = json.loads(response)
            
            if isinstance(questions, list):
                # Add source
                for q in questions:
                    q['source'] = 'graph_rag'
                return questions[:num_questions]
            else:
                return []
        
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            
            # Fallback to graph questions
            fallback = []
            for q in self.kg.questions[:num_questions]:
                fallback.append({
                    'question': q['text'],
                    'category': q['category'],
                    'priority': q['priority'],
                    'reasoning': 'From knowledge base',
                    'source': 'graph'
                })
            return fallback


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("🚀 STANDALONE GRAPH RAG SYSTEM")
    print("="*80 + "\n")
    
    # Initialize
    print("Initializing system...")
    system = StandaloneGraphRAG()
    print("✅ System ready\n")
    
    # Test query
    user_input = "How can we reduce customer churn in our SaaS product?"
    
    print(f"💬 User Input: '{user_input}'")
    print("-" * 80)
    
    # Generate questions
    result = system.generate_questions(user_input, num_questions=5)
    
    # Display results
    print("\n📊 GRAPH INSIGHTS:")
    print("-" * 80)
    
    insights = result['graph_insights']
    print(f"\n🏢 Domains: {', '.join(insights['domains'])}")
    print(f"💡 Concepts: {', '.join(insights['concepts'])}")
    
    if insights['similar_problems']:
        print(f"\n🔍 Similar Problems:")
        for prob in insights['similar_problems']:
            print(f"  • {prob['problem_text']} ({prob['similarity']:.0%})")
    
    print("\n\n❓ GENERATED QUESTIONS:")
    print("-" * 80)
    
    for i, q in enumerate(result['questions'], 1):
        print(f"\n{i}. {q['question']}")
        print(f"   Category: {q.get('category', 'N/A')}")
        print(f"   Priority: {q.get('priority', 'N/A')}")
        print(f"   Source: {q.get('source', 'N/A')}")
        if q.get('reasoning'):
            print(f"   Why: {q['reasoning']}")
    
    print("\n" + "="*80)
    print("✅ Demo completed!")
    print("="*80 + "\n")
