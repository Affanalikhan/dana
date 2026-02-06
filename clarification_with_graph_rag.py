"""
Business Clarification Agent with Knowledge Graph + Graph RAG

Combines:
1. Conversational clarification (like GPT/Claude)
2. Knowledge Graph (structured business knowledge)
3. Graph RAG (intelligent retrieval + generation)
4. Multi-provider LLM support (Groq, Grok, OpenAI)

Features:
- Learns from past conversations
- Uses similar problems to ask better questions
- Stores business patterns in graph
- Gets smarter over time
- Supports multiple LLM providers
"""

from unified_llm_wrapper import UnifiedLLM
import pandas as pd
import json
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
from standalone_graph_rag import InMemoryKnowledgeGraph

load_dotenv()


class GraphEnhancedClarificationAgent:
    """
    Clarification agent enhanced with Knowledge Graph + Graph RAG
    
    Benefits:
    - Learns from past conversations
    - Asks better questions based on similar problems
    - Stores business patterns
    - Gets smarter with each use
    """
    
    def __init__(self, api_key: str = None, provider: str = None):
        """
        Initialize agent with LLM and Knowledge Graph
        
        Args:
            api_key: API key (optional, uses env variable)
            provider: LLM provider - 'groq', 'grok', or 'openai' (optional, uses env LLM_PROVIDER)
        """
        # Initialize LLM with unified wrapper
        self.provider = provider or os.getenv("LLM_PROVIDER", "groq")
        self.llm = UnifiedLLM(provider=self.provider, api_key=api_key)
        
        # Initialize Knowledge Graph
        self.kg = InMemoryKnowledgeGraph()
        
        # Conversation state
        self.conversation_history = []
        self.data_summary = None
        self.questions_asked = 0
        self.current_problem_id = None
        
        print(f"✅ Agent initialized with {self.provider.upper()} + Knowledge Graph")
    
    def start_conversation(
        self,
        business_question: str,
        csv_path: Optional[str] = None
    ) -> str:
        """
        Start clarification conversation with Graph RAG enhancement
        
        Args:
            business_question: User's business question
            csv_path: Optional path to CSV data
        
        Returns:
            First batch of questions (enhanced by graph knowledge)
        """
        # Analyze data if provided
        if csv_path:
            self.data_summary = self._analyze_data(csv_path)
        
        # Extract concepts and domains from question
        extracted = self._extract_concepts_and_domains(business_question)
        concepts = extracted['concepts']
        domains = extracted['domains']
        
        print(f"📊 Detected - Domains: {domains}, Concepts: {concepts}")
        
        # Find similar problems in knowledge graph
        similar_problems = self.kg.find_similar_problems(concepts, domains)
        
        # Get relevant questions from graph
        graph_questions = self.kg.get_questions_for_concepts(concepts)
        
        # Build enhanced context
        graph_context = self._build_graph_context(
            business_question,
            domains,
            concepts,
            similar_problems,
            graph_questions
        )
        
        # System prompt with graph enhancement
        system_prompt = f"""You are an expert Data Scientist and Business Analyst specializing in CRISP-DM Business Understanding.

Your role:
1. Ask ONE clarification question at a time (not multiple questions)
2. For EACH question, provide 4-6 specific multiple-choice options
3. Format as:
   - Question text (clear and specific)
   - Why you're asking (brief explanation)
   - 4-6 clickable options (specific to their context)
   - Always include "Other (please specify)" as last option

4. Use the knowledge graph context to ask BETTER questions
5. **IMPORTANT: Reference their uploaded data columns when relevant**
6. **Use data insights to ask more specific questions**

KNOWLEDGE GRAPH CONTEXT:
{graph_context}

Current business question: "{business_question}"

{f'''DATA ANALYSIS:
{self.data_summary['summary']}

Columns Available:
- Numeric: {', '.join(self.data_summary.get('numeric_columns', [])[:10])}
- Categorical: {', '.join(self.data_summary.get('categorical_columns', [])[:10])}
- Date: {', '.join(self.data_summary.get('date_columns', [])[:5])}

Data Quality: {self.data_summary['data_quality']['completeness']} complete, {self.data_summary['data_quality']['duplicate_rows']} duplicates

AI Insights:
{self.data_summary.get('ai_insights', 'No insights available')}

**USE THIS DATA INFORMATION IN YOUR QUESTIONS!**
- Reference specific columns by name
- Ask about data quality for relevant columns
- Suggest analyses based on available columns
''' if self.data_summary else "No data uploaded"}

IMPORTANT:
- ONE question at a time
- Provide specific multiple-choice options
- Make options relevant to their business context
- **Reference data columns when applicable**
- Use AI insights to ask smarter questions

EXAMPLE FORMAT:
**What customer data do you currently have access to?**

💡 *Why I'm asking: Customer analysis effectiveness depends on available data quality and completeness.*

**Select your answer:**
- Comprehensive customer database with transaction history
- Basic customer demographics and contact info
- Customer interaction and engagement data
- Support tickets and feedback data
- Limited customer data
- Other (please specify)

CRITICAL: Always use "- " (dash + space) for options. Each option should be on its own line."""
        
        # Generate first batch with graph enhancement
        initial_prompt = f"""I need help with: {business_question}

{f"I have data with columns: {', '.join(self.data_summary['columns'][:10])}" if self.data_summary else ""}

Based on the knowledge graph context, please ask me your FIRST clarification question.

Remember:
- Ask ONE question at a time
- Provide 4-6 specific multiple-choice options
- Include brief explanation of why you're asking
- Make options relevant to my business context
- Reference my data if applicable"""
        
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": initial_prompt}
                ],
                temperature=0.7,
                max_tokens=1500,
                timeout=20.0
            )
            
            assistant_response = response
        
        except Exception as e:
            print(f"⚠️ {self.provider.upper()} API Error: {e}")
            print("Using graph-enhanced fallback...")
            assistant_response = self._get_graph_fallback_questions(
                business_question,
                domains,
                concepts,
                similar_problems,
                graph_questions
            )
        
        # Store in conversation
        self.conversation_history.append({
            "role": "system",
            "content": system_prompt
        })
        self.conversation_history.append({
            "role": "user",
            "content": f"Business Question: {business_question}"
        })
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        self.questions_asked += assistant_response.count('?')
        
        # Store problem in graph
        import hashlib
        self.current_problem_id = f"prob_{hashlib.md5(business_question.encode()).hexdigest()[:8]}"
        self.kg.add_problem(
            self.current_problem_id,
            business_question,
            domains,
            concepts
        )
        
        return assistant_response
    
    def answer_questions(self, user_answers: str) -> str:
        """
        Process answers and ask next batch (with graph enhancement)
        
        Args:
            user_answers: User's responses
        
        Returns:
            Next questions or final summary
        """
        # Add to conversation
        self.conversation_history.append({
            "role": "user",
            "content": user_answers
        })
        
        # Determine next step
        if self.questions_asked >= 12:
            prompt = f"""You've asked about {self.questions_asked} questions.

If you have ONE more CRITICAL question with options, ask it now.

Otherwise, provide a comprehensive Business Understanding Summary."""
        else:
            prompt = f"""You've asked about {self.questions_asked} questions.

Ask your NEXT clarification question with 4-6 multiple-choice options.

Focus on filling gaps in: success metrics, data quality, constraints, expected outcomes."""
        
        try:
            response = self.llm.chat(
                messages=self.conversation_history + [
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2000,
                timeout=20.0
            )
            
            assistant_response = response
        
        except Exception as e:
            print(f"⚠️ {self.provider.upper()} API Error: {e}")
            assistant_response = self._get_fallback_followup(self.questions_asked)
        
        self.conversation_history.append({
            "role": "assistant",
            "content": assistant_response
        })
        
        self.questions_asked += assistant_response.count('?')
        
        return assistant_response
    
    def get_final_summary(self) -> str:
        """Get final business understanding summary"""
        
        prompt = """Based on our entire conversation, provide a comprehensive Business Understanding document:

# Business Understanding Summary

## 1. Problem Statement
[Clear, specific problem]

## 2. Business Objectives
[Success criteria and metrics]

## 3. Current Situation
[Baseline and current state]

## 4. Success Metrics
[How to measure success]

## 5. Constraints
[Time, budget, technical constraints]

## 6. Data Characteristics
[Available data and quality]

## 7. Recommended Approach
[ML/Analytics approach with reasoning]

## 8. Next Steps
[Specific actions for data prep and modeling]

## 9. Lessons from Similar Problems
[Insights from knowledge graph]

Make this actionable for CRISP-DM next phases."""
        
        try:
            response = self.llm.chat(
                messages=self.conversation_history + [
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=3000,
                timeout=30.0
            )
            
            return response
        
        except Exception as e:
            print(f"⚠️ {self.provider.upper()} API Error: {e}")
            return self._get_fallback_summary()
    
    def _extract_concepts_and_domains(self, text: str) -> Dict:
        """Extract business concepts and domains from text"""
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
    
    def _build_graph_context(
        self,
        question: str,
        domains: List[str],
        concepts: List[str],
        similar_problems: List[Dict],
        graph_questions: List[Dict]
    ) -> str:
        """Build context from knowledge graph"""
        
        context_parts = []
        
        context_parts.append(f"DETECTED DOMAINS: {', '.join(domains)}")
        context_parts.append(f"KEY CONCEPTS: {', '.join(concepts)}")
        context_parts.append("")
        
        if similar_problems:
            context_parts.append("SIMILAR PROBLEMS IN KNOWLEDGE BASE:")
            for prob in similar_problems[:3]:
                context_parts.append(f"  - {prob['problem_text']} (Similarity: {prob['similarity']:.0%})")
            context_parts.append("")
        
        if graph_questions:
            context_parts.append("RELEVANT QUESTIONS FROM GRAPH:")
            for q in graph_questions[:5]:
                context_parts.append(f"  - [{q['priority']}] {q['text']}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _get_graph_fallback_questions(
        self,
        business_question: str,
        domains: List[str],
        concepts: List[str],
        similar_problems: List[Dict],
        graph_questions: List[Dict]
    ) -> str:
        """Fallback questions enhanced with graph knowledge"""
        
        similar_context = ""
        if similar_problems:
            similar_context = f"\n\n💡 **I found {len(similar_problems)} similar problems in my knowledge base that can help guide our analysis.**\n"
        
        return f"""**What is your primary goal for this analysis?**

💡 *Why I'm asking: Understanding your end goal helps me ask the most relevant questions and recommend the right approach.*

**Select your answer:**
- Increase revenue or profitability
- Reduce costs or improve efficiency
- Improve customer satisfaction or retention
- Optimize a specific business process
- Make better data-driven decisions
- Other (please specify)

{similar_context}

Please select the option that best describes your situation, or provide additional details below."""
    
    def _get_fallback_followup(self, questions_asked: int) -> str:
        """Fallback follow-up questions"""
        
        if questions_asked >= 12:
            return """# 📋 Business Understanding Summary

Based on our conversation, I have gathered comprehensive information about your business problem.

## Key Insights Captured
- Problem scope and definition
- Current situation and baseline metrics
- Success criteria and target outcomes
- Data availability and quality assessment
- Constraints and limitations
- Stakeholder expectations

## Recommended Next Steps
1. **Data Preparation**: Clean and validate your data
2. **Feature Engineering**: Create relevant features based on our discussion
3. **Model Selection**: Choose appropriate algorithms for your use case
4. **Evaluation Framework**: Set up metrics aligned with your goals

## Ready for CRISP-DM Next Phase
You now have a solid business understanding foundation to proceed with data preparation and modeling.

Would you like me to provide more specific recommendations for any area?"""
        
        return f"""**What are your main constraints for this project?**

💡 *Why I'm asking: Understanding constraints helps set realistic expectations and choose the right approach.*

**Select your answer:**
- Limited time (need results quickly)
- Limited budget or resources
- Limited data availability or quality
- Technical infrastructure limitations
- Regulatory or compliance requirements
- Multiple constraints (please specify)"""
    
    def _get_fallback_summary(self) -> str:
        """Fallback summary"""
        return """# Business Understanding Summary

## Problem Statement
Based on our conversation, your business problem has been clarified.

## Key Points
- Problem definition and scope discussed
- Success metrics identified
- Current situation assessed
- Data availability confirmed

## Next Steps
1. Data Preparation
2. Feature Engineering
3. Model Development
4. Evaluation

## Ready for CRISP-DM Next Phase
You can proceed to data preparation."""
    
    def _analyze_data(self, csv_path: str) -> Dict:
        """Deep analysis of uploaded data"""
        try:
            df = pd.read_csv(csv_path)
            
            # Basic info
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
            
            # Try to detect date columns from object type
            for col in categorical_cols[:]:
                try:
                    pd.to_datetime(df[col].head(100), errors='raise')
                    date_cols.append(col)
                    categorical_cols.remove(col)
                except:
                    pass
            
            # Deep analysis
            analysis = {
                'rows': len(df),
                'columns': df.columns.tolist(),
                'numeric_columns': numeric_cols,
                'categorical_columns': categorical_cols,
                'date_columns': date_cols,
                'missing_values': df.isnull().sum().to_dict(),
                'summary': f"{len(df)} rows, {len(df.columns)} columns"
            }
            
            # Statistical summary for numeric columns
            if numeric_cols:
                analysis['numeric_stats'] = {}
                for col in numeric_cols[:10]:  # Limit to first 10
                    analysis['numeric_stats'][col] = {
                        'min': float(df[col].min()),
                        'max': float(df[col].max()),
                        'mean': float(df[col].mean()),
                        'median': float(df[col].median()),
                        'std': float(df[col].std()) if df[col].std() == df[col].std() else 0  # Check for NaN
                    }
            
            # Categorical analysis
            if categorical_cols:
                analysis['categorical_stats'] = {}
                for col in categorical_cols[:10]:  # Limit to first 10
                    unique_count = df[col].nunique()
                    analysis['categorical_stats'][col] = {
                        'unique_values': int(unique_count),
                        'top_values': df[col].value_counts().head(5).to_dict()
                    }
            
            # Data quality assessment
            total_cells = len(df) * len(df.columns)
            missing_cells = df.isnull().sum().sum()
            analysis['data_quality'] = {
                'completeness': f"{((total_cells - missing_cells) / total_cells * 100):.1f}%",
                'missing_percentage': f"{(missing_cells / total_cells * 100):.1f}%",
                'duplicate_rows': int(df.duplicated().sum())
            }
            
            # Intelligent insights using AI
            analysis['ai_insights'] = self._generate_data_insights(df, analysis)
            
            return analysis
            
        except Exception as e:
            return {'error': str(e), 'summary': f"Error: {str(e)}"}
    
    def _generate_data_insights(self, df: pd.DataFrame, analysis: Dict) -> str:
        """Generate AI-powered insights about the data"""
        try:
            # Build data summary for AI
            data_summary = f"""Dataset Overview:
- Rows: {len(df)}
- Columns: {len(df.columns)}
- Column Names: {', '.join(df.columns.tolist()[:20])}

Numeric Columns: {', '.join(analysis.get('numeric_columns', [])[:10])}
Categorical Columns: {', '.join(analysis.get('categorical_columns', [])[:10])}
Date Columns: {', '.join(analysis.get('date_columns', [])[:5])}

Data Quality:
- Completeness: {analysis['data_quality']['completeness']}
- Missing Data: {analysis['data_quality']['missing_percentage']}
- Duplicate Rows: {analysis['data_quality']['duplicate_rows']}
"""
            
            # Add sample data
            sample_data = df.head(3).to_string()
            
            prompt = f"""Analyze this dataset and provide 3-5 key insights:

{data_summary}

Sample Data:
{sample_data}

Provide insights about:
1. What type of business problem this data could solve
2. Key columns that seem important
3. Potential data quality issues
4. Suggested analysis approaches

Keep it brief (3-5 bullet points)."""
            
            insights = self.llm.generate(
                prompt=prompt,
                system_prompt="You are a data scientist analyzing business data.",
                temperature=0.7,
                max_tokens=300
            )
            
            return insights
        
        except Exception as e:
            return f"Could not generate AI insights: {str(e)}"
    
    def save_conversation(self, filepath: str):
        """Save conversation and graph state"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump({
                'conversation': self.conversation_history,
                'data_summary': self.data_summary,
                'problem_id': self.current_problem_id,
                'graph_state': {
                    'problems': self.kg.problems,
                    'questions': self.kg.questions
                }
            }, f, indent=2)
        print(f"✅ Conversation and graph state saved to {filepath}")


# CLI Interface
def run_cli():
    """CLI with Graph RAG enhancement"""
    print("="*80)
    print("BUSINESS CLARIFICATION AGENT + KNOWLEDGE GRAPH + GRAPH RAG")
    print("="*80)
    print("\n✨ Enhanced with:")
    print("  • Knowledge Graph - Learns from past conversations")
    print("  • Graph RAG - Uses similar problems to ask better questions")
    print("  • Gets smarter with each use\n")
    
    agent = GraphEnhancedClarificationAgent()
    
    print("="*80)
    business_question = input("\n💬 What is your business question?\n> ")
    
    print("\n" + "="*80)
    csv_path = input("\n📊 CSV file path (or press Enter to skip):\n> ")
    
    if csv_path and not os.path.exists(csv_path):
        print(f"⚠️ File not found: {csv_path}")
        csv_path = None
    
    print("\n" + "="*80)
    print("🚀 STARTING GRAPH-ENHANCED CLARIFICATION")
    print("="*80 + "\n")
    
    first_response = agent.start_conversation(
        business_question=business_question,
        csv_path=csv_path
    )
    
    print(first_response)
    
    # Conversation loop
    while True:
        print("\n" + "-"*80)
        user_input = input("\n💬 Your answers (or 'summary' for final report):\n> ")
        
        if user_input.lower() in ['summary', 'done', 'finish']:
            print("\n" + "="*80)
            print("📋 GENERATING FINAL SUMMARY")
            print("="*80 + "\n")
            
            summary = agent.get_final_summary()
            print(summary)
            
            # Save
            import datetime
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = f"business_understanding_{timestamp}.md"
            
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            print(f"\n✅ Summary saved to: {save_path}")
            agent.save_conversation(save_path.replace('.md', '_full.json'))
            break
        
        response = agent.answer_questions(user_input)
        print("\n" + response)


if __name__ == "__main__":
    run_cli()
