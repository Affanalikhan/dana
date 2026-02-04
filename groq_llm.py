import os
import json
from typing import List, Dict, Any, Optional, Tuple
from groq import Groq
from dotenv import load_dotenv
import time
import random

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Import CRISP-DM components
try:
    from crisp_dm_framework import Question, Answer, Dimension, QuestionType, CRISPDMFramework
    from adaptive_engine import AdaptiveEngine, AnalysisResult, ComplexityLevel, VaguenessLevel
    from analysis_engine import AnalysisEngine
    CRISP_DM_AVAILABLE = True
except ImportError:
    CRISP_DM_AVAILABLE = False
    print("Warning: CRISP-DM components not available. Running in basic mode.")

class CRISPDMPromptEngine:
    """
    Enhanced prompt engine for CRISP-DM methodology with intelligent 
    question generation and adaptive prompting capabilities.
    """
    
    def __init__(self):
        """Initialize the CRISP-DM prompt engine."""
        if CRISP_DM_AVAILABLE:
            self.framework = CRISPDMFramework()
            self.adaptive_engine = AdaptiveEngine()
            self.analysis_engine = AnalysisEngine()
        else:
            self.framework = None
            self.adaptive_engine = None
            self.analysis_engine = None
        
        self.fallback_questions = self._initialize_fallback_questions()
        self.context_templates = self._initialize_context_templates()
        self.crisp_dm_prompts = self._initialize_crisp_dm_prompts()
    
    def _initialize_fallback_questions(self) -> Dict[str, List[str]]:
        """Initialize fallback questions when AI is unavailable."""
        return {
            "problem_definition": [
                "What is the core business problem you're trying to solve?",
                "What triggered the need to address this problem now?",
                "How does this problem align with your overall business strategy?"
            ],
            "business_objectives": [
                "What specific, measurable outcomes do you want to achieve?",
                "What would constitute success for this initiative?",
                "What key performance indicators (KPIs) will you track?"
            ],
            "stakeholders": [
                "Who are the primary stakeholders for this analysis?",
                "Who will be the main users of the analysis results?",
                "Who has decision-making authority for implementing recommendations?"
            ],
            "current_situation": [
                "What is the current baseline state you're measuring against?",
                "What existing approaches have you tried to address this problem?",
                "What data and systems are currently available?"
            ],
            "constraints": [
                "What is your budget range for this analysis and implementation?",
                "What is your expected timeline for results?",
                "Are there any regulatory or compliance requirements?"
            ],
            "success_criteria": [
                "How will you measure the success of this analysis?",
                "What level of accuracy do you need in the results?",
                "What improvement threshold would make this worthwhile?"
            ],
            "business_domain": [
                "What industry or business domain are you in?",
                "Are there industry-specific regulations or standards?",
                "What are the key market dynamics affecting your business?"
            ],
            "implementation": [
                "How will the analysis results be integrated into your operations?",
                "What are the main barriers to implementing recommendations?",
                "Who will be responsible for implementing changes?"
            ]
        }
    
    def _initialize_context_templates(self) -> Dict[str, str]:
        """Initialize context templates for different scenarios."""
        return {
            "dimension_introduction": """
            We're now exploring the {dimension_name} aspect of your business understanding.
            This dimension focuses on {dimension_purpose}.
            Understanding this area is crucial because {dimension_importance}.
            """,
            "adaptive_followup": """
            Based on your previous response about {topic}, I'd like to explore this further.
            Your answer indicated {analysis_insight}, which suggests we should {followup_reason}.
            """,
            "contradiction_resolution": """
            I noticed some potentially conflicting information in your responses.
            Earlier you mentioned {statement_1}, but also indicated {statement_2}.
            This could impact our analysis, so let's clarify this.
            """,
            "assumption_validation": """
            In your response, I detected an assumption: {assumption}.
            Assumptions can significantly impact the success of data mining initiatives.
            Let's validate this to ensure our analysis is built on solid foundations.
            """
        }
    
    def _initialize_crisp_dm_prompts(self) -> Dict[str, str]:
        """Initialize CRISP-DM specific prompts for intelligent question generation."""
        return {
            "business_understanding_intro": """
            You are a CRISP-DM Business Understanding Specialist. Your role is to conduct thorough 
            business analysis following the Cross-Industry Standard Process for Data Mining methodology.
            
            You will ask strategic questions across 8 key dimensions:
            1. Problem Definition - Core problems, scope, triggers, strategic fit
            2. Business Objectives - Measurable goals, success criteria, KPIs
            3. Stakeholders - Primary stakeholders, decision makers, resistance points
            4. Current Situation - Baseline state, existing approaches, previous attempts
            5. Constraints - Budget, timeline, data availability, regulatory limits
            6. Success Criteria - Measurement approaches, improvement thresholds
            7. Business Domain - Industry context, regulations, market dynamics
            8. Implementation - Integration approaches, adoption barriers, change management
            
            Ask 5-7 questions per batch, ensuring comprehensive coverage while maintaining focus.
            """,
            
            "adaptive_question_generation": """
            Based on the user's previous responses, generate intelligent follow-up questions that:
            
            1. Address any vagueness or ambiguity in their answers
            2. Explore complex topics in more depth
            3. Validate assumptions they may have made
            4. Resolve any contradictions between responses
            5. Adapt to their specific business domain and context
            
            Previous conversation context: {conversation_context}
            
            Analysis insights: {analysis_insights}
            
            Generate 1-3 targeted follow-up questions that will provide the most valuable additional insight.
            """,
            
            "domain_adaptation": """
            The user appears to be in the {domain} industry. Adapt your questions to be more relevant 
            to this specific business context, considering:
            
            - Industry-specific terminology and concepts
            - Common challenges and opportunities in this domain
            - Regulatory requirements and compliance considerations
            - Typical stakeholder structures and decision-making processes
            - Domain-specific success metrics and KPIs
            
            Make your questions more targeted and valuable for their specific industry context.
            """,
            
            "contradiction_resolution": """
            I've detected potential contradictions in the user's responses:
            
            {contradictions}
            
            Generate clarifying questions that will help resolve these contradictions without 
            being confrontational. Focus on understanding their perspective and getting 
            consistent information.
            """,
            
            "assumption_validation": """
            The following assumptions have been detected in the user's responses:
            
            {assumptions}
            
            Generate questions that will help validate these assumptions and make implicit 
            beliefs explicit. This is crucial for ensuring the analysis is built on solid foundations.
            """
        }


def call_groq_llm(messages: List[Dict[str, str]], response_format: Optional[Dict] = None, 
                  max_retries: int = 3, retry_delay: float = 1.0) -> str:
    """Generic function to call the Groq API with rate limit handling."""
    try:
        params = {
            "model": "llama-3.1-8b-instant",  # Updated to current model
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2000,
        }
        
        # Note: Groq doesn't support response_format like OpenAI yet
        # We'll handle JSON parsing manually
        
        response = client.chat.completions.create(**params)
        return response.choices[0].message.content or ""
    except Exception as e:
        error_msg = str(e)
        print(f"Error in call_groq_llm: {error_msg}")
        
        # Check if it's a rate limit error
        if "rate_limit_exceeded" in error_msg or "429" in error_msg:
            print("⚠️ Groq API rate limit reached. Falling back to demo mode for this request.")
            raise Exception("RATE_LIMIT_EXCEEDED")
        
        return ""

def generate_initial_questions(business_question: str) -> List[Dict[str, Any]]:
    """Generate initial clarification question using Groq - one question at a time for natural conversation."""
    
    # More conversational, contextual prompt with specific options
    conversational_prompt = f"""
You are a senior business analyst having a conversation with a stakeholder. They just asked: "{business_question}"

As an experienced analyst, you need to understand their context better before providing insights. Ask ONE thoughtful clarification question that would help you understand their business context.

Return ONLY a JSON object with this format:
{{
  "question": "I'd like to understand your context better. [Your specific question here]?",
  "options": ["[Specific relevant option 1]", "[Specific relevant option 2]", "[Specific relevant option 3]", "[Specific relevant option 4]", "[Specific relevant option 5]", "I'm not sure"],
  "category": "Context Understanding",
  "reasoning": "Brief explanation of why this question helps"
}}

IMPORTANT: Make the options specific and relevant to their business question. For example:
- If they ask about customer segmentation, options might be: "Increase revenue", "Improve marketing campaigns", "Reduce customer churn", "Personalize customer experience", "General business insight"
- If they ask about revenue growth, options might be: "New customer acquisition", "Existing customer expansion", "Product optimization", "Market expansion", "Pricing strategy"

Make it conversational and provide meaningful, actionable options.
"""

    response = ""
    try:
        messages = [
            {"role": "system", "content": "You are a senior business analyst conducting a stakeholder interview. Be conversational and insightful."},
            {"role": "user", "content": conversational_prompt}
        ]
        
        response = call_groq_llm(messages)
        print(f"Raw Groq response: {response}")
        
        # Clean the response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        # Parse single question and return as array for compatibility
        parsed_response = json.loads(response)
        return [parsed_response]  # Return as single-item array
        
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating question with Groq: {error_msg}")
        print(f"Raw response was: {response}")
        
        # If rate limit exceeded, fall back to demo mode
        if "RATE_LIMIT_EXCEEDED" in error_msg:
            print("🔄 Falling back to demo questions due to rate limit")
            from demo_llm import generate_demo_questions
            return generate_demo_questions(business_question)
        
        return []

def generate_next_questions(
    business_question: str,
    previous_qa: List[Dict[str, str]]
) -> List[Dict[str, Any]]:
    """Generate the next conversational question based on previous answers."""
    
    # Build conversation context
    conversation_context = f"Original question: {business_question}\n\n"
    for i, qa in enumerate(previous_qa, 1):
        conversation_context += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"
    
    conversational_prompt = f"""
You are a senior business analyst continuing a conversation with a stakeholder. Here's the conversation so far:

{conversation_context}

Based on their answers, what's the MOST IMPORTANT follow-up question you would ask to gain deeper insight? 

Consider:
- What critical information is still missing?
- What would help you provide better recommendations?
- What business context do you need to understand?
- What are the practical constraints or requirements?

Ask ONE thoughtful follow-up question that builds on their previous answers.

Return ONLY a JSON object:
{{
  "question": "[Your natural, conversational follow-up question]?",
  "options": ["[Specific relevant option 1]", "[Specific relevant option 2]", "[Specific relevant option 3]", "[Specific relevant option 4]", "[Specific relevant option 5]", "I'm not sure"],
  "category": "Deep Dive",
  "reasoning": "Why this question is important based on their previous answers"
}}

IMPORTANT: Make the options specific and relevant to their business context and previous answers. Don't use generic "Option 1", "Option 2" - provide meaningful business choices that relate to their situation.

Be conversational and show that you're listening to their responses.
"""

    response = ""
    try:
        messages = [
            {"role": "system", "content": "You are a senior business analyst conducting an insightful stakeholder interview. Build on previous answers naturally."},
            {"role": "user", "content": conversational_prompt}
        ]
        
        response = call_groq_llm(messages)
        print(f"Raw Groq next question response: {response}")
        
        # Clean the response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        if response.strip():
            parsed_response = json.loads(response)
            return [parsed_response]  # Return as single-item array
        else:
            return []
            
    except Exception as e:
        error_msg = str(e)
        print(f"Error generating next question with Groq: {error_msg}")
        print(f"Raw response was: {response}")
        
        # If rate limit exceeded, fall back to demo mode
        if "RATE_LIMIT_EXCEEDED" in error_msg:
            print("🔄 Falling back to demo questions due to rate limit")
            from demo_llm import generate_demo_next_questions
            return generate_demo_next_questions(business_question, previous_qa)
        
        return []

def check_completeness(
    business_question: str,
    collected_answers: List[Dict[str, str]]
) -> Dict[str, Any]:
    """Check if we have enough information using conversational AI."""
    
    # Build conversation summary
    conversation_summary = f"Original question: {business_question}\n\n"
    for i, qa in enumerate(collected_answers, 1):
        conversation_summary += f"Q{i}: {qa['question']}\nA{i}: {qa['answer']}\n\n"
    
    completeness_prompt = f"""
You are a senior business analyst reviewing this conversation with a stakeholder:

{conversation_summary}

As an experienced analyst, do you have enough information to provide meaningful insights and recommendations? 

Consider:
- Do you understand their business context and objectives?
- Do you know what success looks like to them?
- Do you understand their constraints and requirements?
- Can you provide actionable recommendations?

Return ONLY a JSON object:
{{
  "complete": true/false,
  "confidence_score": 0-100,
  "missing_areas": ["area1", "area2"],
  "next_focus": "What should be explored next if not complete"
}}

Be realistic - you need enough context to provide valuable business insights.
"""

    response = ""
    try:
        messages = [
            {"role": "system", "content": "You are a senior business analyst evaluating conversation completeness."},
            {"role": "user", "content": completeness_prompt}
        ]
        
        response = call_groq_llm(messages)
        
        # Clean the response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        return json.loads(response)
        
    except Exception as e:
        print(f"Error checking completeness with Groq: {str(e)}")
        print(f"Raw response was: {response}")
        
        # Fallback logic - need at least 3-4 meaningful exchanges
        num_answers = len(collected_answers)
        if num_answers >= 4:
            return {"complete": True, "missing_areas": [], "confidence_score": 85}
        elif num_answers >= 2:
            return {"complete": False, "missing_areas": ["Implementation details", "Success criteria"], "confidence_score": 60}
        else:
            return {"complete": False, "missing_areas": ["Business context", "Objectives", "Constraints"], "confidence_score": 30}


# Enhanced CRISP-DM AI Integration Functions
def generate_crisp_dm_questions_with_ai(dimension: str, business_context: str, 
                                       previous_answers: List[Dict[str, str]] = None) -> List[Dict[str, Any]]:
    """
    Generate CRISP-DM questions using AI with business context awareness.
    
    Args:
        dimension: The CRISP-DM dimension to focus on
        business_context: Business context from initial question
        previous_answers: Previous answers for context
        
    Returns:
        List of generated questions with fallback to templates
    """
    try:
        # Build context for AI
        context_info = f"Business context: {business_context}\n"
        context_info += f"Current dimension: {dimension}\n"
        
        if previous_answers:
            context_info += "Previous answers:\n"
            for answer in previous_answers[-3:]:  # Last 3 answers for context
                context_info += f"- {answer.get('answer', '')[:100]}...\n"
        
        # Create AI prompt based on dimension
        dimension_prompts = {
            "problem_definition": "Focus on core problems, scope boundaries, triggers, and strategic alignment",
            "business_objectives": "Focus on measurable goals, success criteria, KPIs, and failure conditions",
            "stakeholders": "Focus on primary stakeholders, decision makers, end users, and potential resistance",
            "current_situation": "Focus on baseline state, existing approaches, previous attempts, and urgency",
            "constraints": "Focus on budget, timeline, data availability, and regulatory constraints",
            "success_criteria": "Focus on measurement approaches, improvement thresholds, and accuracy requirements",
            "business_domain": "Focus on industry context, regulations, market dynamics, and seasonal patterns",
            "implementation": "Focus on integration approaches, adoption barriers, and change management"
        }
        
        dimension_focus = dimension_prompts.get(dimension, "Focus on comprehensive business understanding")
        
        prompt = f"""
        You are a CRISP-DM Business Understanding Specialist conducting thorough business analysis.
        
        Context: {context_info}
        
        Generate 5-7 strategic questions for the {dimension} dimension. {dimension_focus}.
        
        Make questions:
        1. Specific to their business context
        2. Strategic and insightful
        3. Include multiple choice options where appropriate
        4. Build on previous answers if available
        
        Return as JSON array:
        [{{
            "question": "Question text",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"],
            "category": "{dimension}",
            "reasoning": "Why this question is important"
        }}]
        """
        
        messages = [
            {"role": "system", "content": "You are a CRISP-DM Business Understanding Specialist."},
            {"role": "user", "content": prompt}
        ]
        
        response = call_groq_llm(messages)
        
        # Parse response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        questions = json.loads(response)
        return questions if isinstance(questions, list) else [questions]
        
    except Exception as e:
        print(f"Error in AI question generation: {e}")
        # Fallback to basic questions
        return get_fallback_questions_for_dimension(dimension)


def generate_adaptive_followups_with_ai(answers: List[Dict[str, str]], 
                                      analysis_insights: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate adaptive follow-up questions based on answer analysis.
    
    Args:
        answers: List of previous Q&A pairs
        analysis_insights: Analysis results from adaptive engine
        
    Returns:
        List of adaptive follow-up questions
    """
    try:
        # Build conversation context
        conversation_context = ""
        for i, qa in enumerate(answers[-5:], 1):  # Last 5 Q&As
            conversation_context += f"Q{i}: {qa.get('question', '')}\n"
            conversation_context += f"A{i}: {qa.get('answer', '')[:150]}...\n\n"
        
        # Build analysis insights summary
        insights_summary = []
        if analysis_insights.get('complexity_level'):
            insights_summary.append(f"Complexity: {analysis_insights['complexity_level']}")
        if analysis_insights.get('vagueness_level'):
            insights_summary.append(f"Vagueness: {analysis_insights['vagueness_level']}")
        if analysis_insights.get('contradictions'):
            insights_summary.append(f"Contradictions detected: {len(analysis_insights['contradictions'])}")
        if analysis_insights.get('assumptions'):
            insights_summary.append(f"Assumptions detected: {len(analysis_insights['assumptions'])}")
        if analysis_insights.get('domain_indicators'):
            insights_summary.append(f"Domain: {', '.join(analysis_insights['domain_indicators'])}")
        
        insights_text = "\n".join(insights_summary)
        
        prompt = f"""
        Based on the user's previous responses, generate intelligent follow-up questions that:
        
        1. Address any vagueness or ambiguity in their answers
        2. Explore complex topics in more depth
        3. Validate assumptions they may have made
        4. Resolve any contradictions between responses
        5. Adapt to their specific business domain and context
        
        Previous conversation:
        {conversation_context}
        
        Analysis insights:
        {insights_text}
        
        Generate 1-3 targeted follow-up questions that will provide the most valuable additional insight.
        
        Return as JSON array:
        [{{
            "question": "Follow-up question text",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"],
            "category": "Adaptive Follow-up",
            "reasoning": "Why this follow-up is needed"
        }}]
        """
        
        messages = [
            {"role": "system", "content": "You are a CRISP-DM specialist focused on adaptive questioning."},
            {"role": "user", "content": prompt}
        ]
        
        response = call_groq_llm(messages)
        
        # Parse response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        questions = json.loads(response)
        return questions if isinstance(questions, list) else [questions]
        
    except Exception as e:
        print(f"Error generating adaptive followups: {e}")
        return []


def generate_domain_specific_questions(domain_indicators: List[str], 
                                     dimension: str) -> List[Dict[str, Any]]:
    """
    Generate domain-specific questions based on detected business domain.
    
    Args:
        domain_indicators: List of detected domain indicators
        dimension: Current CRISP-DM dimension
        
    Returns:
        List of domain-adapted questions
    """
    if not domain_indicators:
        return []
    
    try:
        primary_domain = domain_indicators[0]  # Use primary detected domain
        
        prompt = f"""
        The user appears to be in the {primary_domain} industry. Generate questions for the {dimension} 
        dimension that are specifically relevant to this business context, considering:
        
        - Industry-specific terminology and concepts
        - Common challenges and opportunities in {primary_domain}
        - Regulatory requirements and compliance considerations
        - Typical stakeholder structures and decision-making processes
        - Domain-specific success metrics and KPIs
        
        Generate 3-5 questions that are highly relevant to the {primary_domain} industry.
        
        Return as JSON array:
        [{{
            "question": "Domain-specific question text",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"],
            "category": "{dimension} - {primary_domain}",
            "reasoning": "Why this is important for {primary_domain}"
        }}]
        """
        
        messages = [
            {"role": "system", "content": f"You are a CRISP-DM specialist with deep {primary_domain} industry knowledge."},
            {"role": "user", "content": prompt}
        ]
        
        response = call_groq_llm(messages)
        
        # Parse response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        questions = json.loads(response)
        return questions if isinstance(questions, list) else [questions]
        
    except Exception as e:
        print(f"Error generating domain-specific questions: {e}")
        return []


def generate_contradiction_resolution_questions(contradictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate questions to resolve detected contradictions."""
    if not contradictions:
        return []
    
    try:
        contradiction_text = "\n".join([
            f"- {c.get('description', 'Contradiction detected')}" for c in contradictions
        ])
        
        prompt = f"""
        I've detected potential contradictions in the user's responses:
        
        {contradiction_text}
        
        Generate clarifying questions that will help resolve these contradictions without 
        being confrontational. Focus on understanding their perspective and getting 
        consistent information.
        
        Return as JSON array:
        [{{
            "question": "Clarifying question text",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"],
            "category": "Contradiction Resolution",
            "reasoning": "Why this clarification is needed"
        }}]
        """
        
        messages = [
            {"role": "system", "content": "You are a CRISP-DM specialist focused on clarification."},
            {"role": "user", "content": prompt}
        ]
        
        response = call_groq_llm(messages)
        
        # Parse response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        questions = json.loads(response)
        return questions if isinstance(questions, list) else [questions]
        
    except Exception as e:
        print(f"Error generating contradiction resolution questions: {e}")
        return []


def generate_assumption_validation_questions(assumptions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate questions to validate detected assumptions."""
    if not assumptions:
        return []
    
    try:
        assumption_text = "\n".join([
            f"- {a.get('description', 'Assumption detected')}" for a in assumptions
        ])
        
        prompt = f"""
        The following assumptions have been detected in the user's responses:
        
        {assumption_text}
        
        Generate questions that will help validate these assumptions and make implicit 
        beliefs explicit. This is crucial for ensuring the analysis is built on solid foundations.
        
        Return as JSON array:
        [{{
            "question": "Assumption validation question",
            "options": ["Option 1", "Option 2", "Option 3", "Option 4", "Option 5"],
            "category": "Assumption Validation",
            "reasoning": "Why this assumption needs validation"
        }}]
        """
        
        messages = [
            {"role": "system", "content": "You are a CRISP-DM specialist focused on assumption validation."},
            {"role": "user", "content": prompt}
        ]
        
        response = call_groq_llm(messages)
        
        # Parse response
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()
        
        questions = json.loads(response)
        return questions if isinstance(questions, list) else [questions]
        
    except Exception as e:
        print(f"Error generating assumption validation questions: {e}")
        return []


def get_fallback_questions_for_dimension(dimension: str) -> List[Dict[str, Any]]:
    """Get fallback questions when AI is unavailable."""
    fallback_questions = {
        "problem_definition": [
            {
                "question": "What is the core business problem you're trying to solve?",
                "options": ["Revenue decline", "Customer churn", "Operational inefficiency", "Market competition", "Other"],
                "category": "Problem Definition",
                "reasoning": "Understanding the fundamental problem helps scope the entire analysis"
            },
            {
                "question": "What triggered the need to address this problem now?",
                "options": ["Recent performance decline", "Strategic initiative", "Competitive pressure", "Regulatory requirement", "New opportunity"],
                "category": "Problem Definition", 
                "reasoning": "Identifying triggers helps understand urgency and context"
            }
        ],
        "business_objectives": [
            {
                "question": "What specific, measurable outcomes do you want to achieve?",
                "options": ["Increase revenue by X%", "Reduce costs by X%", "Improve efficiency by X%", "Increase customer satisfaction", "Other quantifiable goal"],
                "category": "Business Objectives",
                "reasoning": "Clear objectives guide analysis direction and success measurement"
            }
        ],
        "stakeholders": [
            {
                "question": "Who are the primary stakeholders for this analysis?",
                "options": ["Executive leadership", "Department managers", "End users", "External customers", "Multiple stakeholder groups"],
                "category": "Stakeholders",
                "reasoning": "Identifying stakeholders ensures proper alignment and communication"
            }
        ],
        "current_situation": [
            {
                "question": "What is the current baseline state you're measuring against?",
                "options": ["Well-documented current state", "Partially documented", "Estimated current state", "Unknown current state", "Inconsistent measurements"],
                "category": "Current Situation",
                "reasoning": "Baseline understanding is crucial for measuring improvement"
            }
        ],
        "constraints": [
            {
                "question": "What is your budget range for this analysis and implementation?",
                "options": ["Under $10K", "$10K-$50K", "$50K-$200K", "Over $200K", "Budget not yet determined"],
                "category": "Constraints",
                "reasoning": "Budget constraints affect scope and approach"
            }
        ],
        "success_criteria": [
            {
                "question": "How will you measure the success of this analysis?",
                "options": ["Quantitative improvements", "Qualitative insights", "Process improvements", "Decision support", "Risk reduction"],
                "category": "Success Criteria",
                "reasoning": "Clear success metrics ensure value delivery"
            }
        ],
        "business_domain": [
            {
                "question": "What industry or business domain are you in?",
                "options": ["Technology", "Healthcare", "Financial services", "Retail/E-commerce", "Manufacturing", "Other"],
                "category": "Business Domain",
                "reasoning": "Industry context affects analysis approach and benchmarks"
            }
        ],
        "implementation": [
            {
                "question": "How will the analysis results be integrated into your operations?",
                "options": ["Automated systems integration", "Manual process changes", "Decision support tools", "Strategic planning input", "Multiple integration approaches"],
                "category": "Implementation",
                "reasoning": "Implementation approach affects analysis design and outputs"
            }
        ]
    }
    
    return fallback_questions.get(dimension, [])


def create_enhanced_ai_integration() -> 'CRISPDMPromptEngine':
    """Create an enhanced AI integration instance with fallback mechanisms."""
    try:
        return CRISPDMPromptEngine()
    except Exception as e:
        print(f"Warning: Could not initialize enhanced AI integration: {e}")
        print("Falling back to basic functionality")
        return None