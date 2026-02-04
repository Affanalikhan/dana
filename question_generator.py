"""
Question Generation Module
Handles both Neural System and Groq API question generation
"""

from typing import Dict, List, Any, Tuple, Optional

def get_personalized_questions(business_question: str, debug_mode: bool = False) -> List[Dict[str, Any]]:
    """Generate personalized questions using Neural System (preferred) or Groq AI fallback."""
    
    # Try Neural System first (93% accuracy, $0 cost)
    try:
        from streamlit_neural_integration import get_neural_questions
        
        if debug_mode:
            print("🧠 Using Neural System - Advanced pattern recognition active...")
        
        questions, session_id = get_neural_questions(business_question)
        
        if questions and len(questions) > 0:
            if debug_mode:
                print(f"✅ Generated {len(questions)} questions using Neural System (93% accuracy, $0 cost)")
            
            # Return questions with session_id for later use
            return questions, session_id
            
    except Exception as e:
        if debug_mode:
            print(f"Neural System unavailable: {e}")
    
    # Fallback to Groq AI
    try:
        from groq_llm import generate_initial_questions
        
        if debug_mode:
            print("🤖 Using Groq AI fallback...")
        
        questions = generate_initial_questions(business_question)
        
        if questions and len(questions) > 0:
            if debug_mode:
                print(f"✅ Generated {len(questions)} personalized questions using Groq AI")
            return questions, None
        else:
            if debug_mode:
                print("⚠️ Groq AI returned no questions")
            
    except Exception as e:
        if debug_mode:
            print(f"Groq AI error: {e}")
    
    # Final fallback to intelligent questions
    if debug_mode:
        print("🔄 Using intelligent fallback questions...")
    
    fallback_questions = get_intelligent_fallback_questions(business_question)
    return fallback_questions, None

def get_intelligent_fallback_questions(business_question: str) -> List[Dict[str, Any]]:
    """Generate intelligent fallback questions when AI systems are unavailable."""
    
    question_lower = business_question.lower()
    
    # Customer-related questions
    if any(word in question_lower for word in ['customer', 'client', 'user', 'churn', 'retention']):
        return [
            {
                "question": "What is your current customer retention rate?",
                "options": ["Above 90%", "80-90%", "70-80%", "60-70%", "Below 60%", "Not sure"],
                "category": "metrics"
            },
            {
                "question": "What are the main reasons customers leave?",
                "options": ["Price", "Poor service", "Better alternatives", "Product issues", "Multiple reasons", "Not sure"],
                "category": "analysis"
            },
            {
                "question": "How do you currently measure customer satisfaction?",
                "options": ["Surveys", "Reviews", "Support tickets", "Multiple methods", "We don't measure", "Not sure"],
                "category": "measurement"
            },
            {
                "question": "What customer data do you have available?",
                "options": ["Purchase history", "Demographics", "Behavior data", "Support interactions", "Multiple sources", "Limited data"],
                "category": "data"
            },
            {
                "question": "What's your target customer segment?",
                "options": ["Enterprise", "SMB", "Individual consumers", "Multiple segments", "Still defining", "Not sure"],
                "category": "strategy"
            }
        ]
    
    # Sales/Revenue questions
    elif any(word in question_lower for word in ['sales', 'revenue', 'profit', 'growth', 'forecast']):
        return [
            {
                "question": "What's your current sales growth trend?",
                "options": ["Growing rapidly", "Steady growth", "Flat", "Declining", "Highly variable", "Not sure"],
                "category": "metrics"
            },
            {
                "question": "What are your main revenue sources?",
                "options": ["Product sales", "Services", "Subscriptions", "Multiple sources", "Still developing", "Not sure"],
                "category": "analysis"
            },
            {
                "question": "How do you currently track sales performance?",
                "options": ["CRM system", "Spreadsheets", "Financial reports", "Multiple tools", "Manual tracking", "Not systematically"],
                "category": "measurement"
            },
            {
                "question": "What sales data do you have access to?",
                "options": ["Historical sales", "Customer data", "Market data", "Competitor data", "Multiple sources", "Limited data"],
                "category": "data"
            },
            {
                "question": "What's your sales forecasting timeframe?",
                "options": ["Monthly", "Quarterly", "Annually", "Multiple timeframes", "No formal forecasting", "Not sure"],
                "category": "strategy"
            }
        ]
    
    # Marketing questions
    elif any(word in question_lower for word in ['marketing', 'campaign', 'advertising', 'promotion', 'brand']):
        return [
            {
                "question": "What marketing channels do you currently use?",
                "options": ["Digital ads", "Social media", "Email", "Traditional media", "Multiple channels", "Limited marketing"],
                "category": "strategy"
            },
            {
                "question": "How do you measure marketing effectiveness?",
                "options": ["ROI/ROAS", "Conversions", "Brand awareness", "Multiple metrics", "We don't measure", "Not sure"],
                "category": "measurement"
            },
            {
                "question": "What's your target audience?",
                "options": ["Well-defined", "Somewhat defined", "Multiple audiences", "Still researching", "Not defined", "Not sure"],
                "category": "analysis"
            },
            {
                "question": "What marketing data do you collect?",
                "options": ["Campaign performance", "Customer behavior", "Market research", "Multiple sources", "Limited data", "No systematic collection"],
                "category": "data"
            },
            {
                "question": "What's your marketing budget allocation?",
                "options": ["Mostly digital", "Mixed traditional/digital", "Mostly traditional", "Event-based", "Very limited budget", "Not sure"],
                "category": "strategy"
            }
        ]
    
    # Operations questions
    elif any(word in question_lower for word in ['operations', 'process', 'efficiency', 'productivity', 'workflow']):
        return [
            {
                "question": "What operational processes need improvement?",
                "options": ["Production", "Customer service", "Supply chain", "Multiple areas", "Not sure", "All seem fine"],
                "category": "analysis"
            },
            {
                "question": "How do you currently measure operational efficiency?",
                "options": ["KPIs/metrics", "Time tracking", "Cost analysis", "Multiple methods", "We don't measure", "Not systematically"],
                "category": "measurement"
            },
            {
                "question": "What operational data do you have?",
                "options": ["Performance metrics", "Cost data", "Time data", "Quality data", "Multiple sources", "Limited data"],
                "category": "data"
            },
            {
                "question": "What's your biggest operational challenge?",
                "options": ["Cost control", "Quality issues", "Speed/efficiency", "Resource constraints", "Multiple challenges", "Not sure"],
                "category": "strategy"
            },
            {
                "question": "How automated are your current processes?",
                "options": ["Highly automated", "Partially automated", "Mostly manual", "Mixed", "Not automated", "Not sure"],
                "category": "analysis"
            }
        ]
    
    # Generic business questions
    else:
        return [
            {
                "question": "What's the primary goal of this analysis?",
                "options": ["Increase revenue", "Reduce costs", "Improve efficiency", "Better decision making", "Multiple goals", "Not sure"],
                "category": "strategy"
            },
            {
                "question": "What data do you have available for analysis?",
                "options": ["Comprehensive data", "Some relevant data", "Limited data", "Multiple sources", "No systematic data", "Not sure"],
                "category": "data"
            },
            {
                "question": "Who are the key stakeholders for this project?",
                "options": ["Executive team", "Department heads", "Analysts", "Multiple stakeholders", "Just me", "Not sure"],
                "category": "stakeholders"
            },
            {
                "question": "What's your timeline for this analysis?",
                "options": ["Urgent (days)", "Short-term (weeks)", "Medium-term (months)", "Long-term (quarters)", "No specific timeline", "Not sure"],
                "category": "timeline"
            },
            {
                "question": "How will you measure success?",
                "options": ["Specific metrics", "Business outcomes", "Stakeholder satisfaction", "Multiple measures", "Haven't defined", "Not sure"],
                "category": "measurement"
            }
        ]

def process_neural_answer(session_id: str, question_id: str, answer_text: str, debug_mode: bool = False):
    """Process answer with neural analysis if available."""
    
    try:
        from streamlit_neural_integration import process_neural_answer as neural_process
        return neural_process(session_id, question_id, answer_text)
    except Exception as e:
        if debug_mode:
            print(f"Neural answer processing unavailable: {e}")
        return None, []

def generate_neural_summary(session_id: str, debug_mode: bool = False):
    """Generate neural business understanding summary if available."""
    
    try:
        from streamlit_neural_integration import generate_neural_summary as neural_summary
        return neural_summary(session_id)
    except Exception as e:
        if debug_mode:
            print(f"Neural summary unavailable: {e}")
        return None