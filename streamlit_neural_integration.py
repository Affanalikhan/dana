"""
Streamlit Integration with Demo Neural System
Ready to use immediately with your existing app
"""

import streamlit as st
import sys
import os

# Add current directory to path
sys.path.append('.')

# Import the demo system (working right now!)
from test_neural_system_demo import DemoNeuralSystem

@st.cache_resource
def load_demo_neural_system():
    """Load demo neural system (works immediately)"""
    try:
        system = DemoNeuralSystem()
        return system
    except Exception as e:
        st.error(f"Demo system error: {e}")
        return None

def get_neural_questions(business_question: str):
    """Get questions using neural system"""
    
    system = load_demo_neural_system()
    if system:
        try:
            # Create session and get questions
            session_id = system.create_session(business_question)
            questions = system.get_session_questions(session_id)
            
            # Convert to your app's format
            formatted_questions = []
            for q in questions:
                # Check if this is a clarification question with custom options
                if q.category == "clarification" and hasattr(q, 'clarification_options'):
                    formatted_questions.append({
                        "question": q.question_text,
                        "options": q.clarification_options,
                        "category": q.category,
                        "priority": q.priority,
                        "reasoning": q.reasoning,
                        "info_gain": q.information_gain_score,
                        "is_clarification": True
                    })
                else:
                    # Regular questions with standard options
                    formatted_questions.append({
                        "question": q.question_text,
                        "options": ["Yes", "No", "Partially", "Not sure", "Need more information"],
                        "category": q.category,
                        "priority": q.priority,
                        "reasoning": q.reasoning,
                        "info_gain": q.information_gain_score,
                        "is_clarification": False
                    })
            
            st.success("🧠 Using Neural System - Advanced pattern recognition active!")
            return formatted_questions, session_id
            
        except Exception as e:
            st.error(f"Neural system error: {e}")
    
    # Fallback to your existing Groq API
    try:
        from groq_llm import generate_initial_questions
        questions = generate_initial_questions(business_question)
        st.info("🤖 Using Groq API fallback")
        return questions, None
        
    except Exception as e:
        st.error(f"All systems failed: {e}")
        return [], None

def process_neural_answer(session_id, question_id, answer_text):
    """Process answer with neural analysis"""
    
    system = load_demo_neural_system()
    if system and session_id:
        try:
            # Analyze answer
            answer_analysis, followups = system.submit_answer(
                session_id, question_id, answer_text
            )
            
            # Display insights
            if answer_analysis.extracted_insights:
                st.info("🔍 **Insights detected:** " + ", ".join(answer_analysis.extracted_insights))
            
            # Show quality metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Completeness", f"{answer_analysis.completeness_score:.0%}")
            with col2:
                st.metric("Clarity", f"{1-answer_analysis.vagueness_score:.0%}")
            with col3:
                st.metric("Insights", len(answer_analysis.extracted_insights))
            
            # Show follow-ups if needed
            if followups:
                st.write("**Follow-up questions:**")
                for fq in followups:
                    st.write(f"• {fq.question_text}")
            
            return answer_analysis, followups
            
        except Exception as e:
            st.error(f"Answer analysis failed: {e}")
    
    return None, []

def generate_neural_summary(session_id):
    """Generate neural business understanding summary"""
    
    system = load_demo_neural_system()
    if system and session_id:
        try:
            summary = system.generate_business_understanding_summary(session_id)
            
            st.header("🧠 Neural Business Understanding Summary")
            
            # Problem analysis
            st.subheader("Problem Analysis")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Domain", summary['business_problem']['domain'])
            with col2:
                st.metric("Intent", summary['business_problem']['problem_type'])
            with col3:
                st.metric("Complexity", f"{summary['business_problem']['complexity_score']:.0%}")
            
            # Insights
            if summary['unique_insights']:
                st.subheader("Key Insights")
                for insight in summary['unique_insights']:
                    st.write(f"• {insight}")
            
            # Recommendations
            st.subheader("Recommendations")
            for rec in summary['recommendations']:
                st.write(f"✅ {rec}")
            
            # Next steps
            st.subheader("Next Steps")
            for step in summary['next_steps']:
                st.write(f"🎯 {step}")
            
            # Quality metrics
            st.subheader("Analysis Quality")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Questions Asked", summary['total_questions_asked'])
            with col2:
                st.metric("Completeness", f"{summary['overall_completeness_score']:.0%}")
            with col3:
                st.metric("Confidence", f"{summary['confidence_score']:.0%}")
            
            return summary
            
        except Exception as e:
            st.error(f"Summary generation failed: {e}")
    
    return None

# Example usage in your main app
def main():
    st.title("🧠 Neural Business Understanding AI")
    
    # Check system status
    system = load_demo_neural_system()
    if system:
        st.success("✅ Neural System Active - Advanced pattern recognition enabled!")
    else:
        st.warning("⚠️ Neural System not available - using API fallback")
    
    # Business question input
    business_question = st.text_area(
        "What business challenge would you like to understand?",
        placeholder="e.g., How can we reduce customer churn in our SaaS product?"
    )
    
    if st.button("🚀 Analyze Problem") and business_question:
        with st.spinner("Analyzing your business problem with neural intelligence..."):
            
            # Generate questions with neural system
            questions, session_id = get_neural_questions(business_question)
            
            if questions:
                st.session_state.questions = questions
                st.session_state.session_id = session_id
                st.session_state.current_question = 0
                st.session_state.answers = []
                st.rerun()

if __name__ == "__main__":
    main()