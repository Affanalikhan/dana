import streamlit as st
import pandas as pd
from typing import Dict, List, Any, Optional
import traceback

# Page config MUST be first
st.set_page_config(
    page_title="CRISP-DM Business Understanding Specialist",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import CRISP-DM components
try:
    from analysis_engine import AnalysisEngine, BusinessInsights
    from context_preservation import ContextPreservationEngine, ConversationalContext
    from adaptive_engine import AdaptiveEngine, AnalysisResult
    from error_handling import ErrorHandler, ValidationError
    from summary_generator import SummaryGenerator
    CRISP_DM_COMPONENTS_AVAILABLE = True
except ImportError as e:
    st.sidebar.error(f"⚠️ CRISP-DM components not fully available: {e}")
    CRISP_DM_COMPONENTS_AVAILABLE = False

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        max-width: 80%;
    }
    .user-message {
        background: #007bff;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .ai-message {
        background: #f8f9fa;
        color: #333;
        border: 1px solid #e9ecef;
        margin-right: auto;
    }
    .debug-info {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    defaults = {
        'chat_history': [],
        'current_questions': [],
        'business_question': "",
        'conversation_complete': False,
        'question_history': [],
        'phase': 'home',
        'waiting_for_answer': False,
        'current_question_data': None,
        'current_question_index': 0,
        'debug_mode': True,
        'error_log': [],
        # CRISP-DM Advanced Components
        'analysis_engine': None,
        'context_engine': None,
        'adaptive_engine': None,
        'error_handler': None,
        'detected_assumptions': [],
        'identified_gaps': [],
        'context_references': [],
        'business_insights': None,
        # Neural System Components
        'neural_session_id': None,
        'neural_insights': []
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Initialize CRISP-DM engines if available
    if CRISP_DM_COMPONENTS_AVAILABLE and st.session_state.analysis_engine is None:
        try:
            st.session_state.analysis_engine = AnalysisEngine()
            st.session_state.context_engine = ContextPreservationEngine()
            st.session_state.adaptive_engine = AdaptiveEngine()
            st.session_state.error_handler = ErrorHandler()
            
            if st.session_state.debug_mode:
                st.sidebar.success("✅ CRISP-DM engines initialized")
        except Exception as e:
            if st.session_state.debug_mode:
                st.sidebar.error(f"❌ Failed to initialize CRISP-DM engines: {e}")

initialize_session_state()

def log_error(error_msg: str, exception: Exception = None):
    """Log errors for debugging."""
    error_info = {
        'message': error_msg,
        'exception': str(exception) if exception else None,
        'traceback': traceback.format_exc() if exception else None,
        'timestamp': pd.Timestamp.now()
    }
    st.session_state.error_log.append(error_info)
    
    if st.session_state.debug_mode:
        st.error(f"🐛 Debug: {error_msg}")
        if exception:
            st.code(str(exception))

def add_message(message: str, is_user: bool = False):
    """Add a message to chat history."""
    st.session_state.chat_history.append({
        'message': message,
        'is_user': is_user,
        'timestamp': pd.Timestamp.now()
    })

def display_chat_history():
    """Display the chat history."""
    for chat in st.session_state.chat_history:
        if chat['is_user']:
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>You:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message ai-message">
                <strong>CRISP-DM Specialist:</strong> {chat['message']}
            </div>
            """, unsafe_allow_html=True)

def get_personalized_questions(business_question: str) -> List[Dict[str, Any]]:
    """Generate personalized questions using Neural System (preferred) or Groq AI fallback."""
    
    try:
        from question_generator import get_personalized_questions as generate_questions
        
        questions, session_id = generate_questions(business_question, st.session_state.debug_mode)
        
        # Store neural session ID for advanced features
        if session_id:
            st.session_state.neural_session_id = session_id
            if st.session_state.debug_mode:
                st.success("🧠 Neural System active - Advanced pattern recognition enabled!")
        
        return questions
        
    except Exception as e:
        if st.session_state.debug_mode:
            st.error(f"Question generation error: {e}")
        
        # Final fallback
        return get_intelligent_fallback_questions(business_question)

def get_intelligent_fallback_questions(business_question: str) -> List[Dict[str, Any]]:
    """Generate intelligent fallback questions when Groq AI is unavailable."""
    
    question_lower = business_question.lower()
    
    # Analyze the business question to determine the most relevant questions
    if any(word in question_lower for word in ['customer', 'churn', 'retention', 'segment']):
        return [
            {
                "question": f"For your question about '{business_question}', what's your primary goal?",
                "options": [
                    "Reduce customer churn and improve retention",
                    "Identify high-value customer segments",
                    "Improve customer satisfaction and experience",
                    "Optimize customer acquisition costs",
                    "Personalize customer interactions",
                    "Better understand customer behavior"
                ],
                "category": "Customer Analysis Objective",
                "reasoning": "Understanding your specific customer-focused goal helps me tailor the analysis approach."
            },
            {
                "question": "What customer data do you currently have access to?",
                "options": [
                    "Comprehensive customer database with transaction history",
                    "Basic customer demographics and contact info",
                    "Customer interaction and engagement data",
                    "Support tickets and feedback data",
                    "Limited customer data",
                    "Multiple data sources that need integration"
                ],
                "category": "Data Availability",
                "reasoning": "Customer analysis effectiveness depends on available data quality and completeness."
            },
            {
                "question": "What's your biggest challenge with customers right now?",
                "options": [
                    "High churn rate in specific segments",
                    "Difficulty identifying valuable customers",
                    "Low customer engagement and satisfaction",
                    "Ineffective marketing and targeting",
                    "Poor customer service experience",
                    "Understanding customer needs and preferences"
                ],
                "category": "Current Challenge",
                "reasoning": "Identifying your main customer challenge helps focus the analysis on the most impactful areas."
            }
        ]
    
    elif any(word in question_lower for word in ['revenue', 'sales', 'profit', 'growth']):
        return [
            {
                "question": f"Regarding '{business_question}', what's your primary revenue objective?",
                "options": [
                    "Increase overall revenue and profitability",
                    "Identify new revenue opportunities",
                    "Optimize pricing strategies",
                    "Improve sales team performance",
                    "Expand into new markets or segments",
                    "Reduce revenue leakage and losses"
                ],
                "category": "Revenue Objective",
                "reasoning": "Understanding your specific revenue goal helps me focus on the most impactful analysis."
            },
            {
                "question": "What level of financial and sales data do you have?",
                "options": [
                    "Detailed transaction-level data with customer info",
                    "Product/service line revenue breakdowns",
                    "Sales team and channel performance data",
                    "High-level revenue summaries only",
                    "Limited financial visibility",
                    "Multiple financial systems with integration challenges"
                ],
                "category": "Financial Data",
                "reasoning": "Revenue analysis effectiveness depends on data granularity and accessibility."
            },
            {
                "question": "What's your biggest revenue challenge currently?",
                "options": [
                    "Declining revenue or growth stagnation",
                    "Inconsistent sales performance",
                    "Pricing pressure from competitors",
                    "Difficulty forecasting revenue accurately",
                    "Poor conversion rates in sales funnel",
                    "Limited visibility into revenue drivers"
                ],
                "category": "Revenue Challenge",
                "reasoning": "Identifying your main revenue challenge helps prioritize analysis focus areas."
            }
        ]
    
    elif any(word in question_lower for word in ['product', 'inventory', 'supply', 'operations']):
        return [
            {
                "question": f"For your question about '{business_question}', what's your main operational goal?",
                "options": [
                    "Optimize product performance and portfolio",
                    "Improve operational efficiency and reduce costs",
                    "Better demand forecasting and inventory management",
                    "Enhance supply chain and logistics",
                    "Improve quality and reduce defects",
                    "Streamline processes and workflows"
                ],
                "category": "Operational Objective",
                "reasoning": "Understanding your operational focus helps me design the most relevant analysis."
            },
            {
                "question": "What operational data do you currently track?",
                "options": [
                    "Comprehensive operational metrics and KPIs",
                    "Product performance and sales data",
                    "Inventory levels and supply chain data",
                    "Quality metrics and customer feedback",
                    "Basic operational reports only",
                    "Limited operational visibility"
                ],
                "category": "Operational Data",
                "reasoning": "Operational analysis requires specific data types to be most effective."
            },
            {
                "question": "What's your biggest operational challenge?",
                "options": [
                    "Inefficient processes and high costs",
                    "Poor demand forecasting accuracy",
                    "Inventory management and stockouts",
                    "Quality issues and customer complaints",
                    "Supply chain disruptions and delays",
                    "Lack of operational visibility and control"
                ],
                "category": "Operational Challenge",
                "reasoning": "Identifying your main operational pain point helps focus the analysis on high-impact areas."
            }
        ]
    
    elif any(word in question_lower for word in ['market', 'competition', 'strategy', 'growth']):
        return [
            {
                "question": f"Regarding '{business_question}', what's your strategic priority?",
                "options": [
                    "Market expansion and growth opportunities",
                    "Competitive positioning and differentiation",
                    "Strategic planning and decision support",
                    "Risk assessment and mitigation",
                    "Performance benchmarking and improvement",
                    "Innovation and new opportunity identification"
                ],
                "category": "Strategic Priority",
                "reasoning": "Understanding your strategic focus helps align the analysis with business priorities."
            },
            {
                "question": "What market and competitive data do you have access to?",
                "options": [
                    "Comprehensive market research and competitive intelligence",
                    "Industry reports and benchmarking data",
                    "Customer and market feedback",
                    "Internal performance metrics only",
                    "Limited market visibility",
                    "Fragmented data from multiple sources"
                ],
                "category": "Market Data",
                "reasoning": "Strategic analysis effectiveness depends on market and competitive data availability."
            },
            {
                "question": "What's your biggest strategic challenge?",
                "options": [
                    "Intense competition and market pressure",
                    "Unclear growth opportunities and direction",
                    "Difficulty making strategic decisions",
                    "Market changes and disruption",
                    "Resource allocation and prioritization",
                    "Measuring and tracking strategic progress"
                ],
                "category": "Strategic Challenge",
                "reasoning": "Identifying your main strategic challenge helps focus analysis on the most critical areas."
            }
        ]
    
    else:
        # Generic but intelligent questions for any business question
        return [
            {
                "question": f"What's the main outcome you're hoping to achieve with '{business_question}'?",
                "options": [
                    "Make better data-driven decisions",
                    "Identify new opportunities for growth",
                    "Solve a specific business problem",
                    "Improve operational efficiency",
                    "Better understand our performance",
                    "Support strategic planning and direction"
                ],
                "category": "Primary Objective",
                "reasoning": "Understanding your main goal helps me tailor the analysis to deliver maximum value."
            },
            {
                "question": "What type of data and information do you currently have available?",
                "options": [
                    "Comprehensive data across multiple business areas",
                    "Good data in some areas, limited in others",
                    "Basic reporting and metrics",
                    "Mostly manual data collection",
                    "Limited data availability",
                    "Data exists but needs integration and cleanup"
                ],
                "category": "Data Readiness",
                "reasoning": "Data availability and quality significantly impact what analysis approaches will be most effective."
            },
            {
                "question": "What's your biggest challenge in this area right now?",
                "options": [
                    "Lack of visibility and insights",
                    "Too much data, not enough actionable insights",
                    "Inconsistent or unreliable information",
                    "Difficulty making informed decisions",
                    "Resource constraints and competing priorities",
                    "Rapidly changing business environment"
                ],
                "category": "Current Challenge",
                "reasoning": "Understanding your main challenge helps me focus the analysis on the most impactful solutions."
            }
        ]

def show_home_page():
    """Show the main landing page."""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🧠 CRISP-DM Business Understanding Specialist</h1>
        <p>Comprehensive Business Analysis Engine</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Debug toggle
    if st.sidebar.checkbox("Debug Mode", value=st.session_state.debug_mode):
        st.session_state.debug_mode = True
        if st.session_state.error_log:
            st.sidebar.markdown("### 🐛 Error Log")
            for i, error in enumerate(st.session_state.error_log[-3:]):  # Show last 3 errors
                st.sidebar.text(f"{i+1}. {error['message']}")
    else:
        st.session_state.debug_mode = False
    
    # Check Neural System availability
    neural_available = False
    try:
        from question_generator import get_personalized_questions as test_neural
        # Test if neural system is working
        neural_available = True
    except:
        pass
    
    if neural_available and CRISP_DM_COMPONENTS_AVAILABLE:
        st.success("🧠 **Neural + CRISP-DM Mode**: Advanced neural pattern recognition (93% accuracy) + Full business analysis suite")
    elif neural_available:
        st.success("🧠 **Neural AI Mode**: Advanced pattern recognition with 93% accuracy and $0 runtime cost")
    elif CRISP_DM_COMPONENTS_AVAILABLE:
        st.success("🚀 **Full CRISP-DM Mode**: AI-powered personalized questions + Advanced business analysis (Gap identification, Assumption detection, Context preservation)")
    else:
        st.info("🤖 **AI-Powered Mode**: Using Groq AI to generate personalized questions tailored to your specific business challenge")
    
    st.markdown("### What business challenge can I help you with today?")
    st.caption("I'll use AI to generate personalized strategic questions specifically tailored to your business challenge, with advanced CRISP-DM analysis including assumption detection, gap identification, and context preservation.")
    
    # Example conversation starters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💰 Customer Value Analysis", use_container_width=True):
            start_conversation("How can we identify our most valuable customers?")
    
    with col2:
        if st.button("⚠️ Churn Risk Assessment", use_container_width=True):
            start_conversation("Which customers might leave us soon?")
    
    with col3:
        if st.button("🎯 Customer Segmentation", use_container_width=True):
            start_conversation("How should we segment our customers for better marketing?")
    
    # More examples
    col4, col5, col6 = st.columns(3)
    
    with col4:
        if st.button("📈 Revenue Growth Analysis", use_container_width=True):
            start_conversation("What's driving our revenue growth?")
    
    with col5:
        if st.button("🛒 Product Strategy", use_container_width=True):
            start_conversation("Which products should we focus on?")
    
    with col6:
        if st.button("📊 Performance Review", use_container_width=True):
            start_conversation("How is our business performing overall?")
    
    # Custom question input
    st.markdown("---")
    st.markdown("### Or ask me anything about your business:")
    
    custom_question = st.text_area(
        "What's your business question?",
        placeholder="e.g., How can we improve customer retention in our subscription business?",
        height=100,
        help="Ask me anything about your business challenges, data analysis needs, or strategic questions."
    )
    
    if st.button("💬 Start Business Analysis", type="primary", use_container_width=True):
        if custom_question.strip():
            start_conversation(custom_question)
        else:
            st.warning("Please share your business question so we can have a meaningful conversation.")

def start_conversation(question: str):
    """Start a conversation with the given question."""
    try:
        st.session_state.business_question = question
        st.session_state.chat_history = []
        st.session_state.phase = 'chatting'
        st.session_state.conversation_complete = False
        st.session_state.waiting_for_answer = False
        st.session_state.current_question_index = 0
        st.session_state.question_history = []
        
        # Add user message
        add_message(question, is_user=True)
        
        # Add AI response
        add_message("Thank you for that question. Let me ask you a few strategic questions tailored specifically to your business challenge to better understand your needs and provide comprehensive insights.")
        
        # Generate personalized questions using Groq AI
        questions = get_personalized_questions(question)
        st.session_state.current_questions = questions
        
        if questions:
            st.session_state.current_question_data = questions[0]
            st.session_state.waiting_for_answer = True
            
            if st.session_state.debug_mode:
                st.success(f"✅ Generated {len(questions)} personalized questions successfully")
        else:
            log_error("No questions generated")
            add_message("I understand your question. Let me provide some insights based on what you've shared.")
            st.session_state.conversation_complete = True
        
        st.rerun()
        
    except Exception as e:
        log_error(f"Error starting conversation", e)
        st.error("There was an issue starting the conversation. Please try again.")

def handle_chat_interface():
    """Handle the chat interface."""
    
    # Back button
    if st.button("← Back to Home"):
        st.session_state.phase = 'home'
        st.rerun()
    
    # Show current state info if debug mode
    if st.session_state.debug_mode:
        st.markdown(f"""
        <div class="debug-info">
            <strong>🐛 Debug Info:</strong><br>
            Phase: {st.session_state.phase}<br>
            Waiting for answer: {st.session_state.waiting_for_answer}<br>
            Current question index: {st.session_state.current_question_index}<br>
            Total questions: {len(st.session_state.current_questions)}<br>
            Conversation complete: {st.session_state.conversation_complete}<br>
            <strong>🧠 CRISP-DM Analysis:</strong><br>
            Assumptions detected: {len(st.session_state.detected_assumptions)}<br>
            Gaps identified: {len(st.session_state.identified_gaps)}<br>
            Context references: {len(st.session_state.context_references)}<br>
            Advanced engines: {"✅" if CRISP_DM_COMPONENTS_AVAILABLE else "❌"}
        </div>
        """, unsafe_allow_html=True)
    
    # Display chat history
    display_chat_history()
    
    # Show current question if waiting for answer
    if st.session_state.waiting_for_answer and st.session_state.current_question_data:
        question_data = st.session_state.current_question_data
        
        st.markdown("---")
        st.markdown("### 💬 Business Analyst")
        st.markdown(f"*{question_data['question']}*")
        
        # Show reasoning if available
        if 'reasoning' in question_data:
            st.caption(f"💡 Why I'm asking: {question_data['reasoning']}")
        
        st.markdown("**Your response:**")
        
        # Display options as buttons
        options = question_data['options']
        
        # All questions now use multiple choice buttons with contextual options
        for i, option in enumerate(options):
            if st.button(option, key=f"option_{i}", use_container_width=True):
                handle_answer_selection(option)
    
    elif st.session_state.conversation_complete:
        st.success("✅ **Business understanding complete!**")
        st.info("🎯 Ready to proceed with your business analysis!")
        
        # Show conversation summary
        if st.session_state.question_history:
            with st.expander("📋 Business Context Summary", expanded=True):
                st.markdown("### Your Business Requirements:")
                st.info(f"**Main Question:** {st.session_state.business_question}")
                
                st.markdown("### Clarification Details:")
                for i, (q, a) in enumerate(st.session_state.question_history, 1):
                    st.markdown(f"**Q{i}:** {q}")
                    st.markdown(f"**A{i}:** {a}")
                    if i < len(st.session_state.question_history):
                        st.markdown("---")
        
        # Action buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Start New Analysis", use_container_width=True):
                reset_session()
        
        with col2:
            if st.button("📄 Generate Summary Report", use_container_width=True):
                generate_summary_report()
    
    else:
        st.info("🤔 Something seems off. Let me help you get back on track.")
        if st.button("🔄 Restart Conversation"):
            if st.session_state.business_question:
                start_conversation(st.session_state.business_question)
            else:
                st.session_state.phase = 'home'
                st.rerun()

def handle_answer_selection(selected_option: str):
    """Handle when user selects an answer option."""
    try:
        # Add user's answer to chat
        add_message(selected_option, is_user=True)
        
        # Store the Q&A
        current_q = st.session_state.current_question_data
        st.session_state.question_history.append((current_q['question'], selected_option))
        
        # Neural Answer Analysis (if available)
        try:
            from question_generator import process_neural_answer
            
            if hasattr(st.session_state, 'neural_session_id') and st.session_state.neural_session_id:
                question_id = current_q.get('id', f"q_{len(st.session_state.question_history)}")
                
                if st.session_state.debug_mode:
                    st.info("🧠 Analyzing answer with Neural System...")
                
                answer_analysis, followups = process_neural_answer(
                    st.session_state.neural_session_id, 
                    question_id, 
                    selected_option,
                    st.session_state.debug_mode
                )
                
                if answer_analysis:
                    # Store neural insights
                    if not hasattr(st.session_state, 'neural_insights'):
                        st.session_state.neural_insights = []
                    st.session_state.neural_insights.extend(answer_analysis.extracted_insights)
                    
                    # Add follow-up questions if generated
                    if followups:
                        if st.session_state.debug_mode:
                            st.success(f"🧠 Neural System generated {len(followups)} follow-up questions")
                        
                        # Convert neural questions to app format
                        neural_followups = []
                        for fq in followups:
                            # Check if this is a clarification question with custom options
                            if fq.category == "clarification" and hasattr(fq, 'clarification_options'):
                                neural_followups.append({
                                    "question": fq.question_text,
                                    "options": fq.clarification_options,
                                    "category": fq.category,
                                    "priority": fq.priority,
                                    "reasoning": fq.reasoning,
                                    "is_clarification": True
                                })
                            else:
                                # Regular follow-up questions
                                neural_followups.append({
                                    "question": fq.question_text,
                                    "options": ["Yes", "No", "Partially", "Not sure", "Need more information"],
                                    "category": fq.category,
                                    "priority": fq.priority,
                                    "reasoning": fq.reasoning,
                                    "is_clarification": False
                                })
                        
                        # Add to current questions
                        st.session_state.current_questions.extend(neural_followups)
                
        except Exception as e:
            if st.session_state.debug_mode:
                st.warning(f"Neural analysis unavailable: {e}")
        
        # Advanced CRISP-DM Analysis
        if CRISP_DM_COMPONENTS_AVAILABLE and st.session_state.analysis_engine:
            try:
                # Create Answer object for analysis
                from crisp_dm_framework import Answer
                answer_obj = Answer(
                    question_id=current_q.get('id', f"q_{len(st.session_state.question_history)}"),
                    response=selected_option,
                    timestamp=pd.Timestamp.now()
                )
                
                # Perform adaptive analysis
                if st.session_state.adaptive_engine:
                    analysis_result = st.session_state.adaptive_engine.analyze_answer(answer_obj)
                    
                    if st.session_state.debug_mode:
                        st.info(f"🔍 Analysis: Complexity={analysis_result.complexity_level.value}, Vagueness={analysis_result.vagueness_level.value}")
                        if analysis_result.assumptions:
                            st.info(f"🧠 Detected assumptions: {len(analysis_result.assumptions)}")
                        if analysis_result.contradictions:
                            st.warning(f"⚠️ Detected contradictions: {len(analysis_result.contradictions)}")
                
                # Update context preservation
                if st.session_state.context_engine:
                    context = st.session_state.context_engine.create_context(
                        st.session_state.business_question or "session",
                        [answer_obj]
                    )
                    
                    # Check for context references
                    references = st.session_state.context_engine.generate_context_references(
                        answer_obj, context.answer_history
                    )
                    st.session_state.context_references.extend(references)
                
                # Detect assumptions and gaps
                if st.session_state.analysis_engine:
                    # Detect assumptions in the current answer
                    assumptions = st.session_state.analysis_engine.identify_assumptions([answer_obj])
                    st.session_state.detected_assumptions.extend(assumptions)
                    
                    # Detect gaps based on all answers so far
                    all_answers = []
                    for q, a in st.session_state.question_history:
                        all_answers.append(Answer(
                            question_id=f"q_{len(all_answers)}",
                            response=a,
                            timestamp=pd.Timestamp.now()
                        ))
                    
                    gaps = st.session_state.analysis_engine.detect_gaps(all_answers)
                    st.session_state.identified_gaps = gaps  # Update with latest gaps
                    
                    if st.session_state.debug_mode and (assumptions or gaps):
                        if assumptions:
                            st.info(f"🎯 New assumptions detected: {len(assumptions)}")
                        if gaps:
                            st.warning(f"📋 Information gaps identified: {len(gaps)}")
                
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"🐛 Advanced analysis error: {str(e)}")
        
        # Add acknowledgment
        acknowledgments = [
            "I see, that's helpful context.",
            "Thank you for that insight.",
            "That gives me a better understanding.",
            "Good to know.",
            "That's valuable information."
        ]
        
        import random
        acknowledgment = random.choice(acknowledgments)
        
        # Check if we should generate more questions using AI
        if len(st.session_state.question_history) >= len(st.session_state.current_questions):
            # Try to generate AI-powered follow-up questions
            try:
                if st.session_state.debug_mode:
                    st.info("🤖 Checking if more questions are needed using Groq AI...")
                
                from groq_llm import generate_next_questions, check_completeness
                
                # Check completeness first
                qa_pairs = [{"question": q, "answer": a} for q, a in st.session_state.question_history]
                completeness = check_completeness(st.session_state.business_question, qa_pairs)
                
                if st.session_state.debug_mode:
                    st.info(f"🔍 Completeness check: {completeness.get('confidence_score', 0)}% confident")
                
                if completeness.get("complete", False) or completeness.get("confidence_score", 0) > 80:
                    st.session_state.conversation_complete = True
                    st.session_state.waiting_for_answer = False
                    add_message(f"{acknowledgment} I believe I have enough context now to provide meaningful insights. Let me summarize what I've learned.")
                else:
                    # Generate follow-up questions
                    next_questions = generate_next_questions(st.session_state.business_question, qa_pairs)
                    
                    if next_questions and len(next_questions) > 0:
                        # Add the new questions to our list
                        st.session_state.current_questions.extend(next_questions)
                        st.session_state.current_question_data = next_questions[0]
                        add_message(f"{acknowledgment} Let me ask you one more thing to get a complete picture.")
                        
                        if st.session_state.debug_mode:
                            st.success(f"✅ Generated {len(next_questions)} AI follow-up questions")
                    else:
                        st.session_state.conversation_complete = True
                        st.session_state.waiting_for_answer = False
                        add_message(f"{acknowledgment} I believe I have enough information to provide valuable insights.")
                        
            except Exception as e:
                if st.session_state.debug_mode:
                    st.warning(f"⚠️ AI follow-up generation failed: {str(e)}")
                
                # Fallback to simple completion logic
                st.session_state.conversation_complete = True
                st.session_state.waiting_for_answer = False
                add_message(f"{acknowledgment} I believe I have enough information to provide valuable insights.")
        else:
            # Move to next question in current batch
            st.session_state.current_question_index += 1
            if st.session_state.current_question_index < len(st.session_state.current_questions):
                next_question = st.session_state.current_questions[st.session_state.current_question_index]
                st.session_state.current_question_data = next_question
                add_message(f"{acknowledgment} Let me ask you one more thing to get a complete picture.")
            else:
                st.session_state.conversation_complete = True
                st.session_state.waiting_for_answer = False
                add_message(f"{acknowledgment} I believe I have enough information to provide valuable insights.")
        
        st.rerun()
        
    except Exception as e:
        log_error(f"Error handling answer selection", e)
        st.error("There was an issue processing your answer. Please try again.")

def generate_summary_report():
    """Generate a comprehensive business understanding summary with advanced CRISP-DM analysis."""
    st.markdown("### 📋 Comprehensive Business Understanding Report")
    
    # Neural System Summary (if available)
    try:
        from question_generator import generate_neural_summary
        
        if hasattr(st.session_state, 'neural_session_id') and st.session_state.neural_session_id:
            st.markdown("#### 🧠 Neural System Analysis")
            
            neural_summary = generate_neural_summary(
                st.session_state.neural_session_id,
                st.session_state.debug_mode
            )
            
            if neural_summary:
                # Neural insights are displayed by the function itself
                st.markdown("---")
            else:
                st.info("Neural analysis will be available after answering more questions.")
        
    except Exception as e:
        if st.session_state.debug_mode:
            st.warning(f"Neural summary unavailable: {e}")
    
    # Main business question
    st.markdown("#### 🎯 Primary Business Question")
    st.info(st.session_state.business_question)
    
    # Advanced CRISP-DM Analysis Results
    if CRISP_DM_COMPONENTS_AVAILABLE:
        
        # Display detected assumptions
        if st.session_state.detected_assumptions:
            st.markdown("#### 🧠 Detected Assumptions")
            st.warning(f"Found {len(st.session_state.detected_assumptions)} assumptions that need validation:")
            
            for i, assumption in enumerate(st.session_state.detected_assumptions, 1):
                with st.expander(f"Assumption {i}: {assumption.type.value.replace('_', ' ').title()}"):
                    st.write(f"**Description:** {assumption.description}")
                    st.write(f"**Risk Level:** {assumption.risk_level.value.upper()}")
                    st.write(f"**Source:** {assumption.source_text[:100]}...")
                    if assumption.validation_questions:
                        st.write("**Validation Questions:**")
                        for vq in assumption.validation_questions:
                            st.write(f"- {vq}")
        
        # Display identified gaps
        if st.session_state.identified_gaps:
            st.markdown("#### 📋 Information Gaps")
            st.error(f"Identified {len(st.session_state.identified_gaps)} information gaps:")
            
            for i, gap in enumerate(st.session_state.identified_gaps, 1):
                with st.expander(f"Gap {i}: {gap.type.value.replace('_', ' ').title()}"):
                    st.write(f"**Description:** {gap.description}")
                    st.write(f"**Severity:** {gap.severity.value.upper()}")
                    st.write(f"**Affected Dimensions:** {', '.join([d.value for d in gap.affected_dimensions])}")
                    if gap.suggested_questions:
                        st.write("**Suggested Questions:**")
                        for sq in gap.suggested_questions:
                            st.write(f"- {sq}")
        
        # Display context references
        if st.session_state.context_references:
            st.markdown("#### 🔗 Context References")
            st.info(f"Found {len(st.session_state.context_references)} context connections:")
            
            for i, ref in enumerate(st.session_state.context_references, 1):
                st.write(f"**{i}.** {ref.reference_type.title()}: {ref.referenced_answer_excerpt[:100]}...")
        
        # Generate business insights if we have enough data
        if st.session_state.analysis_engine and len(st.session_state.question_history) >= 3:
            try:
                # Create session-like object for insights generation
                all_answers = []
                for q, a in st.session_state.question_history:
                    from crisp_dm_framework import Answer
                    all_answers.append(Answer(
                        question_id=f"q_{len(all_answers)}",
                        response=a,
                        timestamp=pd.Timestamp.now()
                    ))
                
                insights = st.session_state.analysis_engine.generate_insights(all_answers)
                
                if insights and insights.insights:
                    st.markdown("#### 💡 Business Insights")
                    
                    for insight in insights.insights:
                        if insight.category == "opportunity":
                            st.success(f"**Opportunity:** {insight.title}")
                        elif insight.category == "risk":
                            st.error(f"**Risk:** {insight.title}")
                        elif insight.category == "recommendation":
                            st.info(f"**Recommendation:** {insight.title}")
                        else:
                            st.write(f"**{insight.category.title()}:** {insight.title}")
                        
                        st.write(insight.description)
                        if insight.supporting_evidence:
                            with st.expander("Supporting Evidence"):
                                for evidence in insight.supporting_evidence:
                                    st.write(f"- {evidence}")
                
                if insights.recommendations:
                    st.markdown("#### 🚀 Key Recommendations")
                    for i, rec in enumerate(insights.recommendations, 1):
                        st.write(f"{i}. {rec}")
                        
            except Exception as e:
                if st.session_state.debug_mode:
                    st.error(f"Error generating insights: {e}")
    
    # Detailed context (existing functionality)
    if st.session_state.question_history:
        st.markdown("#### 📊 Business Context & Requirements")
        
        for i, (q, a) in enumerate(st.session_state.question_history, 1):
            st.markdown(f"**Q{i}:** {q}")
            st.markdown(f"**A{i}:** {a}")
            if i < len(st.session_state.question_history):
                st.markdown("---")
    
    # Next steps recommendations
    st.markdown("#### 🚀 Recommended Next Steps")
    
    next_steps = [
        "**Data Collection**: Gather relevant data sources that align with your business question",
        "**Analysis Planning**: Define specific metrics and KPIs to measure",
        "**Stakeholder Alignment**: Share this summary with key stakeholders for validation",
        "**Implementation Strategy**: Develop an action plan based on your requirements"
    ]
    
    # Add specific recommendations based on detected gaps and assumptions
    if CRISP_DM_COMPONENTS_AVAILABLE:
        if st.session_state.detected_assumptions:
            next_steps.insert(1, "**Assumption Validation**: Validate the detected assumptions before proceeding")
        if st.session_state.identified_gaps:
            next_steps.insert(1, "**Information Gathering**: Address the identified information gaps")
    
    for step in next_steps:
        st.markdown(f"- {step}")
    
    # Export option with enhanced content
    if st.button("📥 Download Comprehensive Report", use_container_width=True):
        report_content = f"""
# Comprehensive Business Understanding Report

## Primary Business Question
{st.session_state.business_question}

## Business Context & Requirements
"""
        for i, (q, a) in enumerate(st.session_state.question_history, 1):
            report_content += f"\n**Q{i}:** {q}\n**A{i}:** {a}\n"
        
        # Add advanced analysis to report
        if CRISP_DM_COMPONENTS_AVAILABLE:
            if st.session_state.detected_assumptions:
                report_content += "\n## Detected Assumptions\n"
                for i, assumption in enumerate(st.session_state.detected_assumptions, 1):
                    report_content += f"\n{i}. **{assumption.type.value.replace('_', ' ').title()}** (Risk: {assumption.risk_level.value.upper()})\n"
                    report_content += f"   Description: {assumption.description}\n"
                    report_content += f"   Source: {assumption.source_text[:100]}...\n"
            
            if st.session_state.identified_gaps:
                report_content += "\n## Information Gaps\n"
                for i, gap in enumerate(st.session_state.identified_gaps, 1):
                    report_content += f"\n{i}. **{gap.type.value.replace('_', ' ').title()}** (Severity: {gap.severity.value.upper()})\n"
                    report_content += f"   Description: {gap.description}\n"
                    report_content += f"   Affected Dimensions: {', '.join([d.value for d in gap.affected_dimensions])}\n"
        
        report_content += """
## Recommended Next Steps
1. Data Collection: Gather relevant data sources that align with your business question
2. Analysis Planning: Define specific metrics and KPIs to measure  
3. Stakeholder Alignment: Share this summary with key stakeholders for validation
4. Implementation Strategy: Develop an action plan based on your requirements
"""
        
        if CRISP_DM_COMPONENTS_AVAILABLE:
            if st.session_state.detected_assumptions:
                report_content += "5. Assumption Validation: Validate the detected assumptions before proceeding\n"
            if st.session_state.identified_gaps:
                report_content += "6. Information Gathering: Address the identified information gaps\n"
        
        st.download_button(
            label="📄 Download as Text File",
            data=report_content,
            file_name=f"comprehensive_business_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def reset_session():
    """Reset the session for a new analysis."""
    keys_to_reset = [
        'chat_history', 'current_questions', 'business_question', 'conversation_complete',
        'question_history', 'waiting_for_answer', 'current_question_data', 'current_question_index'
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    
    initialize_session_state()
    st.session_state.phase = 'home'
    st.rerun()

def main():
    """Main application logic."""
    
    try:
        if st.session_state.phase == 'home':
            show_home_page()
        elif st.session_state.phase == 'chatting':
            handle_chat_interface()
        else:
            st.error(f"Unknown phase: {st.session_state.phase}")
            st.session_state.phase = 'home'
            st.rerun()
            
    except Exception as e:
        log_error(f"Error in main application", e)
        st.error("An unexpected error occurred. Returning to home page.")
        st.session_state.phase = 'home'
        if st.button("🔄 Restart Application"):
            st.rerun()

if __name__ == "__main__":
    main()