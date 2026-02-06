"""
CRISP-DM Business Understanding Specialist
Beautiful UI with data upload + intelligent clarification questions

Features:
- Predefined business challenges
- Custom business question input
- CSV data upload
- Analyzes BOTH question AND data
- Asks intelligent clarification questions
- Knowledge Graph + Graph RAG enhanced
"""

import streamlit as st
import pandas as pd
import os
import tempfile
import datetime
from clarification_with_graph_rag import GraphEnhancedClarificationAgent

# Page config
st.set_page_config(
    page_title="CRISP-DM Business Understanding",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS matching the image style with dark background
st.markdown("""
<style>
    /* Dark background */
    .main {
        background-color: #0a0e27;
        color: #ffffff;
    }
    
    .stApp {
        background-color: #0a0e27;
    }
    
    /* Header */
    .header-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.1rem;
    }
    
    /* Mode badge */
    .mode-badge {
        background: rgba(76, 175, 80, 0.2);
        color: #4CAF50;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        display: inline-block;
        margin: 1rem 0;
        font-weight: bold;
    }
    
    /* Challenge buttons */
    .stButton>button {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        transition: all 0.3s;
        width: 100%;
        text-align: left;
        font-size: 0.95rem;
    }
    
    .stButton>button:hover {
        background: rgba(102, 126, 234, 0.15);
        border-color: rgba(102, 126, 234, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:active {
        background: rgba(102, 126, 234, 0.25);
        transform: translateY(0);
    }
    
    /* Text inputs and text areas */
    .stTextArea textarea, .stTextInput input {
        background-color: #1a1f3a;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
    
    .stTextArea textarea:focus, .stTextInput input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 1px #667eea;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    
    /* User message banner */
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Assistant message box */
    .assistant-message {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* File uploader */
    .uploadedFile {
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        color: white;
    }
    
    /* Dataframe */
    .stDataFrame {
        background-color: #1a1f3a;
    }
    
    /* Metrics */
    .stMetric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 8px;
    }
    
    /* Markdown text */
    .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Divider */
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    /* Success/Error messages */
    .stSuccess {
        background-color: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.3);
        color: #4CAF50;
    }
    
    .stError {
        background-color: rgba(244, 67, 54, 0.1);
        border: 1px solid rgba(244, 67, 54, 0.3);
        color: #f44336;
    }
    
    /* Download button */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = False
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'csv_path' not in st.session_state:
    st.session_state.csv_path = None
if 'business_question' not in st.session_state:
    st.session_state.business_question = ""

# Header
st.markdown("""
<div class="header-container">
    <div class="main-title">🧠 CRISP-DM Business Understanding Specialist</div>
    <div class="subtitle">Comprehensive Business Analysis Engine</div>
    <div class="mode-badge">
        🚀 Full CRISP-DM Mode: AI-powered personalized questions + Advanced business analysis 
        (Gap identification, Assumption detection, Context preservation)
    </div>
    <div style="margin-top: 0.5rem; font-size: 0.9rem; color: rgba(255,255,255,0.7);">
        🤖 Powered by: """ + os.getenv("LLM_PROVIDER", "groq").upper() + """ | 🧠 Knowledge Graph + Graph RAG
    </div>
</div>
""", unsafe_allow_html=True)

# Main interface
if not st.session_state.conversation_started:
    
    # Predefined business challenges
    st.markdown("### 💼 What business challenge can I help you with today?")
    st.markdown("*I'll use AI to generate personalized strategic questions specifically tailored to your business challenge, with advanced CRISP-DM analysis including assumption detection, gap identification, and context preservation.*")
    
    # Challenge buttons in grid
    col1, col2, col3 = st.columns(3)
    
    challenges = [
        ("📊", "Customer Value Analysis"),
        ("⚠️", "Churn Risk Assessment"),
        ("🎯", "Customer Segmentation"),
        ("📈", "Revenue Growth Analysis"),
        ("🎨", "Product Strategy"),
        ("📊", "Performance Review")
    ]
    
    for idx, (icon, challenge) in enumerate(challenges):
        col = [col1, col2, col3][idx % 3]
        with col:
            if st.button(f"{icon} {challenge}", key=f"challenge_{idx}", use_container_width=True):
                st.session_state.business_question = f"Help me with {challenge}"
                st.rerun()
    
    st.markdown("---")
    
    # Custom question input
    st.markdown("### 💬 Or ask me anything about your business:")
    st.markdown("*What's your business question?*")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        custom_question = st.text_area(
            "What's your business question?",
            value=st.session_state.business_question,
            placeholder="e.g., How can we improve customer retention in our subscription business?",
            height=100,
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("**📊 Upload Your Data**")
        st.markdown("*(Optional but recommended)*")
        
        uploaded_file = st.file_uploader(
            "CSV, Excel, or JSON",
            type=['csv', 'xlsx', 'xls', 'json'],
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            # Save file
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, mode='wb') as tmp:
                tmp.write(uploaded_file.getvalue())
                st.session_state.csv_path = tmp.name
            
            # Preview data
            try:
                if suffix == '.csv':
                    df = pd.read_csv(st.session_state.csv_path)
                elif suffix in ['.xlsx', '.xls']:
                    df = pd.read_excel(st.session_state.csv_path)
                elif suffix == '.json':
                    df = pd.read_json(st.session_state.csv_path)
                
                st.success(f"✅ {len(df)} rows × {len(df.columns)} columns")
                
                with st.expander("👀 Preview Data"):
                    st.dataframe(df.head(10), use_container_width=True)
                    
                    # Show data summary
                    st.markdown("**📊 Data Summary:**")
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Rows", f"{len(df):,}")
                    with col_b:
                        st.metric("Columns", len(df.columns))
                    with col_c:
                        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                        st.metric("Numeric", len(numeric_cols))
                    
                    st.markdown("**Columns:**")
                    st.write(", ".join(df.columns.tolist()))
            
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    # Start button
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Start Business Analysis", type="primary", use_container_width=True):
        question = custom_question or st.session_state.business_question
        
        if not question:
            st.error("❌ Please enter a business question or select a challenge")
        else:
            with st.spinner("🧠 Initializing CRISP-DM Analysis Engine with Graph RAG..."):
                # Initialize agent
                st.session_state.agent = GraphEnhancedClarificationAgent()
                
                # Start conversation
                response = st.session_state.agent.start_conversation(
                    business_question=question,
                    csv_path=st.session_state.csv_path
                )
                
                # Add to messages
                st.session_state.messages.append({
                    "role": "user",
                    "content": f"**Business Question:** {question}" + 
                              (f"\n**Data Uploaded:** {uploaded_file.name} ({len(df)} rows × {len(df.columns)} columns)" if uploaded_file else "")
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                
                st.session_state.conversation_started = True
                st.success("✅ Analysis started! The system is analyzing both your question and data.")
                st.rerun()

else:
    # Conversation interface
    st.markdown("### 💬 Business Understanding Conversation")
    
    # Display conversation history
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            # User message - show as blue banner
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1rem; border-radius: 10px; margin: 1rem 0;
                        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);">
                <strong>You:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Assistant message - show with dark theme
            content = msg['content']
            
            st.markdown(f"""
            <div style="background: rgba(255,255,255,0.05); 
                        border: 1px solid rgba(255,255,255,0.1);
                        padding: 1.5rem; border-radius: 10px; margin: 1rem 0;">
                <div style="color: rgba(255,255,255,0.7); font-size: 0.9rem; margin-bottom: 0.5rem;">
                    💬 <strong>Business Analyst</strong>
                </div>
                <div style="color: rgba(255,255,255,0.95);">
                    {content}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # If this is the latest assistant message, show interactive options
            if idx == len(st.session_state.messages) - 1 and msg["role"] == "assistant":
                st.markdown("---")
                
                # Try to extract options from the message
                # Look for bullet points or numbered lists
                lines = content.split('\n')
                options = []
                
                for line in lines:
                    line = line.strip()
                    # Check for bullet points or dashes (more flexible)
                    if line.startswith('- ') or line.startswith('• ') or line.startswith('* '):
                        option_text = line[2:].strip()
                        # Valid option: not empty, reasonable length, not a heading
                        if option_text and 5 < len(option_text) < 200 and not option_text.endswith(':'):
                            options.append(option_text)
                
                # If we found options, show them as buttons
                if len(options) >= 3:
                    st.markdown("### 🎯 Select your answer:")
                    
                    # Show option buttons in a grid with better styling
                    cols_per_row = 2
                    for i in range(0, len(options), cols_per_row):
                        cols = st.columns(cols_per_row)
                        for j, col in enumerate(cols):
                            if i + j < len(options):
                                option = options[i + j]
                                with col:
                                    # Create button with emoji for visual appeal
                                    button_label = f"✓ {option}" if len(option) < 80 else f"✓ {option[:77]}..."
                                    if st.button(
                                        button_label,
                                        key=f"option_{idx}_{i+j}",
                                        use_container_width=True,
                                        help=option  # Show full text on hover
                                    ):
                                        # User selected this option
                                        st.session_state.messages.append({
                                            "role": "user",
                                            "content": f"✓ Selected: {option}"
                                        })
                                        
                                        # Get next question
                                        with st.spinner("🧠 Processing your selection..."):
                                            response = st.session_state.agent.answer_questions(option)
                                        
                                        st.session_state.messages.append({
                                            "role": "assistant",
                                            "content": response
                                        })
                                        
                                        st.rerun()
                    
                    st.markdown("---")
                    st.markdown("### ✍️ Or provide your own answer:")
                
                # Always show text area for custom response
                user_input = st.text_area(
                    "Type your detailed answer:",
                    height=100,
                    placeholder="Type your answer here or select an option above...",
                    key=f"input_{idx}"
                )
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    if st.button("📤 Send Answer", type="primary", use_container_width=True):
                        if user_input:
                            st.session_state.messages.append({
                                "role": "user",
                                "content": user_input
                            })
                            
                            with st.spinner("🧠 Analyzing your response..."):
                                response = st.session_state.agent.answer_questions(user_input)
                            
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })
                            
                            st.rerun()
                
                with col2:
                    if st.button("📋 Get Final Summary", use_container_width=True):
                        with st.spinner("📋 Generating Business Understanding document..."):
                            summary = st.session_state.agent.get_final_summary()
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": "# 📋 Business Understanding Summary\n\n" + summary
                        })
                        
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        st.download_button(
                            "📥 Download Summary",
                            summary,
                            f"business_understanding_{timestamp}.md",
                            mime="text/markdown"
                        )
                        
                        st.success("✅ Complete! Knowledge graph updated.")
                        st.rerun()
    
    # New session button at bottom
    st.markdown("---")
    if st.button("🔄 Start New Analysis", use_container_width=True):
        st.session_state.agent = None
        st.session_state.conversation_started = False
        st.session_state.messages = []
        st.session_state.csv_path = None
        st.session_state.business_question = ""
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: rgba(255,255,255,0.7); padding: 1rem;">
    <p>🧠 CRISP-DM Business Understanding Specialist</p>
    <p style="font-size: 0.9rem;">
        Powered by Knowledge Graph + Graph RAG • Learns from conversations • Gets smarter over time
    </p>
</div>
""", unsafe_allow_html=True)
