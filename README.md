# CRISP-DM Business Understanding Specialist

A comprehensive business analysis application following the Cross-Industry Standard Process for Data Mining (CRISP-DM) methodology with **AI-powered personalized questioning**.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up your Groq API key in .env file
GROQ_API_KEY=your_groq_api_key_here

# Run the application
streamlit run app.py
```

Access the application at: http://localhost:8501

## 🤖 AI-Powered Personalized Questions

Unlike generic business analysis tools, this application uses **Groq AI** to generate personalized clarification questions that are specifically tailored to your unique business challenge:

- **Customer Analysis**: "How can we reduce customer churn?" → Gets questions about retention metrics, customer segments, and churn drivers
- **Revenue Analysis**: "What's driving revenue decline?" → Gets questions about market factors, competition, and internal dynamics  
- **Product Strategy**: "Which products to focus on?" → Gets questions about profitability drivers, market expansion, and product lifecycles
- **Marketing Optimization**: "How to improve campaigns?" → Gets questions about KPIs, target audiences, and campaign performance

## 📋 Advanced CRISP-DM Features

### 🧠 **Assumption Detection & Validation**
- **Automatic Detection**: AI identifies implicit beliefs and unstated requirements in your responses
- **Risk Assessment**: Each assumption is categorized by type and risk level (Low/Medium/High/Critical)
- **Validation Questions**: System generates specific questions to validate critical assumptions
- **Types Detected**: Implicit beliefs, stakeholder assumptions, resource assumptions, timeline assumptions

### 📊 **Gap Identification & Highlighting**
- **Information Gap Detection**: Identifies missing critical business information
- **Severity Assessment**: Gaps are prioritized by severity and impact on analysis
- **Suggested Questions**: Provides specific questions to fill identified gaps
- **Gap Types**: Missing stakeholders, unclear objectives, undefined constraints, missing success metrics

### 🔗 **Context Preservation & Referencing**
- **Answer History Tracking**: Maintains complete conversation context across all questions
- **Conversational Continuity**: References previous answers to show active listening
- **Conflict Detection**: Identifies contradictions between different responses
- **Context Citations**: Shows how current questions build on previous answers

### ⚡ **Adaptive Logic Engine**
- **Answer Analysis**: Evaluates complexity, vagueness, and contradictions in real-time
- **Dynamic Follow-ups**: Generates targeted follow-up questions based on response analysis
- **Domain Adaptation**: Adjusts questioning approach based on detected business domain
- **Intelligent Completion**: AI determines when sufficient context has been gathered

### 🛡️ **Error Handling & Recovery**
- **Input Validation**: Validates business questions and answers for completeness
- **Session Recovery**: Automatic recovery from browser refresh or connection issues
- **Graceful Degradation**: Falls back to intelligent questions when AI services are unavailable
- **Data Consistency**: Validates and repairs session data integrity

## 🏗️ Project Structure

### Core Application
- `app.py` - Main Streamlit application
- `requirements.txt` - Python dependencies
- `.env` - Environment variables (API keys)

### CRISP-DM Components
- `crisp_dm_framework.py` - 8-dimension question framework
- `session_manager.py` - Session state and progress tracking
- `adaptive_engine.py` - Intelligent question adaptation
- `analysis_engine.py` - Business insight generation
- `summary_generator.py` - Report generation

### AI Integration
- `groq_llm.py` - Groq API integration for dynamic questions
- `demo_llm.py` - Fallback demo questions
- `enhanced_ai_integration.py` - Advanced AI features

### UI & UX
- `advanced_ui_components.py` - Professional interface components
- `explanation_system.py` - Question reasoning and explanations
- `context_preservation.py` - Conversation continuity

### Support Systems
- `error_handling.py` - Error management and recovery
- `export_system.py` - Document export capabilities
- `integration_system.py` - System integration and monitoring
- `data_handler.py` - Data processing utilities

## 🎯 How to Use

1. **Start Analysis** - Choose a business scenario or enter your own question
2. **Get Personalized Questions** - AI generates questions specifically tailored to your business challenge
3. **Answer Strategic Questions** - Respond to AI-powered follow-ups with real-time analysis
4. **Advanced Analysis** - System automatically detects assumptions, identifies gaps, and preserves context
5. **AI Completeness Check** - System intelligently determines when enough context is gathered
6. **Comprehensive Report** - Get detailed business understanding with assumptions, gaps, and insights
7. **Export Results** - Download your complete analysis for future reference

## 🔍 What You'll See in Debug Mode

- **Real-time Analysis**: See complexity and vagueness levels of your responses
- **Assumption Detection**: Watch as the system identifies implicit beliefs in your answers
- **Gap Identification**: See information gaps being detected and prioritized
- **Context References**: Observe how the system maintains conversational continuity
- **AI Decision Making**: Understand how the system determines next questions

## 🔧 Configuration

- **Required**: Set `GROQ_API_KEY` in `.env` file for AI-powered personalized questions
- **Fallback**: Application automatically uses intelligent backup questions if API unavailable
- **Debug Mode**: Enable in sidebar to see AI interactions and decision-making process

## 📊 Business Dimensions Covered

1. **Problem Definition** - Core problems and scope
2. **Business Objectives** - Goals and success criteria  
3. **Stakeholders** - Key people and decision makers
4. **Current Situation** - Baseline and existing approaches
5. **Constraints** - Budget, timeline, and limitations
6. **Success Criteria** - Measurement and thresholds
7. **Business Domain** - Industry context and regulations
8. **Implementation** - Integration and change management

## 🎉 Ready to Analyze

Your CRISP-DM Business Understanding Specialist is ready to help you thoroughly understand any business challenge before diving into data analysis!