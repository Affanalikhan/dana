"""
Example Conversation Templates
High-quality examples for customer interaction and segmentation
"""

def get_customer_interaction_conversation():
    """Example conversation for customer interaction optimization"""
    return {
        "business_question": "How to increase customer interaction?",
        "domain": "Marketing",
        "problem_type": "Customer engagement",
        "conversation_batches": [
            {
                "batch_number": 1,
                "questions": [
                    {
                        "question_id": "q1",
                        "question_text": "What type of business are you running?",
                        "dimension_category": "problem_definition",
                        "options": [
                            "E-commerce/Online retail",
                            "Physical retail store", 
                            "Service-based business (consulting, agency, etc.)",
                            "SaaS/Software product",
                            "Restaurant/Hospitality",
                            "Other"
                        ],
                        "reasoning": "Business type determines interaction channels and strategies"
                    },
                    {
                        "question_id": "q2", 
                        "question_text": "What is your primary customer interaction goal?",
                        "dimension_category": "business_objectives",
                        "options": [
                            "Increase purchase frequency",
                            "Build brand loyalty and community",
                            "Get more feedback and reviews",
                            "Drive engagement on social media",
                            "Improve customer support satisfaction",
                            "Increase time spent with your brand"
                        ],
                        "reasoning": "Clear objectives guide strategy selection"
                    },
                    {
                        "question_id": "q3",
                        "question_text": "Where are your customers currently most active?",
                        "dimension_category": "current_situation", 
                        "options": [
                            "Your website",
                            "Social media platforms",
                            "Email",
                            "In physical locations",
                            "Mobile app",
                            "We're not sure"
                        ],
                        "reasoning": "Understanding current touchpoints is essential"
                    },
                    {
                        "question_id": "q4",
                        "question_text": "What's your current level of customer interaction?",
                        "dimension_category": "current_situation",
                        "options": [
                            "Very low (customers only interact during purchase)",
                            "Low (occasional engagement)",
                            "Moderate (regular but could be better)",
                            "High (strong engagement already)"
                        ],
                        "reasoning": "Baseline measurement for improvement tracking"
                    },
                    {
                        "question_id": "q5",
                        "question_text": "What resources can you allocate to this initiative?",
                        "dimension_category": "constraints",
                        "options": [
                            "Limited budget, mostly time/effort",
                            "Moderate budget ($1,000-$10,000)",
                            "Significant budget (>$10,000)",
                            "We have a dedicated team",
                            "Just me/small team with limited time"
                        ],
                        "reasoning": "Resource constraints shape feasible strategies"
                    }
                ],
                "user_responses": [
                    {"question_id": "q1", "selected_option": "E-commerce/Online retail", "triggers_followup": True},
                    {"question_id": "q2", "selected_option": "Build brand loyalty and community", "triggers_followup": True},
                    {"question_id": "q3", "selected_option": "Social media platforms", "triggers_followup": True},
                    {"question_id": "q4", "selected_option": "Low (occasional engagement)", "triggers_followup": True},
                    {"question_id": "q5", "selected_option": "Moderate budget ($1,000-$10,000)", "triggers_followup": False}
                ]
            },
            {
                "batch_number": 2,
                "questions": [
                    {
                        "question_id": "q6",
                        "question_text": "What's your current customer communication frequency?",
                        "dimension_category": "current_situation",
                        "options": [
                            "Daily",
                            "Weekly", 
                            "Monthly",
                            "Only when they purchase",
                            "Rarely or never"
                        ],
                        "reasoning": "Communication frequency affects engagement levels"
                    },
                    {
                        "question_id": "q7",
                        "question_text": "What type of content resonates most with your audience?",
                        "dimension_category": "data_availability",
                        "options": [
                            "Educational/How-to content",
                            "Entertainment/Humor",
                            "Product updates and promotions",
                            "User-generated content and testimonials",
                            "Behind-the-scenes/Company culture",
                            "We haven't tested this yet"
                        ],
                        "reasoning": "Content strategy drives engagement quality"
                    },
                    {
                        "question_id": "q8",
                        "question_text": "What's the biggest barrier to customer interaction currently?",
                        "dimension_category": "problem_definition",
                        "options": [
                            "Customers don't know how to reach us",
                            "We don't provide value beyond our product/service",
                            "Our communication is too sales-focused",
                            "We're not present on the right channels",
                            "Slow response times",
                            "Lack of personalization"
                        ],
                        "reasoning": "Identifying barriers is crucial for solution design"
                    },
                    {
                        "question_id": "q9",
                        "question_text": "What customer segment are you targeting?",
                        "dimension_category": "stakeholders",
                        "options": [
                            "New customers/prospects",
                            "Recent buyers (first 3 months)",
                            "Regular customers",
                            "Lapsed customers (haven't purchased in 6+ months)",
                            "VIP/High-value customers",
                            "All segments equally"
                        ],
                        "reasoning": "Different segments require different interaction strategies"
                    },
                    {
                        "question_id": "q10",
                        "question_text": "What does success look like for you?",
                        "dimension_category": "success_criteria",
                        "options": [
                            "X% increase in social media engagement",
                            "More customer reviews and testimonials",
                            "Higher email open/click rates",
                            "Increased repeat purchase rate",
                            "More customer-initiated conversations",
                            "Better Net Promoter Score (NPS)"
                        ],
                        "reasoning": "Clear success metrics enable progress tracking"
                    }
                ],
                "user_responses": [
                    {"question_id": "q6", "selected_option": "Only when they purchase", "triggers_followup": True},
                    {"question_id": "q7", "selected_option": "We haven't tested this yet", "triggers_followup": True},
                    {"question_id": "q8", "selected_option": "We don't provide value beyond our product/service", "triggers_followup": True},
                    {"question_id": "q9", "selected_option": "All segments equally", "triggers_followup": True},
                    {"question_id": "q10", "selected_option": "Increased repeat purchase rate", "triggers_followup": False}
                ]
            }
        ],
        "dimensions_covered": ["problem_definition", "business_objectives", "stakeholders", "current_situation", "constraints", "success_criteria", "data_availability"],
        "total_questions": 10,
        "conversation_quality_score": 0.92
    }

def get_customer_segmentation_conversation():
    """Example conversation for customer segmentation"""
    return {
        "business_question": "How to improve customer segmentation?",
        "domain": "Retail",
        "problem_type": "Customer segmentation",
        "conversation_batches": [
            {
                "batch_number": 1,
                "questions": [
                    {
                        "question_id": "q1",
                        "question_text": "What is your primary goal for customer segmentation?",
                        "dimension_category": "business_objectives",
                        "options": [
                            "Personalize marketing messages",
                            "Identify high-value customers",
                            "Reduce churn/retain at-risk customers",
                            "Improve product recommendations",
                            "Optimize pricing strategies",
                            "Allocate resources more efficiently"
                        ],
                        "reasoning": "Clear objectives guide segmentation approach"
                    },
                    {
                        "question_id": "q2",
                        "question_text": "What stage is your business currently in?",
                        "dimension_category": "current_situation",
                        "options": [
                            "Startup (< 1 year, limited customer data)",
                            "Early growth (1-3 years, building customer base)",
                            "Established (3+ years, substantial customer data)",
                            "Mature (extensive historical data and customers)"
                        ],
                        "reasoning": "Business maturity affects available data and segmentation complexity"
                    },
                    {
                        "question_id": "q3",
                        "question_text": "How much customer data do you currently have access to?",
                        "dimension_category": "data_availability",
                        "options": [
                            "Basic contact information only (name, email)",
                            "Demographic data (age, location, gender, income)",
                            "Behavioral data (purchase history, website activity)",
                            "Psychographic data (interests, values, lifestyle)",
                            "Comprehensive data across all categories",
                            "Very limited or messy data"
                        ],
                        "reasoning": "Data availability determines segmentation sophistication"
                    },
                    {
                        "question_id": "q4",
                        "question_text": "What is your current customer base size?",
                        "dimension_category": "current_situation",
                        "options": [
                            "Less than 100 customers",
                            "100-1,000 customers",
                            "1,000-10,000 customers",
                            "10,000-100,000 customers",
                            "100,000+ customers"
                        ],
                        "reasoning": "Customer base size affects segmentation methods and statistical validity"
                    },
                    {
                        "question_id": "q5",
                        "question_text": "What tools and technology do you have available?",
                        "dimension_category": "constraints",
                        "options": [
                            "Spreadsheets only (Excel, Google Sheets)",
                            "Basic CRM (HubSpot, Salesforce, etc.)",
                            "Marketing automation platform",
                            "Advanced analytics tools (Tableau, Power BI)",
                            "AI/ML capabilities",
                            "Custom data infrastructure"
                        ],
                        "reasoning": "Technology constraints shape implementation approach"
                    }
                ],
                "user_responses": [
                    {"question_id": "q1", "selected_option": "Personalize marketing messages", "triggers_followup": True},
                    {"question_id": "q2", "selected_option": "Established (3+ years, substantial customer data)", "triggers_followup": True},
                    {"question_id": "q3", "selected_option": "Behavioral data (purchase history, website activity)", "triggers_followup": True},
                    {"question_id": "q4", "selected_option": "10,000-100,000 customers", "triggers_followup": True},
                    {"question_id": "q5", "selected_option": "Basic CRM (HubSpot, Salesforce, etc.)", "triggers_followup": False}
                ]
            },
            {
                "batch_number": 2,
                "questions": [
                    {
                        "question_id": "q6",
                        "question_text": "Which segmentation approach interests you most?",
                        "dimension_category": "problem_definition",
                        "options": [
                            "Demographic (age, gender, location, income)",
                            "Behavioral (purchase frequency, spending, engagement)",
                            "Psychographic (lifestyle, values, interests)",
                            "Geographic (location-based)",
                            "Technographic (device usage, tech adoption)",
                            "Customer lifecycle stage (new, active, at-risk, churned)",
                            "RFM (Recency, Frequency, Monetary value)",
                            "Not sure, need guidance"
                        ],
                        "reasoning": "Segmentation approach affects data requirements and outcomes"
                    },
                    {
                        "question_id": "q7",
                        "question_text": "What industry or business type are you in?",
                        "dimension_category": "problem_definition",
                        "options": [
                            "E-commerce/Retail",
                            "SaaS/Technology",
                            "Financial services",
                            "Healthcare",
                            "B2B services",
                            "Consumer packaged goods",
                            "Travel/Hospitality",
                            "Other"
                        ],
                        "reasoning": "Industry context influences relevant segmentation variables"
                    },
                    {
                        "question_id": "q8",
                        "question_text": "How will you use the segments once created?",
                        "dimension_category": "business_objectives",
                        "options": [
                            "Targeted email campaigns",
                            "Personalized website experiences",
                            "Customized product offerings",
                            "Sales team prioritization",
                            "Ad targeting and retargeting",
                            "Customer service approach",
                            "Multiple channels/purposes"
                        ],
                        "reasoning": "Usage determines required segment characteristics"
                    },
                    {
                        "question_id": "q9",
                        "question_text": "What's your biggest challenge with customers right now?",
                        "dimension_category": "problem_definition",
                        "options": [
                            "Low conversion rates",
                            "High churn/retention issues",
                            "One-size-fits-all messaging isn't working",
                            "Inefficient marketing spend",
                            "Don't understand customer needs",
                            "Can't identify best customers",
                            "Overwhelming data, no insights"
                        ],
                        "reasoning": "Current challenges guide segmentation priorities"
                    },
                    {
                        "question_id": "q10",
                        "question_text": "What level of segmentation complexity are you aiming for?",
                        "dimension_category": "success_criteria",
                        "options": [
                            "Simple (2-4 broad segments)",
                            "Moderate (5-8 segments)",
                            "Complex (9-15 segments)",
                            "Dynamic/micro-segmentation (continuous, personalized)",
                            "Whatever makes most sense for our situation"
                        ],
                        "reasoning": "Complexity level affects implementation and maintenance requirements"
                    }
                ],
                "user_responses": [
                    {"question_id": "q6", "selected_option": "Behavioral (purchase frequency, spending, engagement)", "triggers_followup": True},
                    {"question_id": "q7", "selected_option": "E-commerce/Retail", "triggers_followup": False},
                    {"question_id": "q8", "selected_option": "Multiple channels/purposes", "triggers_followup": True},
                    {"question_id": "q9", "selected_option": "One-size-fits-all messaging isn't working", "triggers_followup": True},
                    {"question_id": "q10", "selected_option": "Moderate (5-8 segments)", "triggers_followup": False}
                ]
            }
        ],
        "dimensions_covered": ["problem_definition", "business_objectives", "current_situation", "constraints", "success_criteria", "data_availability"],
        "total_questions": 10,
        "conversation_quality_score": 0.94
    }

# Additional domain-specific conversation templates
DOMAIN_CONVERSATION_TEMPLATES = {
    "SaaS/Technology": [
        {
            "business_question": "How to reduce customer churn in our SaaS product?",
            "problem_type": "Churn",
            "key_dimensions": ["problem_definition", "current_situation", "data_availability", "success_criteria"]
        },
        {
            "business_question": "What features should we prioritize for maximum user engagement?",
            "problem_type": "Feature prioritization", 
            "key_dimensions": ["business_objectives", "stakeholders", "constraints", "timeline_urgency"]
        }
    ],
    "Finance": [
        {
            "business_question": "How to improve fraud detection accuracy?",
            "problem_type": "Fraud detection",
            "key_dimensions": ["problem_definition", "data_availability", "success_criteria", "constraints"]
        },
        {
            "business_question": "What's the optimal portfolio allocation for risk management?",
            "problem_type": "Portfolio optimization",
            "key_dimensions": ["business_objectives", "current_situation", "constraints", "success_criteria"]
        }
    ],
    "Healthcare": [
        {
            "business_question": "How to improve patient outcomes for chronic conditions?",
            "problem_type": "Patient outcomes",
            "key_dimensions": ["problem_definition", "stakeholders", "current_situation", "success_criteria"]
        },
        {
            "business_question": "What's the optimal staffing allocation across departments?",
            "problem_type": "Resource allocation",
            "key_dimensions": ["business_objectives", "constraints", "data_availability", "timeline_urgency"]
        }
    ]
}