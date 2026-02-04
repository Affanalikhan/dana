"""
CRISP-DM Business Understanding Question Framework

This module implements the 8-dimension question structure for comprehensive
business understanding following the CRISP-DM methodology.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
import random


class QuestionType(Enum):
    """Types of questions in the framework."""
    CORE = "core"
    FOLLOWUP = "followup"
    CLARIFICATION = "clarification"


class Dimension(Enum):
    """The 8 dimensions of business understanding."""
    PROBLEM_DEFINITION = "problem_definition"
    BUSINESS_OBJECTIVES = "business_objectives"
    STAKEHOLDERS = "stakeholders"
    CURRENT_SITUATION = "current_situation"
    CONSTRAINTS = "constraints"
    SUCCESS_CRITERIA = "success_criteria"
    BUSINESS_DOMAIN = "business_domain"
    IMPLEMENTATION = "implementation"


@dataclass
class Question:
    """Represents a single question in the framework."""
    id: str
    dimension: Dimension
    text: str
    reasoning: str
    question_type: QuestionType = QuestionType.CORE
    dependencies: List[str] = field(default_factory=list)
    options: Optional[List[str]] = None


@dataclass
class QuestionBatch:
    """Represents a batch of questions to be delivered together."""
    batch_id: str
    dimension: Dimension
    questions: List[Question]
    context_bridge: str  # Explanation connecting to previous context
    
    def __post_init__(self):
        """Validate batch size constraints."""
        if not (1 <= len(self.questions) <= 7):
            raise ValueError(f"Question batch must contain 1-7 questions, got {len(self.questions)}")


@dataclass
class Answer:
    """Represents an answer to a question."""
    question_id: str
    response: str
    timestamp: Optional[str] = None
    confidence_level: Optional[str] = None
    requires_followup: bool = False


class CRISPDMFramework:
    """
    Main framework class that manages the 8-dimension question structure
    and provides question batching and progression logic.
    """
    
    def __init__(self):
        """Initialize the framework with question templates."""
        self.question_templates = self._initialize_question_templates()
        self.dimension_order = [
            Dimension.PROBLEM_DEFINITION,
            Dimension.BUSINESS_OBJECTIVES,
            Dimension.STAKEHOLDERS,
            Dimension.CURRENT_SITUATION,
            Dimension.CONSTRAINTS,
            Dimension.SUCCESS_CRITERIA,
            Dimension.BUSINESS_DOMAIN,
            Dimension.IMPLEMENTATION
        ]
    
    def _initialize_question_templates(self) -> Dict[Dimension, List[Question]]:
        """Initialize the question templates for each dimension."""
        templates = {}
        
        # Problem Definition Questions
        templates[Dimension.PROBLEM_DEFINITION] = [
            Question(
                id="pd_1",
                dimension=Dimension.PROBLEM_DEFINITION,
                text="What is the core business problem you're trying to solve?",
                reasoning="Understanding the fundamental problem helps scope the entire analysis",
                options=["Revenue decline", "Customer churn", "Operational inefficiency", "Market competition", "Other"]
            ),
            Question(
                id="pd_2", 
                dimension=Dimension.PROBLEM_DEFINITION,
                text="What triggered the need to address this problem now?",
                reasoning="Identifying triggers helps understand urgency and context",
                options=["Recent performance decline", "Strategic initiative", "Competitive pressure", "Regulatory requirement", "New opportunity"]
            ),
            Question(
                id="pd_3",
                dimension=Dimension.PROBLEM_DEFINITION,
                text="How does this problem align with your overall business strategy?",
                reasoning="Ensures the analysis supports broader strategic goals",
                options=["Critical strategic priority", "Important operational improvement", "Exploratory investigation", "Compliance requirement", "Innovation project"]
            ),
            Question(
                id="pd_4",
                dimension=Dimension.PROBLEM_DEFINITION,
                text="What is the scope of this problem within your organization?",
                reasoning="Defines boundaries and scale of the analysis needed",
                options=["Single department", "Multiple departments", "Entire organization", "External partnerships", "Industry-wide"]
            )
        ]
        
        # Business Objectives Questions
        templates[Dimension.BUSINESS_OBJECTIVES] = [
            Question(
                id="bo_1",
                dimension=Dimension.BUSINESS_OBJECTIVES,
                text="What specific, measurable outcomes do you want to achieve?",
                reasoning="Clear objectives guide analysis direction and success measurement",
                options=["Increase revenue by X%", "Reduce costs by X%", "Improve efficiency by X%", "Increase customer satisfaction", "Other quantifiable goal"]
            ),
            Question(
                id="bo_2",
                dimension=Dimension.BUSINESS_OBJECTIVES,
                text="What would constitute success for this initiative?",
                reasoning="Success criteria help define when objectives are met",
                options=["Specific numeric targets", "Qualitative improvements", "Process optimization", "Risk reduction", "Competitive advantage"]
            ),
            Question(
                id="bo_3",
                dimension=Dimension.BUSINESS_OBJECTIVES,
                text="What key performance indicators (KPIs) will you track?",
                reasoning="KPIs provide measurable progress indicators",
                options=["Financial metrics", "Operational metrics", "Customer metrics", "Quality metrics", "Multiple KPI categories"]
            ),
            Question(
                id="bo_4",
                dimension=Dimension.BUSINESS_OBJECTIVES,
                text="What would you consider a failure condition?",
                reasoning="Understanding failure helps identify risks and boundaries",
                options=["No measurable improvement", "Negative impact on other areas", "Exceeding budget/timeline", "Stakeholder dissatisfaction", "Regulatory non-compliance"]
            )
        ]
        
        # Stakeholders Questions
        templates[Dimension.STAKEHOLDERS] = [
            Question(
                id="st_1",
                dimension=Dimension.STAKEHOLDERS,
                text="Who are the primary stakeholders for this analysis?",
                reasoning="Identifying stakeholders ensures proper alignment and communication",
                options=["Executive leadership", "Department managers", "End users", "External customers", "Multiple stakeholder groups"]
            ),
            Question(
                id="st_2",
                dimension=Dimension.STAKEHOLDERS,
                text="Who will be the main users of the analysis results?",
                reasoning="Understanding users helps tailor outputs and recommendations",
                options=["Senior executives", "Middle management", "Operational staff", "External partners", "Mixed user groups"]
            ),
            Question(
                id="st_3",
                dimension=Dimension.STAKEHOLDERS,
                text="Who has decision-making authority for implementing recommendations?",
                reasoning="Identifies key influencers and approval processes",
                options=["CEO/President", "Department head", "Committee/Board", "Multiple decision makers", "External authority"]
            ),
            Question(
                id="st_4",
                dimension=Dimension.STAKEHOLDERS,
                text="Are there any stakeholders who might resist changes?",
                reasoning="Anticipating resistance helps plan change management",
                options=["No expected resistance", "Some operational resistance", "Significant organizational resistance", "External resistance", "Unknown resistance level"]
            )
        ]
        
        # Current Situation Questions
        templates[Dimension.CURRENT_SITUATION] = [
            Question(
                id="cs_1",
                dimension=Dimension.CURRENT_SITUATION,
                text="What is the current baseline state you're measuring against?",
                reasoning="Baseline understanding is crucial for measuring improvement",
                options=["Well-documented current state", "Partially documented", "Estimated current state", "Unknown current state", "Inconsistent measurements"]
            ),
            Question(
                id="cs_2",
                dimension=Dimension.CURRENT_SITUATION,
                text="What existing approaches have you tried to address this problem?",
                reasoning="Learning from previous attempts prevents repeating mistakes",
                options=["No previous attempts", "Informal approaches", "Formal initiatives", "Multiple failed attempts", "Ongoing efforts"]
            ),
            Question(
                id="cs_3",
                dimension=Dimension.CURRENT_SITUATION,
                text="What data and systems are currently available?",
                reasoning="Available data determines analysis possibilities",
                options=["Comprehensive data systems", "Limited data availability", "Fragmented data sources", "Manual data collection", "No relevant data"]
            ),
            Question(
                id="cs_4",
                dimension=Dimension.CURRENT_SITUATION,
                text="How urgent is the need for a solution?",
                reasoning="Urgency affects timeline and resource allocation",
                options=["Immediate crisis", "Urgent business need", "Important but not urgent", "Long-term strategic", "Exploratory timeline"]
            )
        ]
        
        # Constraints Questions
        templates[Dimension.CONSTRAINTS] = [
            Question(
                id="co_1",
                dimension=Dimension.CONSTRAINTS,
                text="What is your budget range for this analysis and implementation?",
                reasoning="Budget constraints affect scope and approach",
                options=["Under $10K", "$10K-$50K", "$50K-$200K", "Over $200K", "Budget not yet determined"]
            ),
            Question(
                id="co_2",
                dimension=Dimension.CONSTRAINTS,
                text="What is your expected timeline for results?",
                reasoning="Timeline affects methodology and depth of analysis",
                options=["Within 1 month", "1-3 months", "3-6 months", "6-12 months", "Over 1 year"]
            ),
            Question(
                id="co_3",
                dimension=Dimension.CONSTRAINTS,
                text="Are there any regulatory or compliance requirements?",
                reasoning="Compliance requirements affect data handling and methodology",
                options=["No regulatory constraints", "Industry regulations", "Data privacy requirements", "Financial regulations", "Multiple compliance areas"]
            ),
            Question(
                id="co_4",
                dimension=Dimension.CONSTRAINTS,
                text="What data access limitations do you have?",
                reasoning="Data limitations affect analysis scope and methods",
                options=["Full data access", "Limited access permissions", "Privacy restrictions", "Technical limitations", "External data needed"]
            )
        ]
        
        # Success Criteria Questions
        templates[Dimension.SUCCESS_CRITERIA] = [
            Question(
                id="sc_1",
                dimension=Dimension.SUCCESS_CRITERIA,
                text="How will you measure the success of this analysis?",
                reasoning="Clear success metrics ensure value delivery",
                options=["Quantitative improvements", "Qualitative insights", "Process improvements", "Decision support", "Risk reduction"]
            ),
            Question(
                id="sc_2",
                dimension=Dimension.SUCCESS_CRITERIA,
                text="What level of accuracy do you need in the results?",
                reasoning="Accuracy requirements affect methodology and validation",
                options=["High precision required", "Moderate accuracy acceptable", "Directional insights sufficient", "Exploratory analysis", "Depends on findings"]
            ),
            Question(
                id="sc_3",
                dimension=Dimension.SUCCESS_CRITERIA,
                text="What improvement threshold would make this worthwhile?",
                reasoning="Minimum improvement thresholds justify investment",
                options=["Any measurable improvement", "5-10% improvement", "10-25% improvement", "Over 25% improvement", "Qualitative improvements"]
            ),
            Question(
                id="sc_4",
                dimension=Dimension.SUCCESS_CRITERIA,
                text="How will you validate the analysis results?",
                reasoning="Validation approach ensures reliability and trust",
                options=["Statistical validation", "Business expert review", "Pilot testing", "Historical comparison", "Multiple validation methods"]
            )
        ]
        
        # Business Domain Questions
        templates[Dimension.BUSINESS_DOMAIN] = [
            Question(
                id="bd_1",
                dimension=Dimension.BUSINESS_DOMAIN,
                text="What industry or business domain are you in?",
                reasoning="Industry context affects analysis approach and benchmarks",
                options=["Technology", "Healthcare", "Financial services", "Retail/E-commerce", "Manufacturing", "Other"]
            ),
            Question(
                id="bd_2",
                dimension=Dimension.BUSINESS_DOMAIN,
                text="Are there industry-specific regulations or standards?",
                reasoning="Industry standards affect methodology and compliance",
                options=["Heavily regulated industry", "Some industry standards", "Minimal regulation", "Emerging industry", "International standards"]
            ),
            Question(
                id="bd_3",
                dimension=Dimension.BUSINESS_DOMAIN,
                text="What are the key market dynamics affecting your business?",
                reasoning="Market context influences analysis priorities and interpretation",
                options=["Stable market", "Growing market", "Declining market", "Highly competitive", "Disruptive changes"]
            ),
            Question(
                id="bd_4",
                dimension=Dimension.BUSINESS_DOMAIN,
                text="How does seasonality or cyclical patterns affect your business?",
                reasoning="Temporal patterns affect data analysis and interpretation",
                options=["Strong seasonal patterns", "Some cyclical effects", "Minimal temporal variation", "Irregular patterns", "Unknown patterns"]
            )
        ]
        
        # Implementation Questions
        templates[Dimension.IMPLEMENTATION] = [
            Question(
                id="im_1",
                dimension=Dimension.IMPLEMENTATION,
                text="How will the analysis results be integrated into your operations?",
                reasoning="Implementation approach affects analysis design and outputs",
                options=["Automated systems integration", "Manual process changes", "Decision support tools", "Strategic planning input", "Multiple integration approaches"]
            ),
            Question(
                id="im_2",
                dimension=Dimension.IMPLEMENTATION,
                text="What are the main barriers to implementing recommendations?",
                reasoning="Understanding barriers helps design actionable recommendations",
                options=["Technical limitations", "Resource constraints", "Organizational resistance", "Regulatory barriers", "Multiple barriers"]
            ),
            Question(
                id="im_3",
                dimension=Dimension.IMPLEMENTATION,
                text="Who will be responsible for implementing changes?",
                reasoning="Implementation ownership affects recommendation design",
                options=["Dedicated project team", "Existing department", "External consultants", "Cross-functional team", "Implementation plan needed"]
            ),
            Question(
                id="im_4",
                dimension=Dimension.IMPLEMENTATION,
                text="How will you manage change and adoption?",
                reasoning="Change management affects implementation success",
                options=["Formal change management", "Gradual rollout", "Training programs", "Stakeholder engagement", "Change strategy needed"]
            )
        ]
        
        return templates
    
    def get_dimension_questions(self, dimension: Dimension, context: Dict[str, Any] = None) -> List[Question]:
        """Get questions for a specific dimension."""
        return self.question_templates.get(dimension, [])
    
    def get_next_batch(self, current_dimension: Dimension, answered_questions: List[str] = None) -> QuestionBatch:
        """
        Get the next batch of questions for the current dimension.
        Ensures batch size is between 1-7 questions.
        """
        if answered_questions is None:
            answered_questions = []
        
        dimension_questions = self.get_dimension_questions(current_dimension)
        
        # Filter out already answered questions
        remaining_questions = [
            q for q in dimension_questions 
            if q.id not in answered_questions
        ]
        
        if not remaining_questions:
            return None
        
        # Create batch with appropriate size (5-7 questions, or remaining if fewer)
        batch_size = min(7, max(1, len(remaining_questions)))
        if len(remaining_questions) > 7:
            # If more than 7 questions remain, take 5-7 for this batch
            batch_size = random.randint(5, 7)
        
        batch_questions = remaining_questions[:batch_size]
        
        # Create context bridge based on dimension
        context_bridges = {
            Dimension.PROBLEM_DEFINITION: "Let's start by understanding the core problem you're trying to solve.",
            Dimension.BUSINESS_OBJECTIVES: "Now let's clarify your specific business objectives and success criteria.",
            Dimension.STAKEHOLDERS: "I'd like to understand who the key stakeholders are for this initiative.",
            Dimension.CURRENT_SITUATION: "Let's examine your current situation and what you've tried before.",
            Dimension.CONSTRAINTS: "Now let's discuss any constraints or limitations we need to consider.",
            Dimension.SUCCESS_CRITERIA: "Let's define how we'll measure success and validate results.",
            Dimension.BUSINESS_DOMAIN: "I need to understand your business domain and industry context.",
            Dimension.IMPLEMENTATION: "Finally, let's discuss how you plan to implement the recommendations."
        }
        
        batch = QuestionBatch(
            batch_id=f"{current_dimension.value}_batch_{len(answered_questions)}",
            dimension=current_dimension,
            questions=batch_questions,
            context_bridge=context_bridges.get(current_dimension, "Let's continue with the next set of questions.")
        )
        
        return batch
    
    def get_next_dimension(self, current_dimension: Dimension) -> Optional[Dimension]:
        """Get the next dimension in the progression sequence."""
        try:
            current_index = self.dimension_order.index(current_dimension)
            if current_index < len(self.dimension_order) - 1:
                return self.dimension_order[current_index + 1]
        except ValueError:
            pass
        return None
    
    def is_dimension_complete(self, dimension: Dimension, answered_questions: List[str]) -> bool:
        """Check if a dimension has been sufficiently explored."""
        dimension_questions = self.get_dimension_questions(dimension)
        dimension_question_ids = [q.id for q in dimension_questions]
        
        answered_in_dimension = [
            q_id for q_id in answered_questions 
            if q_id in dimension_question_ids
        ]
        
        # Consider dimension complete if at least 2 questions answered
        # or all questions in dimension answered
        return (len(answered_in_dimension) >= 2 or 
                len(answered_in_dimension) == len(dimension_question_ids))
    
    def get_all_dimensions(self) -> List[Dimension]:
        """Get all 8 dimensions in order."""
        return self.dimension_order.copy()
    
    def validate_dimension_coverage(self, answered_questions: List[str]) -> Dict[Dimension, bool]:
        """
        Validate that all 8 dimensions have been covered.
        Returns a dict mapping each dimension to whether it's been covered.
        """
        coverage = {}
        for dimension in self.dimension_order:
            coverage[dimension] = self.is_dimension_complete(dimension, answered_questions)
        return coverage