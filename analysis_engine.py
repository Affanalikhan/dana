"""
Analysis Engine for CRISP-DM Business Understanding Specialist

This module implements assumption detection, gap identification, risk assessment,
contradiction detection, and business insight generation capabilities.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Tuple
from enum import Enum
import re
from collections import Counter, defaultdict
import uuid

from crisp_dm_framework import Question, Answer, Dimension
from session_manager import SessionState
from adaptive_engine import Contradiction, ContradictionType


class AssumptionType(Enum):
    """Types of assumptions that can be detected."""
    IMPLICIT_BELIEF = "implicit_belief"
    UNSTATED_REQUIREMENT = "unstated_requirement"
    DOMAIN_ASSUMPTION = "domain_assumption"
    STAKEHOLDER_ASSUMPTION = "stakeholder_assumption"
    RESOURCE_ASSUMPTION = "resource_assumption"
    TIMELINE_ASSUMPTION = "timeline_assumption"


class GapType(Enum):
    """Types of information gaps."""
    MISSING_STAKEHOLDER = "missing_stakeholder"
    UNCLEAR_OBJECTIVE = "unclear_objective"
    UNDEFINED_CONSTRAINT = "undefined_constraint"
    MISSING_SUCCESS_METRIC = "missing_success_metric"
    INCOMPLETE_CONTEXT = "incomplete_context"
    UNSPECIFIED_TIMELINE = "unspecified_timeline"
    MISSING_BUDGET_INFO = "missing_budget_info"
    UNCLEAR_SCOPE = "unclear_scope"


class RiskLevel(Enum):
    """Risk levels for assumptions and gaps."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Assumption:
    """Represents a detected assumption in user responses."""
    id: str
    type: AssumptionType
    description: str
    source_question_id: str
    source_text: str
    risk_level: RiskLevel
    validation_questions: List[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 to 1.0


@dataclass
class Gap:
    """Represents an identified information gap."""
    id: str
    type: GapType
    description: str
    affected_dimensions: List[Dimension]
    severity: RiskLevel
    suggested_questions: List[str] = field(default_factory=list)
    related_question_ids: List[str] = field(default_factory=list)


@dataclass
class BusinessInsight:
    """Represents a business insight generated from analysis."""
    id: str
    category: str  # "opportunity", "risk", "recommendation", "pattern"
    title: str
    description: str
    supporting_evidence: List[str]
    confidence: float
    priority: RiskLevel


@dataclass
class BusinessInsights:
    """Collection of business insights from complete session analysis."""
    session_id: str
    insights: List[BusinessInsight] = field(default_factory=list)
    key_themes: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    opportunities: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    overall_readiness: str = "unknown"  # "high", "medium", "low", "unknown"


class AnalysisEngine:
    """
    Performs business understanding analysis including assumption detection,
    gap identification, risk assessment, and business insight generation.
    """
    
    def __init__(self):
        """Initialize the analysis engine with detection patterns."""
        self.assumption_patterns = self._initialize_assumption_patterns()
        self.gap_detection_rules = self._initialize_gap_detection_rules()
        self.insight_generators = self._initialize_insight_generators()
        self.risk_indicators = self._initialize_risk_indicators()
    
    def _initialize_assumption_patterns(self) -> Dict[AssumptionType, List[str]]:
        """Initialize patterns for detecting different types of assumptions."""
        return {
            AssumptionType.IMPLICIT_BELIEF: [
                r"\b(obviously|clearly|of course|naturally|certainly)\b",
                r"\b(everyone knows|it's common|typical|standard|normal)\b",
                r"\b(should|must|need to|have to|required to)\b",
                r"\b(always|never|all|none|every|no one)\b"
            ],
            AssumptionType.UNSTATED_REQUIREMENT: [
                r"\b(we assume|assuming|given that|since)\b",
                r"\b(it goes without saying|needless to say)\b",
                r"\b(by default|automatically|implicitly)\b"
            ],
            AssumptionType.DOMAIN_ASSUMPTION: [
                r"\b(in our industry|industry standard|best practice)\b",
                r"\b(regulatory requirement|compliance|mandated)\b",
                r"\b(market expects|customers expect|users expect)\b"
            ],
            AssumptionType.STAKEHOLDER_ASSUMPTION: [
                r"\b(management will|leadership supports|team agrees)\b",
                r"\b(users will|customers will|people will|stakeholders will)\b",
                r"\b(everyone is on board|full support|buy-in)\b",
                r"\b(will support|will adapt|will embrace|will accept)\b"
            ],
            AssumptionType.RESOURCE_ASSUMPTION: [
                r"\b(we have|available|sufficient|adequate)\b.*\b(budget|resources|time|staff)\b",
                r"\b(can get|will provide|will allocate|will be allocated)\b.*\b(funding|people|support|staff)\b",
                r"\b(staff|people|resources|budget)\b.*\b(will be|can be|are)\b.*\b(allocated|provided|available)\b"
            ],
            AssumptionType.TIMELINE_ASSUMPTION: [
                r"\b(should be done|will be ready|can complete)\b.*\b(by|within|in)\b",
                r"\b(quick|fast|easy|simple)\b.*\b(implementation|deployment|rollout)\b"
            ]
        }
    
    def _initialize_gap_detection_rules(self) -> Dict[GapType, Dict[str, Any]]:
        """Initialize rules for detecting information gaps."""
        return {
            GapType.MISSING_STAKEHOLDER: {
                "required_keywords": ["stakeholder", "user", "customer", "team", "management"],
                "vague_indicators": ["someone", "people", "they", "users", "management"],
                "dimensions": [Dimension.STAKEHOLDERS, Dimension.IMPLEMENTATION]
            },
            GapType.UNCLEAR_OBJECTIVE: {
                "required_keywords": ["goal", "objective", "target", "outcome", "result"],
                "vague_indicators": ["better", "improve", "enhance", "optimize", "good"],
                "dimensions": [Dimension.BUSINESS_OBJECTIVES, Dimension.SUCCESS_CRITERIA]
            },
            GapType.UNDEFINED_CONSTRAINT: {
                "required_keywords": ["budget", "timeline", "resource", "limitation", "constraint"],
                "vague_indicators": ["limited", "tight", "flexible", "depends", "varies"],
                "dimensions": [Dimension.CONSTRAINTS]
            },
            GapType.MISSING_SUCCESS_METRIC: {
                "required_keywords": ["metric", "KPI", "measure", "track", "monitor"],
                "vague_indicators": ["success", "good", "better", "improvement", "progress"],
                "dimensions": [Dimension.SUCCESS_CRITERIA, Dimension.BUSINESS_OBJECTIVES]
            },
            GapType.INCOMPLETE_CONTEXT: {
                "required_keywords": ["current", "existing", "baseline", "status", "situation"],
                "vague_indicators": ["some", "various", "different", "multiple", "several"],
                "dimensions": [Dimension.CURRENT_SITUATION, Dimension.BUSINESS_DOMAIN]
            },
            GapType.UNSPECIFIED_TIMELINE: {
                "required_keywords": ["deadline", "timeline", "schedule", "when", "duration"],
                "vague_indicators": ["soon", "later", "eventually", "sometime", "future"],
                "dimensions": [Dimension.CONSTRAINTS, Dimension.IMPLEMENTATION]
            },
            GapType.MISSING_BUDGET_INFO: {
                "required_keywords": ["budget", "cost", "funding", "investment", "price"],
                "vague_indicators": ["expensive", "cheap", "affordable", "reasonable", "depends"],
                "dimensions": [Dimension.CONSTRAINTS]
            },
            GapType.UNCLEAR_SCOPE: {
                "required_keywords": ["scope", "boundary", "include", "exclude", "limit"],
                "vague_indicators": ["everything", "anything", "all", "some", "various"],
                "dimensions": [Dimension.PROBLEM_DEFINITION, Dimension.BUSINESS_OBJECTIVES]
            }
        }
    
    def _initialize_insight_generators(self) -> Dict[str, Dict[str, Any]]:
        """Initialize patterns for generating business insights."""
        return {
            "opportunity": {
                "patterns": [
                    r"\b(growth|expand|increase|scale|opportunity)\b",
                    r"\b(competitive advantage|market share|differentiation)\b",
                    r"\b(efficiency|optimization|automation|streamline)\b"
                ],
                "indicators": ["new market", "cost savings", "revenue increase", "process improvement"]
            },
            "risk": {
                "patterns": [
                    r"\b(risk|threat|concern|worry|problem|issue)\b",
                    r"\b(failure|fail|unsuccessful|difficult|challenge)\b",
                    r"\b(resistance|opposition|barrier|obstacle)\b"
                ],
                "indicators": ["budget overrun", "timeline delay", "stakeholder resistance", "technical complexity"]
            },
            "recommendation": {
                "patterns": [
                    r"\b(should|recommend|suggest|propose|advise)\b",
                    r"\b(best practice|proven approach|successful|effective)\b",
                    r"\b(pilot|test|prototype|phase|gradual)\b"
                ],
                "indicators": ["phased approach", "stakeholder engagement", "risk mitigation", "success metrics"]
            },
            "pattern": {
                "patterns": [
                    r"\b(pattern|trend|consistent|recurring|common)\b",
                    r"\b(similar|same|typical|usual|standard)\b",
                    r"\b(across|throughout|multiple|various|different)\b"
                ],
                "indicators": ["organizational pattern", "industry trend", "stakeholder behavior", "process similarity"]
            }
        }
    
    def _initialize_risk_indicators(self) -> Dict[str, RiskLevel]:
        """Initialize risk level indicators."""
        return {
            # High risk indicators
            "no budget": RiskLevel.CRITICAL,
            "no timeline": RiskLevel.HIGH,
            "no stakeholder buy-in": RiskLevel.CRITICAL,
            "unclear objectives": RiskLevel.HIGH,
            "regulatory concerns": RiskLevel.HIGH,
            "technical complexity": RiskLevel.MEDIUM,
            "organizational resistance": RiskLevel.HIGH,
            "resource constraints": RiskLevel.MEDIUM,
            "tight deadline": RiskLevel.MEDIUM,
            "multiple stakeholders": RiskLevel.MEDIUM,
            "undefined success criteria": RiskLevel.HIGH,
            "assumption heavy": RiskLevel.MEDIUM
        }
    
    def identify_assumptions(self, answers: List[Answer]) -> List[Assumption]:
        """
        Identify assumptions in user responses across all answer types.
        
        Args:
            answers: List of answers to analyze for assumptions
            
        Returns:
            List of detected assumptions with validation questions
        """
        assumptions = []
        
        for answer in answers:
            answer_assumptions = self._detect_assumptions_in_answer(answer)
            assumptions.extend(answer_assumptions)
        
        # Remove duplicates and rank by confidence
        unique_assumptions = self._deduplicate_assumptions(assumptions)
        
        # Generate validation questions for each assumption
        for assumption in unique_assumptions:
            assumption.validation_questions = self._generate_validation_questions(assumption)
        
        return sorted(unique_assumptions, key=lambda a: a.confidence, reverse=True)
    
    def _detect_assumptions_in_answer(self, answer: Answer) -> List[Assumption]:
        """Detect assumptions in a single answer."""
        assumptions = []
        text_lower = answer.response.lower()
        
        for assumption_type, patterns in self.assumption_patterns.items():
            for pattern in patterns:
                matches = list(re.finditer(pattern, text_lower))
                for match in matches:
                    # Extract the sentence containing the assumption
                    assumption_text = self._extract_assumption_context(answer.response, match)
                    
                    if assumption_text:
                        assumption = Assumption(
                            id=f"assumption_{uuid.uuid4().hex[:8]}",
                            type=assumption_type,
                            description=self._generate_assumption_description(assumption_type, assumption_text),
                            source_question_id=answer.question_id,
                            source_text=assumption_text,
                            risk_level=self._assess_assumption_risk(assumption_type, assumption_text),
                            confidence=self._calculate_assumption_confidence(assumption_type, assumption_text, match)
                        )
                        assumptions.append(assumption)
        
        return assumptions
    
    def _extract_assumption_context(self, text: str, match: re.Match) -> str:
        """Extract the context around an assumption match."""
        # Find the sentence containing the match
        sentences = re.split(r'[.!?]+', text)
        match_pos = match.start()
        
        # Find which sentence contains the match
        current_pos = 0
        for sentence in sentences:
            sentence_end = current_pos + len(sentence)
            if current_pos <= match_pos <= sentence_end:
                return sentence.strip()
            current_pos = sentence_end + 1
        
        # Fallback: return text around the match
        start = max(0, match_pos - 50)
        end = min(len(text), match_pos + 50)
        return text[start:end].strip()
    
    def _generate_assumption_description(self, assumption_type: AssumptionType, text: str) -> str:
        """Generate a description for the detected assumption."""
        type_descriptions = {
            AssumptionType.IMPLICIT_BELIEF: f"Implicit belief detected: {text[:100]}...",
            AssumptionType.UNSTATED_REQUIREMENT: f"Unstated requirement: {text[:100]}...",
            AssumptionType.DOMAIN_ASSUMPTION: f"Domain-specific assumption: {text[:100]}...",
            AssumptionType.STAKEHOLDER_ASSUMPTION: f"Stakeholder behavior assumption: {text[:100]}...",
            AssumptionType.RESOURCE_ASSUMPTION: f"Resource availability assumption: {text[:100]}...",
            AssumptionType.TIMELINE_ASSUMPTION: f"Timeline assumption: {text[:100]}..."
        }
        
        return type_descriptions.get(assumption_type, f"Assumption detected: {text[:100]}...")
    
    def _assess_assumption_risk(self, assumption_type: AssumptionType, text: str) -> RiskLevel:
        """Assess the risk level of an assumption."""
        # High-risk assumption types
        high_risk_types = [
            AssumptionType.STAKEHOLDER_ASSUMPTION,
            AssumptionType.RESOURCE_ASSUMPTION,
            AssumptionType.TIMELINE_ASSUMPTION
        ]
        
        if assumption_type in high_risk_types:
            return RiskLevel.HIGH
        
        # Check for high-risk keywords in text
        high_risk_keywords = ["always", "never", "everyone", "no one", "must", "required"]
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in high_risk_keywords):
            return RiskLevel.MEDIUM
        
        return RiskLevel.LOW
    
    def _calculate_assumption_confidence(self, assumption_type: AssumptionType, 
                                       text: str, match: re.Match) -> float:
        """Calculate confidence score for assumption detection."""
        confidence = 0.5  # Base confidence
        
        # Strong assumption indicators increase confidence
        strong_indicators = ["obviously", "clearly", "must", "always", "never", "everyone"]
        if any(indicator in text.lower() for indicator in strong_indicators):
            confidence += 0.3
        
        # Longer context increases confidence
        if len(text) > 50:
            confidence += 0.1
        
        # Specific assumption types have different base confidence
        type_confidence = {
            AssumptionType.IMPLICIT_BELIEF: 0.8,
            AssumptionType.UNSTATED_REQUIREMENT: 0.7,
            AssumptionType.DOMAIN_ASSUMPTION: 0.6,
            AssumptionType.STAKEHOLDER_ASSUMPTION: 0.7,
            AssumptionType.RESOURCE_ASSUMPTION: 0.8,
            AssumptionType.TIMELINE_ASSUMPTION: 0.7
        }
        
        confidence = max(confidence, type_confidence.get(assumption_type, 0.5))
        
        return min(confidence, 1.0)
    
    def _deduplicate_assumptions(self, assumptions: List[Assumption]) -> List[Assumption]:
        """Remove duplicate assumptions based on similarity."""
        unique_assumptions = []
        seen_texts = set()
        
        for assumption in assumptions:
            # Create a normalized version for comparison
            normalized_text = re.sub(r'\W+', ' ', assumption.source_text.lower()).strip()
            
            if normalized_text not in seen_texts:
                unique_assumptions.append(assumption)
                seen_texts.add(normalized_text)
        
        return unique_assumptions
    
    def _generate_validation_questions(self, assumption: Assumption) -> List[str]:
        """Generate validation questions for an assumption."""
        validation_templates = {
            AssumptionType.IMPLICIT_BELIEF: [
                f"Is it always true that {assumption.source_text.lower()}?",
                f"Are there any exceptions to this belief in your context?"
            ],
            AssumptionType.UNSTATED_REQUIREMENT: [
                f"Is this requirement explicitly documented: {assumption.source_text}?",
                f"Who confirmed this requirement and when?"
            ],
            AssumptionType.DOMAIN_ASSUMPTION: [
                f"Is this standard practice verified for your specific industry context?",
                f"Have you confirmed this applies to your organization specifically?"
            ],
            AssumptionType.STAKEHOLDER_ASSUMPTION: [
                f"Have you confirmed stakeholder agreement on: {assumption.source_text}?",
                f"What evidence supports this stakeholder behavior assumption?"
            ],
            AssumptionType.RESOURCE_ASSUMPTION: [
                f"Have you confirmed resource availability: {assumption.source_text}?",
                f"What is your backup plan if these resources are not available?"
            ],
            AssumptionType.TIMELINE_ASSUMPTION: [
                f"Have you validated this timeline assumption: {assumption.source_text}?",
                f"What factors could affect this timeline estimate?"
            ]
        }
        
        return validation_templates.get(assumption.type, [
            f"Can you validate this assumption: {assumption.source_text}?",
            f"What evidence supports this assumption?"
        ])
    
    def detect_gaps(self, answers: List[Answer]) -> List[Gap]:
        """
        Detect information gaps in the collected answers.
        
        Args:
            answers: List of answers to analyze for gaps
            
        Returns:
            List of identified gaps with suggested questions
        """
        gaps = []
        
        # Analyze answers by dimension
        answers_by_dimension = self._group_answers_by_dimension(answers)
        
        # Check each gap type
        for gap_type, rules in self.gap_detection_rules.items():
            gap = self._detect_specific_gap(gap_type, rules, answers, answers_by_dimension)
            if gap:
                gaps.append(gap)
        
        # Detect cross-dimensional gaps
        cross_gaps = self._detect_cross_dimensional_gaps(answers_by_dimension)
        gaps.extend(cross_gaps)
        
        # Generate suggested questions for each gap
        for gap in gaps:
            gap.suggested_questions = self._generate_gap_questions(gap)
        
        return sorted(gaps, key=lambda g: g.severity.value, reverse=True)
    
    def _group_answers_by_dimension(self, answers: List[Answer]) -> Dict[Dimension, List[Answer]]:
        """Group answers by their question dimensions."""
        # This would need access to questions to determine dimensions
        # For now, return empty dict - would need to be enhanced with question lookup
        return defaultdict(list)
    
    def _detect_specific_gap(self, gap_type: GapType, rules: Dict[str, Any], 
                           answers: List[Answer], answers_by_dimension: Dict[Dimension, List[Answer]]) -> Optional[Gap]:
        """Detect a specific type of gap based on rules."""
        required_keywords = rules["required_keywords"]
        vague_indicators = rules["vague_indicators"]
        relevant_dimensions = rules["dimensions"]
        
        # Check if relevant dimensions have answers
        has_relevant_answers = any(
            dim in answers_by_dimension and answers_by_dimension[dim]
            for dim in relevant_dimensions
        )
        
        if not has_relevant_answers:
            # Missing entire dimension
            return Gap(
                id=f"gap_{gap_type.value}_{uuid.uuid4().hex[:8]}",
                type=gap_type,
                description=f"Missing information about {gap_type.value.replace('_', ' ')}",
                affected_dimensions=relevant_dimensions,
                severity=RiskLevel.HIGH
            )
        
        # Check for vague answers in relevant dimensions
        vague_count = 0
        specific_count = 0
        related_question_ids = []
        
        for answer in answers:
            text_lower = answer.response.lower()
            
            # Count vague indicators
            vague_matches = sum(1 for indicator in vague_indicators if indicator in text_lower)
            if vague_matches > 0:
                vague_count += vague_matches
                related_question_ids.append(answer.question_id)
            
            # Count specific indicators
            specific_matches = sum(1 for keyword in required_keywords if keyword in text_lower)
            if specific_matches > 0:
                specific_count += specific_matches
        
        # Determine if there's a gap based on vague vs specific ratio
        if vague_count > specific_count and vague_count > 2:
            severity = RiskLevel.MEDIUM if vague_count < 5 else RiskLevel.HIGH
            
            return Gap(
                id=f"gap_{gap_type.value}_{uuid.uuid4().hex[:8]}",
                type=gap_type,
                description=f"Vague or incomplete information about {gap_type.value.replace('_', ' ')}",
                affected_dimensions=relevant_dimensions,
                severity=severity,
                related_question_ids=related_question_ids
            )
        
        return None
    
    def _detect_cross_dimensional_gaps(self, answers_by_dimension: Dict[Dimension, List[Answer]]) -> List[Gap]:
        """Detect gaps that span multiple dimensions."""
        gaps = []
        
        # Check for missing dimension coverage
        all_dimensions = set(Dimension)
        covered_dimensions = set(answers_by_dimension.keys())
        missing_dimensions = all_dimensions - covered_dimensions
        
        if missing_dimensions:
            gap = Gap(
                id=f"gap_missing_dimensions_{uuid.uuid4().hex[:8]}",
                type=GapType.INCOMPLETE_CONTEXT,
                description=f"Missing coverage of dimensions: {', '.join(d.value for d in missing_dimensions)}",
                affected_dimensions=list(missing_dimensions),
                severity=RiskLevel.HIGH if len(missing_dimensions) > 2 else RiskLevel.MEDIUM
            )
            gaps.append(gap)
        
        return gaps
    
    def _generate_gap_questions(self, gap: Gap) -> List[str]:
        """Generate suggested questions to fill a gap."""
        gap_question_templates = {
            GapType.MISSING_STAKEHOLDER: [
                "Who are the specific individuals or roles that will be affected by this initiative?",
                "Which stakeholders have decision-making authority for this project?",
                "Are there any stakeholders who might resist or oppose this change?"
            ],
            GapType.UNCLEAR_OBJECTIVE: [
                "Can you provide specific, measurable objectives for this initiative?",
                "What exact outcomes do you want to achieve, with numbers or percentages?",
                "How will you know when you have successfully achieved your objectives?"
            ],
            GapType.UNDEFINED_CONSTRAINT: [
                "What is your specific budget range for this project?",
                "What is your target timeline with specific milestones?",
                "What resources (people, technology, etc.) are available or needed?"
            ],
            GapType.MISSING_SUCCESS_METRIC: [
                "What specific metrics or KPIs will you use to measure success?",
                "How will you track progress toward your objectives?",
                "What would constitute failure or unacceptable results?"
            ],
            GapType.INCOMPLETE_CONTEXT: [
                "Can you describe your current situation in more detail?",
                "What is the baseline or starting point for this initiative?",
                "What relevant background information should we consider?"
            ],
            GapType.UNSPECIFIED_TIMELINE: [
                "When do you need this completed?",
                "Are there any fixed deadlines or time constraints?",
                "What is driving the timing for this initiative?"
            ],
            GapType.MISSING_BUDGET_INFO: [
                "What budget has been allocated for this project?",
                "What is the expected return on investment?",
                "Are there any cost constraints we should be aware of?"
            ],
            GapType.UNCLEAR_SCOPE: [
                "What specifically is included in the scope of this project?",
                "What is explicitly excluded from this initiative?",
                "Where are the boundaries of this project?"
            ]
        }
        
        return gap_question_templates.get(gap.type, [
            f"Can you provide more specific information about {gap.type.value.replace('_', ' ')}?",
            f"What additional details can you share regarding {gap.type.value.replace('_', ' ')}?"
        ])
    
    def generate_insights(self, complete_session: SessionState) -> BusinessInsights:
        """
        Generate comprehensive business insights from a complete session.
        
        Args:
            complete_session: Complete session state with all answers
            
        Returns:
            BusinessInsights: Comprehensive analysis and recommendations
        """
        insights = BusinessInsights(session_id=complete_session.session_id)
        
        # Analyze all answers for insights
        all_answers = complete_session.answers_collected
        
        # Generate different types of insights
        insights.insights.extend(self._generate_opportunity_insights(all_answers))
        insights.insights.extend(self._generate_risk_insights(all_answers))
        insights.insights.extend(self._generate_recommendation_insights(all_answers))
        insights.insights.extend(self._generate_pattern_insights(all_answers))
        
        # Extract key themes
        insights.key_themes = self._extract_session_themes(all_answers)
        
        # Identify risk factors
        insights.risk_factors = self._identify_risk_factors(all_answers)
        
        # Identify opportunities
        insights.opportunities = self._identify_opportunities(all_answers)
        
        # Generate recommendations
        insights.recommendations = self._generate_session_recommendations(all_answers, complete_session)
        
        # Assess overall readiness
        insights.overall_readiness = self._assess_overall_readiness(all_answers, complete_session)
        
        return insights
    
    def _generate_opportunity_insights(self, answers: List[Answer]) -> List[BusinessInsight]:
        """Generate opportunity-focused insights."""
        insights = []
        opportunity_patterns = self.insight_generators["opportunity"]["patterns"]
        
        for answer in answers:
            text_lower = answer.response.lower()
            
            for pattern in opportunity_patterns:
                matches = list(re.finditer(pattern, text_lower))
                for match in matches:
                    context = self._extract_insight_context(answer.response, match)
                    
                    insight = BusinessInsight(
                        id=f"opportunity_{uuid.uuid4().hex[:8]}",
                        category="opportunity",
                        title="Business Opportunity Identified",
                        description=f"Opportunity detected: {context}",
                        supporting_evidence=[f"From response: {answer.response[:100]}..."],
                        confidence=0.7,
                        priority=RiskLevel.MEDIUM
                    )
                    insights.append(insight)
        
        return insights[:3]  # Limit to top 3 opportunities
    
    def _generate_risk_insights(self, answers: List[Answer]) -> List[BusinessInsight]:
        """Generate risk-focused insights."""
        insights = []
        risk_patterns = self.insight_generators["risk"]["patterns"]
        
        for answer in answers:
            text_lower = answer.response.lower()
            
            for pattern in risk_patterns:
                matches = list(re.finditer(pattern, text_lower))
                for match in matches:
                    context = self._extract_insight_context(answer.response, match)
                    
                    # Assess risk severity
                    severity = self._assess_risk_severity(context)
                    
                    insight = BusinessInsight(
                        id=f"risk_{uuid.uuid4().hex[:8]}",
                        category="risk",
                        title="Potential Risk Identified",
                        description=f"Risk factor: {context}",
                        supporting_evidence=[f"From response: {answer.response[:100]}..."],
                        confidence=0.8,
                        priority=severity
                    )
                    insights.append(insight)
        
        return insights[:3]  # Limit to top 3 risks
    
    def _generate_recommendation_insights(self, answers: List[Answer]) -> List[BusinessInsight]:
        """Generate recommendation insights."""
        insights = []
        
        # Analyze answer patterns to generate recommendations
        common_themes = self._extract_session_themes(answers)
        
        for theme in common_themes[:3]:  # Top 3 themes
            recommendation = self._generate_theme_recommendation(theme, answers)
            if recommendation:
                insights.append(recommendation)
        
        return insights
    
    def _generate_pattern_insights(self, answers: List[Answer]) -> List[BusinessInsight]:
        """Generate pattern-based insights."""
        insights = []
        
        # Look for recurring patterns across answers
        word_frequency = Counter()
        for answer in answers:
            words = re.findall(r'\b\w+\b', answer.response.lower())
            # Filter out common words
            meaningful_words = [w for w in words if len(w) > 4 and w not in 
                             {'that', 'this', 'with', 'have', 'will', 'would', 'could', 'should'}]
            word_frequency.update(meaningful_words)
        
        # Generate insights for most common meaningful words
        for word, count in word_frequency.most_common(3):
            if count >= 3:  # Word appears in multiple answers
                insight = BusinessInsight(
                    id=f"pattern_{uuid.uuid4().hex[:8]}",
                    category="pattern",
                    title=f"Recurring Theme: {word.title()}",
                    description=f"The term '{word}' appears frequently ({count} times), indicating it's a key focus area",
                    supporting_evidence=[f"Mentioned {count} times across responses"],
                    confidence=0.6,
                    priority=RiskLevel.LOW
                )
                insights.append(insight)
        
        return insights
    
    def _extract_insight_context(self, text: str, match: re.Match) -> str:
        """Extract context around an insight match."""
        # Get surrounding context (up to 100 characters)
        start = max(0, match.start() - 50)
        end = min(len(text), match.end() + 50)
        context = text[start:end].strip()
        
        # Clean up the context
        if len(context) > 100:
            context = context[:100] + "..."
        
        return context
    
    def _assess_risk_severity(self, context: str) -> RiskLevel:
        """Assess the severity of a risk based on context."""
        high_severity_keywords = ["critical", "major", "significant", "serious", "urgent"]
        medium_severity_keywords = ["important", "concern", "issue", "problem", "challenge"]
        
        context_lower = context.lower()
        
        if any(keyword in context_lower for keyword in high_severity_keywords):
            return RiskLevel.HIGH
        elif any(keyword in context_lower for keyword in medium_severity_keywords):
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _extract_session_themes(self, answers: List[Answer]) -> List[str]:
        """Extract key themes from all session answers."""
        themes = []
        
        # Combine all answer text
        all_text = " ".join(answer.response for answer in answers).lower()
        
        # Define theme categories and their keywords
        theme_keywords = {
            "efficiency": ["efficiency", "optimize", "streamline", "improve", "faster", "better"],
            "growth": ["growth", "expand", "scale", "increase", "revenue", "market"],
            "customer": ["customer", "client", "user", "satisfaction", "experience", "service"],
            "technology": ["technology", "system", "software", "digital", "automation", "platform"],
            "cost": ["cost", "budget", "expense", "savings", "reduce", "cheaper"],
            "quality": ["quality", "accuracy", "reliability", "performance", "standards"],
            "compliance": ["compliance", "regulation", "policy", "standard", "requirement"],
            "innovation": ["innovation", "new", "creative", "novel", "breakthrough", "advanced"]
        }
        
        # Count theme occurrences
        theme_scores = {}
        for theme, keywords in theme_keywords.items():
            score = sum(all_text.count(keyword) for keyword in keywords)
            if score > 0:
                theme_scores[theme] = score
        
        # Return top themes
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        themes = [theme for theme, score in sorted_themes[:5]]
        
        return themes
    
    def _identify_risk_factors(self, answers: List[Answer]) -> List[str]:
        """Identify risk factors from session answers."""
        risk_factors = []
        
        risk_keywords = {
            "budget_risk": ["no budget", "limited budget", "tight budget", "cost overrun"],
            "timeline_risk": ["tight deadline", "unrealistic timeline", "time pressure", "delayed"],
            "stakeholder_risk": ["resistance", "opposition", "conflict", "disagreement"],
            "technical_risk": ["complex", "difficult", "challenging", "untested", "new technology"],
            "resource_risk": ["limited resources", "understaffed", "lack of expertise", "capacity"],
            "regulatory_risk": ["compliance", "regulation", "audit", "legal", "policy"]
        }
        
        all_text = " ".join(answer.response for answer in answers).lower()
        
        for risk_category, keywords in risk_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                risk_factors.append(risk_category.replace("_", " ").title())
        
        return risk_factors
    
    def _identify_opportunities(self, answers: List[Answer]) -> List[str]:
        """Identify opportunities from session answers."""
        opportunities = []
        
        opportunity_keywords = {
            "automation": ["automate", "automation", "streamline", "efficiency"],
            "market_expansion": ["new market", "expand", "growth", "opportunity"],
            "cost_savings": ["save", "reduce cost", "efficiency", "optimize"],
            "competitive_advantage": ["competitive", "advantage", "differentiate", "unique"],
            "innovation": ["innovate", "new", "creative", "breakthrough"],
            "customer_satisfaction": ["customer satisfaction", "user experience", "service quality"]
        }
        
        all_text = " ".join(answer.response for answer in answers).lower()
        
        for opp_category, keywords in opportunity_keywords.items():
            if any(keyword in all_text for keyword in keywords):
                opportunities.append(opp_category.replace("_", " ").title())
        
        return opportunities
    
    def _generate_session_recommendations(self, answers: List[Answer], session: SessionState) -> List[str]:
        """Generate recommendations based on session analysis."""
        recommendations = []
        
        # Analyze session completeness
        if len(answers) < 15:  # Assuming minimum 15 answers for comprehensive understanding
            recommendations.append("Consider gathering more detailed information across all dimensions")
        
        # Check for common patterns and generate recommendations
        all_text = " ".join(answer.response for answer in answers).lower()
        
        if "pilot" in all_text or "test" in all_text:
            recommendations.append("Consider implementing a pilot program to validate assumptions")
        
        if "stakeholder" in all_text and ("resistance" in all_text or "concern" in all_text):
            recommendations.append("Develop a comprehensive stakeholder engagement and change management plan")
        
        if "budget" in all_text and ("limited" in all_text or "tight" in all_text):
            recommendations.append("Prioritize initiatives based on ROI and consider phased implementation")
        
        if "timeline" in all_text and ("urgent" in all_text or "tight" in all_text):
            recommendations.append("Assess timeline feasibility and identify critical path dependencies")
        
        if "data" in all_text or "analytics" in all_text:
            recommendations.append("Ensure data quality and availability before proceeding with analysis")
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _generate_theme_recommendation(self, theme: str, answers: List[Answer]) -> Optional[BusinessInsight]:
        """Generate a recommendation based on a detected theme."""
        theme_recommendations = {
            "efficiency": "Focus on process optimization and automation opportunities to maximize efficiency gains",
            "growth": "Develop a scalable approach that can support future growth requirements",
            "customer": "Prioritize customer-centric metrics and feedback mechanisms in your success criteria",
            "technology": "Ensure technical architecture can support long-term scalability and integration needs",
            "cost": "Implement cost-benefit analysis and ROI tracking throughout the initiative",
            "quality": "Establish quality assurance processes and performance monitoring from the start",
            "compliance": "Engage compliance and legal teams early to ensure all regulatory requirements are met",
            "innovation": "Balance innovation with risk management and establish clear success criteria"
        }
        
        if theme in theme_recommendations:
            return BusinessInsight(
                id=f"recommendation_{theme}_{uuid.uuid4().hex[:8]}",
                category="recommendation",
                title=f"{theme.title()} Recommendation",
                description=theme_recommendations[theme],
                supporting_evidence=[f"Based on recurring {theme} theme in responses"],
                confidence=0.8,
                priority=RiskLevel.MEDIUM
            )
        
        return None
    
    def _assess_overall_readiness(self, answers: List[Answer], session: SessionState) -> str:
        """Assess overall readiness for proceeding to data mining phases."""
        readiness_score = 0
        max_score = 10
        
        # Check answer completeness (2 points)
        if len(answers) >= 20:
            readiness_score += 2
        elif len(answers) >= 15:
            readiness_score += 1
        
        # Check dimension coverage (2 points)
        covered_dimensions = len(session.dimensions_completed)
        if covered_dimensions >= 7:
            readiness_score += 2
        elif covered_dimensions >= 5:
            readiness_score += 1
        
        # Check for clear objectives (2 points)
        all_text = " ".join(answer.response for answer in answers).lower()
        if any(keyword in all_text for keyword in ["specific", "measurable", "target", "goal"]):
            readiness_score += 2
        elif any(keyword in all_text for keyword in ["objective", "outcome", "result"]):
            readiness_score += 1
        
        # Check for stakeholder clarity (2 points)
        if any(keyword in all_text for keyword in ["stakeholder", "decision maker", "sponsor"]):
            readiness_score += 2
        elif any(keyword in all_text for keyword in ["team", "management", "user"]):
            readiness_score += 1
        
        # Check for resource clarity (2 points)
        if any(keyword in all_text for keyword in ["budget", "timeline", "resource", "funding"]):
            readiness_score += 2
        elif any(keyword in all_text for keyword in ["cost", "time", "staff"]):
            readiness_score += 1
        
        # Determine readiness level
        if readiness_score >= 8:
            return "high"
        elif readiness_score >= 6:
            return "medium"
        elif readiness_score >= 4:
            return "low"
        else:
            return "unknown"