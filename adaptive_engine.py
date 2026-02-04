"""
Adaptive Logic Engine for CRISP-DM Business Understanding Specialist

This module implements intelligent question adaptation based on user responses,
including answer analysis for complexity, contradictions, and vagueness detection,
and context-aware follow-up question generation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from enum import Enum
import re
from collections import Counter
import uuid

from crisp_dm_framework import Question, Answer, Dimension, QuestionType


class ComplexityLevel(Enum):
    """Levels of answer complexity."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class VaguenessLevel(Enum):
    """Levels of answer vagueness."""
    CLEAR = "clear"
    SOMEWHAT_VAGUE = "somewhat_vague"
    VERY_VAGUE = "very_vague"


class ContradictionType(Enum):
    """Types of contradictions detected."""
    DIRECT = "direct"  # Direct contradiction between statements
    IMPLICIT = "implicit"  # Implicit contradiction through implications
    TEMPORAL = "temporal"  # Contradiction in timeline or sequence
    SCOPE = "scope"  # Contradiction in scope or boundaries


@dataclass
class AnalysisResult:
    """Results of answer analysis."""
    complexity_level: ComplexityLevel
    vagueness_level: VaguenessLevel
    contradictions: List['Contradiction'] = field(default_factory=list)
    assumptions: List[str] = field(default_factory=list)
    domain_indicators: List[str] = field(default_factory=list)
    requires_followup: bool = False
    followup_reasons: List[str] = field(default_factory=list)


@dataclass
class Contradiction:
    """Represents a detected contradiction."""
    type: ContradictionType
    question_ids: List[str]
    description: str
    severity: str  # "low", "medium", "high"


@dataclass
class DomainContext:
    """Domain-specific context information."""
    industry: Optional[str] = None
    business_type: Optional[str] = None
    regulatory_environment: Optional[str] = None
    market_dynamics: Optional[str] = None
    technical_complexity: Optional[str] = None


class AdaptiveEngine:
    """
    Provides intelligent question adaptation based on user responses.
    Analyzes answers for complexity, contradictions, and vagueness,
    and generates context-aware follow-up questions.
    """
    
    def __init__(self):
        """Initialize the adaptive engine with analysis patterns."""
        self.complexity_patterns = self._initialize_complexity_patterns()
        self.vagueness_patterns = self._initialize_vagueness_patterns()
        self.domain_patterns = self._initialize_domain_patterns()
        self.assumption_patterns = self._initialize_assumption_patterns()
    
    def _initialize_complexity_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting answer complexity."""
        return {
            "high_complexity": [
                r"\b(multiple|several|various|numerous|many)\b",
                r"\b(complex|complicated|intricate|sophisticated)\b",
                r"\b(depends on|varies by|different for)\b",
                r"\b(integration|coordination|alignment)\b",
                r"\b(stakeholders?|departments?|teams?|groups?)\b.*\b(multiple|several|various)\b",
                r"\b(phases?|stages?|steps?)\b.*\b(multiple|several|many)\b",
                r"\b(factors?|variables?|considerations?)\b.*\b(multiple|several|many)\b"
            ],
            "medium_complexity": [
                r"\b(some|few|couple)\b",
                r"\b(moderate|reasonable|manageable)\b",
                r"\b(two|three|four)\b.*\b(options?|approaches?|methods?)\b",
                r"\b(primary|main|key)\b.*\b(and|plus|also)\b.*\b(secondary|additional)\b"
            ],
            "technical_indicators": [
                r"\b(system|software|technology|platform|database|API|integration)\b",
                r"\b(algorithm|model|analytics|data|metrics|KPI)\b",
                r"\b(automation|workflow|process|procedure)\b"
            ]
        }
    
    def _initialize_vagueness_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting answer vagueness."""
        return {
            "very_vague": [
                r"\b(maybe|perhaps|possibly|might|could be|not sure|unclear)\b",
                r"\b(something|anything|everything|nothing specific)\b",
                r"\b(kind of|sort of|more or less|approximately|roughly)\b",
                r"\b(I think|I believe|I guess|I assume)\b",
                r"\b(probably|likely|unlikely|doubtful)\b",
                r"\b(general|generic|broad|wide|overall)\b.*\b(improvement|better|good)\b"
            ],
            "somewhat_vague": [
                r"\b(some|few|several|various)\b(?!\s+specific)",
                r"\b(better|improve|enhance|optimize)\b(?!\s+by\s+\d+)",
                r"\b(significant|substantial|considerable)\b(?!\s+\d+)",
                r"\b(soon|later|eventually|in the future)\b",
                r"\b(relevant|appropriate|suitable|adequate)\b"
            ],
            "hedge_words": [
                r"\b(typically|usually|generally|often|sometimes|occasionally)\b",
                r"\b(tend to|seems to|appears to|looks like)\b"
            ]
        }
    
    def _initialize_domain_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting domain-specific context."""
        return {
            "technology": [
                r"\b(software|hardware|system|platform|cloud|SaaS|API|database)\b",
                r"\b(development|programming|coding|deployment|DevOps)\b",
                r"\b(users?|customers?|clients?)\b.*\b(online|digital|web|mobile)\b"
            ],
            "healthcare": [
                r"\b(patient|medical|clinical|healthcare|hospital|clinic)\b",
                r"\b(treatment|diagnosis|therapy|medication|procedure)\b",
                r"\b(HIPAA|FDA|compliance|regulation|safety)\b"
            ],
            "financial": [
                r"\b(bank|financial|investment|trading|insurance|loan)\b",
                r"\b(revenue|profit|cost|budget|ROI|margin)\b",
                r"\b(compliance|regulation|audit|risk|security)\b"
            ],
            "retail": [
                r"\b(customer|consumer|shopper|buyer|purchase)\b",
                r"\b(product|inventory|sales|marketing|promotion)\b",
                r"\b(store|shop|retail|e-commerce|online)\b"
            ],
            "manufacturing": [
                r"\b(production|manufacturing|assembly|quality|supply chain)\b",
                r"\b(equipment|machinery|process|efficiency|throughput)\b",
                r"\b(inventory|materials|components|parts)\b"
            ]
        }
    
    def _initialize_assumption_patterns(self) -> List[str]:
        """Initialize patterns for detecting assumptions in answers."""
        return [
            r"\b(obviously|clearly|of course|naturally|certainly)\b",
            r"\b(everyone knows|it's common|typical|standard|normal)\b",
            r"\b(should|must|need to|have to|required to)\b",
            r"\b(always|never|all|none|every|no one)\b",
            r"\b(we assume|assuming|given that|since)\b"
        ]
    
    def analyze_answers(self, answers: List[Answer]) -> AnalysisResult:
        """
        Analyze a collection of answers for complexity, vagueness, and contradictions.
        
        Args:
            answers: List of answers to analyze
            
        Returns:
            AnalysisResult: Comprehensive analysis results
        """
        if not answers:
            return AnalysisResult(
                complexity_level=ComplexityLevel.LOW,
                vagueness_level=VaguenessLevel.CLEAR
            )
        
        # Analyze each answer
        complexity_scores = []
        vagueness_scores = []
        all_assumptions = []
        all_domain_indicators = []
        
        for answer in answers:
            complexity = self._analyze_complexity(answer.response)
            vagueness = self._analyze_vagueness(answer.response)
            assumptions = self._detect_assumptions(answer.response)
            domain_indicators = self._detect_domain_context(answer.response)
            
            complexity_scores.append(complexity)
            vagueness_scores.append(vagueness)
            all_assumptions.extend(assumptions)
            all_domain_indicators.extend(domain_indicators)
        
        # Aggregate results
        overall_complexity = self._aggregate_complexity(complexity_scores)
        overall_vagueness = self._aggregate_vagueness(vagueness_scores)
        
        # Detect contradictions across answers
        contradictions = self._detect_contradictions(answers)
        
        # Determine if follow-up is needed
        requires_followup, followup_reasons = self._determine_followup_need(
            overall_complexity, overall_vagueness, contradictions, all_assumptions
        )
        
        return AnalysisResult(
            complexity_level=overall_complexity,
            vagueness_level=overall_vagueness,
            contradictions=contradictions,
            assumptions=list(set(all_assumptions)),  # Remove duplicates
            domain_indicators=list(set(all_domain_indicators)),
            requires_followup=requires_followup,
            followup_reasons=followup_reasons
        )
    
    def _analyze_complexity(self, text: str) -> ComplexityLevel:
        """Analyze the complexity level of a single answer."""
        text_lower = text.lower()
        
        # Count high complexity indicators
        high_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.complexity_patterns["high_complexity"]
        )
        
        # Count medium complexity indicators
        medium_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.complexity_patterns["medium_complexity"]
        )
        
        # Count technical indicators (add to complexity)
        tech_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.complexity_patterns["technical_indicators"]
        )
        
        # Length-based complexity (longer answers tend to be more complex)
        word_count = len(text.split())
        length_complexity = 0
        if word_count > 100:
            length_complexity = 2
        elif word_count > 50:
            length_complexity = 1
        
        # Calculate total complexity score
        total_score = high_count * 3 + medium_count * 2 + tech_count * 1 + length_complexity
        
        if total_score >= 5:
            return ComplexityLevel.HIGH
        elif total_score >= 2:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
    
    def _analyze_vagueness(self, text: str) -> VaguenessLevel:
        """Analyze the vagueness level of a single answer."""
        text_lower = text.lower()
        
        # Count very vague indicators
        very_vague_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.vagueness_patterns["very_vague"]
        )
        
        # Count somewhat vague indicators
        somewhat_vague_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.vagueness_patterns["somewhat_vague"]
        )
        
        # Count hedge words
        hedge_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in self.vagueness_patterns["hedge_words"]
        )
        
        # Check for specific numbers, dates, or concrete details
        specific_patterns = [
            r'\b\d+%\b',  # Percentages
            r'\b\$\d+\b',  # Dollar amounts
            r'\b\d{4}\b',  # Years
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',  # Months
            r'\b\d+\s+(days?|weeks?|months?|years?)\b'  # Time periods
        ]
        
        specific_count = sum(
            len(re.findall(pattern, text_lower))
            for pattern in specific_patterns
        )
        
        # Calculate vagueness score (higher = more vague)
        vagueness_score = very_vague_count * 3 + somewhat_vague_count * 2 + hedge_count * 1
        
        # Reduce score for specific details
        vagueness_score = max(0, vagueness_score - specific_count * 2)
        
        if vagueness_score >= 4:
            return VaguenessLevel.VERY_VAGUE
        elif vagueness_score >= 2:
            return VaguenessLevel.SOMEWHAT_VAGUE
        else:
            return VaguenessLevel.CLEAR
    
    def _detect_assumptions(self, text: str) -> List[str]:
        """Detect assumptions in the answer text."""
        assumptions = []
        text_lower = text.lower()
        
        for pattern in self.assumption_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                # Extract the sentence containing the assumption
                sentences = re.split(r'[.!?]+', text)
                for sentence in sentences:
                    if match.group() in sentence.lower():
                        assumptions.append(sentence.strip())
                        break
        
        return assumptions
    
    def _detect_domain_context(self, text: str) -> List[str]:
        """Detect domain-specific indicators in the answer text."""
        domain_indicators = []
        text_lower = text.lower()
        
        for domain, patterns in self.domain_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    domain_indicators.append(domain)
                    break  # Only add domain once
        
        return domain_indicators
    
    def _detect_contradictions(self, answers: List[Answer]) -> List[Contradiction]:
        """Detect contradictions across multiple answers."""
        contradictions = []
        
        if len(answers) < 2:
            return contradictions
        
        # Simple contradiction detection based on opposing keywords
        opposing_pairs = [
            (["increase", "grow", "expand", "more"], ["decrease", "reduce", "shrink", "less"]),
            (["urgent", "immediate", "critical"], ["not urgent", "low priority", "can wait"]),
            (["simple", "easy", "straightforward"], ["complex", "difficult", "complicated"]),
            (["yes", "definitely", "certainly"], ["no", "not", "never"]),
            (["high budget", "well funded"], ["low budget", "limited funds", "tight budget"])
        ]
        
        for i, answer1 in enumerate(answers):
            for j, answer2 in enumerate(answers[i+1:], i+1):
                text1_lower = answer1.response.lower()
                text2_lower = answer2.response.lower()
                
                for positive_terms, negative_terms in opposing_pairs:
                    has_positive_1 = any(term in text1_lower for term in positive_terms)
                    has_negative_1 = any(term in text1_lower for term in negative_terms)
                    has_positive_2 = any(term in text2_lower for term in positive_terms)
                    has_negative_2 = any(term in text2_lower for term in negative_terms)
                    
                    # Check for direct contradictions
                    if (has_positive_1 and has_negative_2) or (has_negative_1 and has_positive_2):
                        contradiction = Contradiction(
                            type=ContradictionType.DIRECT,
                            question_ids=[answer1.question_id, answer2.question_id],
                            description=f"Contradictory statements detected between responses",
                            severity="medium"
                        )
                        contradictions.append(contradiction)
        
        return contradictions
    
    def _aggregate_complexity(self, complexity_scores: List[ComplexityLevel]) -> ComplexityLevel:
        """Aggregate complexity scores across multiple answers."""
        if not complexity_scores:
            return ComplexityLevel.LOW
        
        # Count occurrences of each level
        high_count = sum(1 for c in complexity_scores if c == ComplexityLevel.HIGH)
        medium_count = sum(1 for c in complexity_scores if c == ComplexityLevel.MEDIUM)
        
        # If majority are high complexity, return high
        if high_count > len(complexity_scores) / 2:
            return ComplexityLevel.HIGH
        # If any high complexity or majority medium, return medium
        elif high_count > 0 or medium_count > len(complexity_scores) / 2:
            return ComplexityLevel.MEDIUM
        else:
            return ComplexityLevel.LOW
    
    def _aggregate_vagueness(self, vagueness_scores: List[VaguenessLevel]) -> VaguenessLevel:
        """Aggregate vagueness scores across multiple answers."""
        if not vagueness_scores:
            return VaguenessLevel.CLEAR
        
        # Count occurrences of each level
        very_vague_count = sum(1 for v in vagueness_scores if v == VaguenessLevel.VERY_VAGUE)
        somewhat_vague_count = sum(1 for v in vagueness_scores if v == VaguenessLevel.SOMEWHAT_VAGUE)
        
        # If majority are very vague, return very vague
        if very_vague_count > len(vagueness_scores) / 2:
            return VaguenessLevel.VERY_VAGUE
        # If any very vague or majority somewhat vague, return somewhat vague
        elif very_vague_count > 0 or somewhat_vague_count > len(vagueness_scores) / 2:
            return VaguenessLevel.SOMEWHAT_VAGUE
        else:
            return VaguenessLevel.CLEAR
    
    def _determine_followup_need(self, complexity: ComplexityLevel, vagueness: VaguenessLevel,
                               contradictions: List[Contradiction], assumptions: List[str]) -> tuple[bool, List[str]]:
        """Determine if follow-up questions are needed and why."""
        reasons = []
        
        if complexity == ComplexityLevel.HIGH:
            reasons.append("High complexity detected - need to break down into specific components")
        
        if vagueness in [VaguenessLevel.VERY_VAGUE, VaguenessLevel.SOMEWHAT_VAGUE]:
            reasons.append("Vague responses detected - need more specific details")
        
        if contradictions:
            reasons.append("Contradictions detected - need clarification")
        
        if assumptions:
            reasons.append("Assumptions detected - need validation")
        
        return len(reasons) > 0, reasons
    
    def generate_followups(self, analysis: AnalysisResult, original_question: Question,
                          answered_questions: List[str] = None) -> List[Question]:
        """
        Generate follow-up questions based on analysis results.
        
        Args:
            analysis: Results from answer analysis
            original_question: The original question that was answered
            answered_questions: List of already answered question IDs
            
        Returns:
            List of follow-up questions
        """
        if answered_questions is None:
            answered_questions = []
        
        followup_questions = []
        
        # Generate questions based on complexity
        if analysis.complexity_level == ComplexityLevel.HIGH:
            followup_questions.extend(
                self._generate_complexity_followups(original_question, answered_questions)
            )
        
        # Generate questions based on vagueness
        if analysis.vagueness_level in [VaguenessLevel.VERY_VAGUE, VaguenessLevel.SOMEWHAT_VAGUE]:
            followup_questions.extend(
                self._generate_vagueness_followups(original_question, answered_questions)
            )
        
        # Generate questions for contradictions
        if analysis.contradictions:
            followup_questions.extend(
                self._generate_contradiction_followups(analysis.contradictions, answered_questions)
            )
        
        # Generate questions for assumptions
        if analysis.assumptions:
            followup_questions.extend(
                self._generate_assumption_followups(analysis.assumptions, original_question, answered_questions)
            )
        
        # Generate domain-specific questions
        if analysis.domain_indicators:
            followup_questions.extend(
                self._generate_domain_followups(analysis.domain_indicators, original_question, answered_questions)
            )
        
        # Remove duplicates and filter out already answered questions
        unique_questions = []
        seen_texts = set()
        
        for question in followup_questions:
            if question.id not in answered_questions and question.text not in seen_texts:
                unique_questions.append(question)
                seen_texts.add(question.text)
        
        return unique_questions
    
    def _generate_complexity_followups(self, original_question: Question, 
                                     answered_questions: List[str]) -> List[Question]:
        """Generate follow-up questions for high complexity answers."""
        followups = []
        
        complexity_templates = {
            Dimension.PROBLEM_DEFINITION: [
                "Can you break down this problem into 2-3 specific sub-problems?",
                "Which aspect of this problem should be addressed first?",
                "What are the most critical components of this problem?"
            ],
            Dimension.BUSINESS_OBJECTIVES: [
                "Can you prioritize these objectives from most to least important?",
                "Which objective would have the biggest business impact?",
                "Are there dependencies between these different objectives?"
            ],
            Dimension.STAKEHOLDERS: [
                "Who among these stakeholders has the most influence on success?",
                "Which stakeholder group is most likely to be affected by changes?",
                "Are there conflicts of interest between different stakeholder groups?"
            ],
            Dimension.CURRENT_SITUATION: [
                "Which current approach has been most/least effective?",
                "What are the top 2-3 pain points in the current situation?",
                "Which aspect of the current situation is most urgent to address?"
            ],
            Dimension.CONSTRAINTS: [
                "Which constraint is most likely to impact the project?",
                "Are any of these constraints negotiable or flexible?",
                "How do these constraints interact with each other?"
            ],
            Dimension.SUCCESS_CRITERIA: [
                "Which success metric is most important to track?",
                "How will you measure progress toward these different criteria?",
                "What would be the minimum acceptable level for each criterion?"
            ],
            Dimension.BUSINESS_DOMAIN: [
                "Which industry factor has the biggest impact on your business?",
                "How do these different market dynamics affect your priorities?",
                "Which domain-specific challenges are most critical?"
            ],
            Dimension.IMPLEMENTATION: [
                "Which implementation approach would you prefer and why?",
                "What are the biggest risks for each implementation option?",
                "Which barrier to implementation concerns you most?"
            ]
        }
        
        templates = complexity_templates.get(original_question.dimension, [
            "Can you provide more specific details about the most important aspects?",
            "Which component of your answer is most critical to address first?",
            "How would you prioritize the different elements you mentioned?"
        ])
        
        for i, template in enumerate(templates[:2]):  # Limit to 2 follow-ups
            question_id = f"{original_question.id}_complexity_followup_{i+1}"
            if question_id not in answered_questions:
                followup = Question(
                    id=question_id,
                    dimension=original_question.dimension,
                    text=template,
                    reasoning="Following up on complex answer to get more specific details",
                    question_type=QuestionType.FOLLOWUP,
                    dependencies=[original_question.id]
                )
                followups.append(followup)
        
        return followups
    
    def _generate_vagueness_followups(self, original_question: Question,
                                    answered_questions: List[str]) -> List[Question]:
        """Generate follow-up questions for vague answers."""
        followups = []
        
        vagueness_templates = {
            Dimension.PROBLEM_DEFINITION: [
                "Can you provide a specific example of this problem occurring?",
                "What exactly triggers this problem - can you describe a recent instance?"
            ],
            Dimension.BUSINESS_OBJECTIVES: [
                "Can you quantify this objective with specific numbers or percentages?",
                "What would success look like in concrete, measurable terms?"
            ],
            Dimension.STAKEHOLDERS: [
                "Can you name the specific roles or departments involved?",
                "Who exactly would be the primary contact person for this initiative?"
            ],
            Dimension.CURRENT_SITUATION: [
                "Can you provide specific metrics or data about the current state?",
                "What exactly happens in the current process - can you walk through it step by step?"
            ],
            Dimension.CONSTRAINTS: [
                "Can you provide specific numbers for budget, timeline, or resource limits?",
                "What are the exact regulatory requirements or compliance standards?"
            ],
            Dimension.SUCCESS_CRITERIA: [
                "What specific metrics or KPIs would you use to measure this?",
                "Can you define exactly what 'improvement' means with concrete numbers?"
            ],
            Dimension.BUSINESS_DOMAIN: [
                "Which specific industry regulations or standards apply to your business?",
                "Can you describe your exact market position or competitive landscape?"
            ],
            Dimension.IMPLEMENTATION: [
                "What specific systems or processes would need to be changed?",
                "Who exactly would be responsible for each aspect of implementation?"
            ]
        }
        
        templates = vagueness_templates.get(original_question.dimension, [
            "Can you provide more specific details or examples?",
            "What exactly do you mean by that - can you be more concrete?"
        ])
        
        for i, template in enumerate(templates[:2]):  # Limit to 2 follow-ups
            question_id = f"{original_question.id}_vagueness_followup_{i+1}"
            if question_id not in answered_questions:
                followup = Question(
                    id=question_id,
                    dimension=original_question.dimension,
                    text=template,
                    reasoning="Following up on vague answer to get specific details",
                    question_type=QuestionType.CLARIFICATION,
                    dependencies=[original_question.id]
                )
                followups.append(followup)
        
        return followups
    
    def _generate_contradiction_followups(self, contradictions: List[Contradiction],
                                        answered_questions: List[str]) -> List[Question]:
        """Generate follow-up questions for detected contradictions."""
        followups = []
        
        for i, contradiction in enumerate(contradictions[:2]):  # Limit to 2 contradictions
            question_id = f"contradiction_clarification_{i+1}"
            if question_id not in answered_questions:
                followup = Question(
                    id=question_id,
                    dimension=Dimension.PROBLEM_DEFINITION,  # Default dimension
                    text=f"I noticed some potentially conflicting information in your previous answers. Can you help clarify: {contradiction.description}?",
                    reasoning="Resolving detected contradictions to ensure consistent understanding",
                    question_type=QuestionType.CLARIFICATION,
                    dependencies=contradiction.question_ids
                )
                followups.append(followup)
        
        return followups
    
    def _generate_assumption_followups(self, assumptions: List[str], original_question: Question,
                                     answered_questions: List[str]) -> List[Question]:
        """Generate follow-up questions to validate assumptions."""
        followups = []
        
        for i, assumption in enumerate(assumptions[:2]):  # Limit to 2 assumptions
            question_id = f"{original_question.id}_assumption_validation_{i+1}"
            if question_id not in answered_questions:
                # Clean up the assumption text for the question
                clean_assumption = assumption.strip()
                if len(clean_assumption) > 100:
                    clean_assumption = clean_assumption[:100] + "..."
                
                followup = Question(
                    id=question_id,
                    dimension=original_question.dimension,
                    text=f"You mentioned: '{clean_assumption}'. Can you validate this assumption - is this always true in your context?",
                    reasoning="Validating assumptions to ensure they are explicitly confirmed",
                    question_type=QuestionType.CLARIFICATION,
                    dependencies=[original_question.id]
                )
                followups.append(followup)
        
        return followups
    
    def _generate_domain_followups(self, domain_indicators: List[str], original_question: Question,
                                 answered_questions: List[str]) -> List[Question]:
        """Generate domain-specific follow-up questions."""
        followups = []
        
        domain_specific_questions = {
            "technology": [
                "What specific technologies or platforms are you currently using?",
                "Are there any technical integration challenges we should consider?"
            ],
            "healthcare": [
                "What specific compliance requirements (HIPAA, FDA, etc.) apply?",
                "How do patient safety and privacy concerns affect this initiative?"
            ],
            "financial": [
                "What specific financial regulations or compliance requirements apply?",
                "How do risk management and security concerns factor into this?"
            ],
            "retail": [
                "How do seasonal patterns or customer behavior trends affect this?",
                "What specific customer segments or channels are most important?"
            ],
            "manufacturing": [
                "What specific production processes or quality standards are involved?",
                "How do supply chain or inventory considerations factor in?"
            ]
        }
        
        for domain in domain_indicators[:2]:  # Limit to 2 domains
            if domain in domain_specific_questions:
                templates = domain_specific_questions[domain]
                for i, template in enumerate(templates[:1]):  # 1 question per domain
                    question_id = f"{original_question.id}_domain_{domain}_{i+1}"
                    if question_id not in answered_questions:
                        followup = Question(
                            id=question_id,
                            dimension=original_question.dimension,
                            text=template,
                            reasoning=f"Domain-specific follow-up for {domain} context",
                            question_type=QuestionType.FOLLOWUP,
                            dependencies=[original_question.id]
                        )
                        followups.append(followup)
        
        return followups
    
    def detect_contradictions(self, answers: List[Answer]) -> List[Contradiction]:
        """
        Public method to detect contradictions across answers.
        
        Args:
            answers: List of answers to analyze for contradictions
            
        Returns:
            List of detected contradictions
        """
        return self._detect_contradictions(answers)
    
    def customize_questions_for_domain(self, questions: List[Question], 
                                     domain_context: DomainContext) -> List[Question]:
        """
        Customize questions based on detected domain context.
        
        Args:
            questions: Original questions to customize
            domain_context: Detected domain context
            
        Returns:
            List of customized questions
        """
        customized_questions = []
        
        for question in questions:
            customized_question = self._customize_single_question(question, domain_context)
            customized_questions.append(customized_question)
        
        return customized_questions
    
    def _customize_single_question(self, question: Question, 
                                 domain_context: DomainContext) -> Question:
        """Customize a single question based on domain context."""
        # Create a copy of the question
        customized = Question(
            id=question.id,
            dimension=question.dimension,
            text=question.text,
            reasoning=question.reasoning,
            question_type=question.question_type,
            dependencies=question.dependencies.copy(),
            options=question.options.copy() if question.options else None
        )
        
        # Customize based on industry
        if domain_context.industry:
            industry_customizations = {
                "technology": {
                    "stakeholders": "users, developers, product managers, and technical teams",
                    "constraints": "technical limitations, scalability requirements, and development resources",
                    "success_criteria": "user adoption, system performance, and technical metrics"
                },
                "healthcare": {
                    "stakeholders": "patients, healthcare providers, administrators, and regulatory bodies",
                    "constraints": "HIPAA compliance, patient safety requirements, and clinical workflows",
                    "success_criteria": "patient outcomes, clinical efficiency, and regulatory compliance"
                },
                "financial": {
                    "stakeholders": "customers, regulators, risk managers, and compliance teams",
                    "constraints": "regulatory requirements, risk tolerance, and security standards",
                    "success_criteria": "financial performance, risk reduction, and regulatory compliance"
                }
            }
            
            if domain_context.industry in industry_customizations:
                customizations = industry_customizations[domain_context.industry]
                
                # Apply customizations to question text
                for key, value in customizations.items():
                    if key in question.dimension.value:
                        # Add industry-specific context to the question
                        customized.text = f"{question.text} (Consider {value} in your response.)"
                        customized.reasoning = f"{question.reasoning} This is particularly important in the {domain_context.industry} industry."
        
        return customized