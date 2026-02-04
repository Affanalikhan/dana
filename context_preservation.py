"""
Context Preservation System for CRISP-DM Business Understanding Specialist

This module implements answer history tracking, conversational continuity,
conflict detection, and detail incorporation from previous responses.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import re
from collections import defaultdict

from crisp_dm_framework import Question, Answer, Dimension
from adaptive_engine import Contradiction, ContradictionType


@dataclass
class ContextReference:
    """Represents a reference to previous context."""
    referenced_question_id: str
    referenced_answer_excerpt: str
    reference_type: str  # "continuation", "clarification", "contradiction", "elaboration"
    confidence: float  # 0.0 to 1.0


@dataclass
class ConversationalContext:
    """Maintains conversational context across questions and dimensions."""
    session_id: str
    answer_history: List[Answer] = field(default_factory=list)
    dimension_summaries: Dict[Dimension, str] = field(default_factory=dict)
    key_themes: List[str] = field(default_factory=list)
    stakeholder_mentions: List[str] = field(default_factory=list)
    timeline_references: List[str] = field(default_factory=list)
    constraint_mentions: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)


class ContextPreservationEngine:
    """
    Manages context preservation across the CRISP-DM questioning session.
    Provides answer history tracking, conversational continuity, and conflict detection.
    """
    
    def __init__(self):
        """Initialize the context preservation engine."""
        self.contexts: Dict[str, ConversationalContext] = {}
        self.reference_patterns = self._initialize_reference_patterns()
        self.theme_extractors = self._initialize_theme_extractors()
    
    def _initialize_reference_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for detecting contextual references."""
        return {
            "continuation": [
                r"\b(also|additionally|furthermore|moreover|in addition)\b",
                r"\b(building on|expanding on|following up on)\b",
                r"\b(similarly|likewise|in the same way)\b"
            ],
            "clarification": [
                r"\b(specifically|to clarify|more precisely|what I mean is)\b",
                r"\b(in other words|that is to say|to be specific)\b",
                r"\b(let me explain|to elaborate|more detail)\b"
            ],
            "contradiction": [
                r"\b(however|but|although|despite|on the other hand)\b",
                r"\b(actually|in fact|contrary to|unlike)\b",
                r"\b(instead|rather than|different from)\b"
            ],
            "elaboration": [
                r"\b(for example|such as|including|like)\b",
                r"\b(particularly|especially|notably|mainly)\b",
                r"\b(details|specifics|breakdown|components)\b"
            ]
        }
    
    def _initialize_theme_extractors(self) -> Dict[str, List[str]]:
        """Initialize patterns for extracting key themes from answers."""
        return {
            "business_goals": [
                r"\b(revenue|profit|growth|efficiency|productivity)\b",
                r"\b(customer|client|user|satisfaction|retention)\b",
                r"\b(market|competitive|advantage|position)\b"
            ],
            "challenges": [
                r"\b(problem|issue|challenge|difficulty|obstacle)\b",
                r"\b(bottleneck|constraint|limitation|barrier)\b",
                r"\b(risk|concern|worry|threat)\b"
            ],
            "stakeholders": [
                r"\b(team|department|manager|executive|leadership)\b",
                r"\b(customer|client|user|vendor|partner)\b",
                r"\b(stakeholder|decision maker|influencer)\b"
            ],
            "timeline": [
                r"\b(\d+\s+(days?|weeks?|months?|years?))\b",
                r"\b(immediately|soon|later|eventually|ongoing)\b",
                r"\b(deadline|timeline|schedule|urgent|priority)\b"
            ],
            "resources": [
                r"\b(budget|cost|funding|investment|resource)\b",
                r"\b(staff|personnel|team|people|expertise)\b",
                r"\b(technology|system|tool|platform|infrastructure)\b"
            ]
        }
    
    def create_context(self, session_id: str) -> ConversationalContext:
        """Create a new conversational context for a session."""
        context = ConversationalContext(session_id=session_id)
        self.contexts[session_id] = context
        return context
    
    def get_context(self, session_id: str) -> Optional[ConversationalContext]:
        """Get the conversational context for a session."""
        return self.contexts.get(session_id)
    
    def add_answer_to_context(self, session_id: str, answer: Answer, 
                            question: Question) -> bool:
        """
        Add an answer to the conversational context and update tracking.
        
        Args:
            session_id: Session identifier
            answer: The answer to add
            question: The question that was answered
            
        Returns:
            bool: True if successfully added
        """
        context = self.get_context(session_id)
        if not context:
            context = self.create_context(session_id)
        
        # Add answer to history
        context.answer_history.append(answer)
        
        # Extract and update themes
        themes = self._extract_themes(answer.response)
        for theme in themes:
            if theme not in context.key_themes:
                context.key_themes.append(theme)
        
        # Update dimension summary
        self._update_dimension_summary(context, question.dimension, answer.response)
        
        # Extract specific mentions
        self._extract_stakeholder_mentions(context, answer.response)
        self._extract_timeline_references(context, answer.response)
        self._extract_constraint_mentions(context, answer.response)
        
        context.last_updated = datetime.now()
        return True
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract key themes from answer text."""
        themes = []
        text_lower = text.lower()
        
        for theme_category, patterns in self.theme_extractors.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    themes.append(theme_category)
                    break  # Only add theme category once
        
        return themes
    
    def _update_dimension_summary(self, context: ConversationalContext, 
                                dimension: Dimension, answer_text: str) -> None:
        """Update the summary for a specific dimension."""
        if dimension not in context.dimension_summaries:
            context.dimension_summaries[dimension] = ""
        
        # Extract key points from the answer (first 200 characters as summary)
        summary_excerpt = answer_text.strip()
        if len(summary_excerpt) > 200:
            summary_excerpt = summary_excerpt[:200] + "..."
        
        # Append to existing summary
        if context.dimension_summaries[dimension]:
            context.dimension_summaries[dimension] += f" | {summary_excerpt}"
        else:
            context.dimension_summaries[dimension] = summary_excerpt
    
    def _extract_stakeholder_mentions(self, context: ConversationalContext, 
                                    text: str) -> None:
        """Extract stakeholder mentions from answer text."""
        stakeholder_patterns = [
            r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b",  # Names like "John Smith"
            r"\b(CEO|CTO|CFO|VP|Director|Manager)\b",  # Titles
            r"\b([A-Z][a-z]+\s+(team|department|group))\b"  # Teams
        ]
        
        for pattern in stakeholder_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                stakeholder = match if isinstance(match, str) else match[0]
                if stakeholder.lower() not in [s.lower() for s in context.stakeholder_mentions]:
                    context.stakeholder_mentions.append(stakeholder)
    
    def _extract_timeline_references(self, context: ConversationalContext, 
                                   text: str) -> None:
        """Extract timeline references from answer text."""
        timeline_patterns = [
            r"\b(\d+\s+(days?|weeks?|months?|years?))\b",
            r"\b(Q[1-4]\s+\d{4})\b",  # Quarters
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b"
        ]
        
        for pattern in timeline_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                timeline = match if isinstance(match, str) else match[0]
                if timeline not in context.timeline_references:
                    context.timeline_references.append(timeline)
    
    def _extract_constraint_mentions(self, context: ConversationalContext, 
                                   text: str) -> None:
        """Extract constraint mentions from answer text."""
        constraint_patterns = [
            r"\$[\d,]+",  # Dollar amounts
            r"\b(\d+%)\b",  # Percentages
            r"\b(limited|restricted|constrained|budget|deadline)\b"
        ]
        
        for pattern in constraint_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if match not in context.constraint_mentions:
                    context.constraint_mentions.append(match)
    
    def generate_context_bridge(self, session_id: str, current_dimension: Dimension,
                               previous_dimension: Optional[Dimension] = None) -> str:
        """
        Generate a context bridge that connects current questions to previous context.
        
        Args:
            session_id: Session identifier
            current_dimension: The dimension we're moving to
            previous_dimension: The dimension we're coming from
            
        Returns:
            str: Context bridge text that references previous answers
        """
        context = self.get_context(session_id)
        if not context or not context.answer_history:
            return f"Let's explore {current_dimension.value.replace('_', ' ')} for your business question."
        
        # Get relevant context from previous answers
        relevant_context = self._get_relevant_context_for_dimension(context, current_dimension)
        
        if previous_dimension and previous_dimension in context.dimension_summaries:
            previous_summary = context.dimension_summaries[previous_dimension]
            bridge = f"Based on what you've shared about {previous_dimension.value.replace('_', ' ')}"
            
            if relevant_context:
                bridge += f" - particularly {relevant_context} - "
            else:
                bridge += ", "
            
            bridge += f"let's now explore {current_dimension.value.replace('_', ' ')}."
        else:
            bridge = f"Building on your previous responses"
            if relevant_context:
                bridge += f" about {relevant_context}"
            bridge += f", let's examine {current_dimension.value.replace('_', ' ')}."
        
        return bridge
    
    def _get_relevant_context_for_dimension(self, context: ConversationalContext,
                                          dimension: Dimension) -> str:
        """Get relevant context from previous answers for the current dimension."""
        if not context.answer_history:
            return ""
        
        # Map dimensions to relevant context elements
        dimension_context_map = {
            Dimension.BUSINESS_OBJECTIVES: ["business_goals", "challenges"],
            Dimension.STAKEHOLDERS: ["stakeholders"],
            Dimension.CURRENT_SITUATION: ["challenges", "resources"],
            Dimension.CONSTRAINTS: ["resources", "timeline"],
            Dimension.SUCCESS_CRITERIA: ["business_goals"],
            Dimension.BUSINESS_DOMAIN: ["business_goals", "challenges"],
            Dimension.IMPLEMENTATION: ["stakeholders", "resources", "timeline"]
        }
        
        relevant_themes = dimension_context_map.get(dimension, [])
        found_themes = [theme for theme in context.key_themes if theme in relevant_themes]
        
        if found_themes:
            return found_themes[0].replace('_', ' ')
        
        return ""
    
    def generate_contextual_question(self, base_question: Question, session_id: str) -> Question:
        """
        Generate a contextual version of a question that references previous answers.
        
        Args:
            base_question: The original question
            session_id: Session identifier
            
        Returns:
            Question: Enhanced question with contextual references
        """
        context = self.get_context(session_id)
        if not context or not context.answer_history:
            return base_question
        
        # Find relevant previous answers
        references = self._find_relevant_references(context, base_question)
        
        if not references:
            return base_question
        
        # Create enhanced question with context
        enhanced_text = base_question.text
        reference = references[0]  # Use the most relevant reference
        
        # Add contextual reference to the question
        if reference.reference_type == "continuation":
            enhanced_text = f"Building on your earlier mention of '{reference.referenced_answer_excerpt}', {base_question.text.lower()}"
        elif reference.reference_type == "clarification":
            enhanced_text = f"You mentioned '{reference.referenced_answer_excerpt}'. To clarify this further, {base_question.text.lower()}"
        elif reference.reference_type == "elaboration":
            enhanced_text = f"Earlier you indicated '{reference.referenced_answer_excerpt}'. Can you elaborate: {base_question.text.lower()}"
        else:
            enhanced_text = f"Considering your previous response about '{reference.referenced_answer_excerpt}', {base_question.text.lower()}"
        
        # Create new question with enhanced text
        contextual_question = Question(
            id=base_question.id,
            dimension=base_question.dimension,
            text=enhanced_text,
            reasoning=f"{base_question.reasoning} (Enhanced with context from previous answers)",
            question_type=base_question.question_type,
            dependencies=base_question.dependencies + [reference.referenced_question_id],
            options=base_question.options
        )
        
        return contextual_question
    
    def _find_relevant_references(self, context: ConversationalContext, 
                                question: Question) -> List[ContextReference]:
        """Find relevant references from previous answers for the current question."""
        references = []
        
        # Look through recent answers (last 5) for relevant context
        recent_answers = context.answer_history[-5:] if len(context.answer_history) > 5 else context.answer_history
        
        for answer in recent_answers:
            # Extract key phrases from the answer (first sentence or up to 50 chars)
            answer_excerpt = self._extract_key_excerpt(answer.response)
            
            # Determine reference type based on question dimension and answer content
            reference_type = self._determine_reference_type(question, answer.response)
            
            # Calculate relevance confidence
            confidence = self._calculate_reference_confidence(question, answer.response)
            
            if confidence > 0.3:  # Only include references with reasonable confidence
                reference = ContextReference(
                    referenced_question_id=answer.question_id,
                    referenced_answer_excerpt=answer_excerpt,
                    reference_type=reference_type,
                    confidence=confidence
                )
                references.append(reference)
        
        # Sort by confidence and return top references
        references.sort(key=lambda r: r.confidence, reverse=True)
        return references[:2]  # Return top 2 references
    
    def _extract_key_excerpt(self, text: str) -> str:
        """Extract a key excerpt from answer text for referencing."""
        # Get first sentence or first 50 characters
        sentences = re.split(r'[.!?]+', text.strip())
        if sentences and len(sentences[0]) > 0:
            excerpt = sentences[0].strip()
            if len(excerpt) > 50:
                excerpt = excerpt[:50] + "..."
            return excerpt
        
        # Fallback to first 50 characters
        excerpt = text.strip()[:50]
        if len(text.strip()) > 50:
            excerpt += "..."
        return excerpt
    
    def _determine_reference_type(self, question: Question, answer_text: str) -> str:
        """Determine the type of reference based on question and answer content."""
        text_lower = answer_text.lower()
        
        # Check for different reference patterns
        for ref_type, patterns in self.reference_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return ref_type
        
        # Default based on question dimension relationships
        dimension_relationships = {
            Dimension.BUSINESS_OBJECTIVES: "continuation",
            Dimension.STAKEHOLDERS: "elaboration", 
            Dimension.CURRENT_SITUATION: "clarification",
            Dimension.CONSTRAINTS: "continuation",
            Dimension.SUCCESS_CRITERIA: "elaboration",
            Dimension.BUSINESS_DOMAIN: "clarification",
            Dimension.IMPLEMENTATION: "continuation"
        }
        
        return dimension_relationships.get(question.dimension, "continuation")
    
    def _calculate_reference_confidence(self, question: Question, answer_text: str) -> float:
        """Calculate confidence score for a potential reference."""
        confidence = 0.0
        text_lower = answer_text.lower()
        question_lower = question.text.lower()
        
        # Keyword overlap between question and answer
        question_words = set(re.findall(r'\b\w+\b', question_lower))
        answer_words = set(re.findall(r'\b\w+\b', text_lower))
        
        # Remove common words
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'what', 'how', 'when', 'where', 'why', 'who'}
        question_words -= common_words
        answer_words -= common_words
        
        if question_words and answer_words:
            overlap = len(question_words.intersection(answer_words))
            confidence += (overlap / len(question_words)) * 0.5
        
        # Dimension relevance
        if question.dimension in [Dimension.STAKEHOLDERS, Dimension.IMPLEMENTATION]:
            # These dimensions often reference previous context
            confidence += 0.3
        
        # Answer length (longer answers more likely to contain relevant context)
        if len(answer_text.split()) > 20:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def detect_context_conflicts(self, session_id: str) -> List[Contradiction]:
        """
        Detect conflicts in context across the conversation.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of detected contradictions with context citations
        """
        context = self.get_context(session_id)
        if not context or len(context.answer_history) < 2:
            return []
        
        contradictions = []
        
        # Check for timeline conflicts
        timeline_conflicts = self._detect_timeline_conflicts(context)
        contradictions.extend(timeline_conflicts)
        
        # Check for stakeholder conflicts
        stakeholder_conflicts = self._detect_stakeholder_conflicts(context)
        contradictions.extend(stakeholder_conflicts)
        
        # Check for constraint conflicts
        constraint_conflicts = self._detect_constraint_conflicts(context)
        contradictions.extend(constraint_conflicts)
        
        return contradictions
    
    def _detect_timeline_conflicts(self, context: ConversationalContext) -> List[Contradiction]:
        """Detect conflicts in timeline references."""
        conflicts = []
        
        if len(context.timeline_references) < 2:
            return conflicts
        
        # Look for conflicting timeline statements
        urgent_patterns = r"\b(immediate|urgent|asap|critical|emergency)\b"
        long_term_patterns = r"\b(long.term|strategic|future|eventually|someday)\b"
        
        urgent_answers = []
        long_term_answers = []
        
        for answer in context.answer_history:
            if re.search(urgent_patterns, answer.response, re.IGNORECASE):
                urgent_answers.append(answer)
            if re.search(long_term_patterns, answer.response, re.IGNORECASE):
                long_term_answers.append(answer)
        
        if urgent_answers and long_term_answers:
            conflict = Contradiction(
                type=ContradictionType.TEMPORAL,
                question_ids=[urgent_answers[0].question_id, long_term_answers[0].question_id],
                description="Conflicting timeline expectations: some responses indicate urgency while others suggest long-term planning",
                severity="medium"
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_stakeholder_conflicts(self, context: ConversationalContext) -> List[Contradiction]:
        """Detect conflicts in stakeholder information."""
        conflicts = []
        
        # Look for conflicting authority statements
        authority_patterns = [
            (r"\b(I|we)\s+(decide|control|manage|own)\b", "high_authority"),
            (r"\b(need\s+approval|must\s+ask|check\s+with)\b", "low_authority")
        ]
        
        authority_answers = {"high_authority": [], "low_authority": []}
        
        for answer in context.answer_history:
            for pattern, authority_type in authority_patterns:
                if re.search(pattern, answer.response, re.IGNORECASE):
                    authority_answers[authority_type].append(answer)
        
        if authority_answers["high_authority"] and authority_answers["low_authority"]:
            conflict = Contradiction(
                type=ContradictionType.IMPLICIT,
                question_ids=[
                    authority_answers["high_authority"][0].question_id,
                    authority_answers["low_authority"][0].question_id
                ],
                description="Conflicting authority levels: some responses suggest decision-making authority while others indicate need for approval",
                severity="medium"
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def _detect_constraint_conflicts(self, context: ConversationalContext) -> List[Contradiction]:
        """Detect conflicts in constraint information."""
        conflicts = []
        
        # Look for budget conflicts
        budget_high_patterns = r"\b(unlimited|generous|substantial|large)\s+(budget|funding|resources)\b"
        budget_low_patterns = r"\b(limited|tight|small|constrained)\s+(budget|funding|resources)\b"
        
        high_budget_answers = []
        low_budget_answers = []
        
        for answer in context.answer_history:
            if re.search(budget_high_patterns, answer.response, re.IGNORECASE):
                high_budget_answers.append(answer)
            if re.search(budget_low_patterns, answer.response, re.IGNORECASE):
                low_budget_answers.append(answer)
        
        if high_budget_answers and low_budget_answers:
            conflict = Contradiction(
                type=ContradictionType.DIRECT,
                question_ids=[high_budget_answers[0].question_id, low_budget_answers[0].question_id],
                description="Conflicting budget information: some responses indicate substantial resources while others suggest constraints",
                severity="high"
            )
            conflicts.append(conflict)
        
        return conflicts
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the conversation context.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Dict containing conversation summary and context
        """
        context = self.get_context(session_id)
        if not context:
            return {"error": "No context found for session"}
        
        return {
            "session_id": session_id,
            "total_answers": len(context.answer_history),
            "dimensions_covered": list(context.dimension_summaries.keys()),
            "key_themes": context.key_themes,
            "stakeholder_mentions": context.stakeholder_mentions,
            "timeline_references": context.timeline_references,
            "constraint_mentions": context.constraint_mentions,
            "dimension_summaries": {
                dim.value: summary for dim, summary in context.dimension_summaries.items()
            },
            "last_updated": context.last_updated.isoformat()
        }
    
    def export_context(self, session_id: str) -> Dict[str, Any]:
        """Export context data for persistence."""
        context = self.get_context(session_id)
        if not context:
            return {"error": "No context found for session"}
        
        return {
            "session_id": context.session_id,
            "answer_history": [
                {
                    "question_id": answer.question_id,
                    "response": answer.response,
                    "timestamp": answer.timestamp,
                    "confidence_level": answer.confidence_level,
                    "requires_followup": answer.requires_followup
                }
                for answer in context.answer_history
            ],
            "dimension_summaries": {
                dim.value: summary for dim, summary in context.dimension_summaries.items()
            },
            "key_themes": context.key_themes,
            "stakeholder_mentions": context.stakeholder_mentions,
            "timeline_references": context.timeline_references,
            "constraint_mentions": context.constraint_mentions,
            "last_updated": context.last_updated.isoformat()
        }
    
    def import_context(self, context_data: Dict[str, Any]) -> bool:
        """Import context data from exported format."""
        try:
            session_id = context_data["session_id"]
            
            # Reconstruct context
            context = ConversationalContext(session_id=session_id)
            
            # Reconstruct answer history
            for answer_data in context_data["answer_history"]:
                answer = Answer(
                    question_id=answer_data["question_id"],
                    response=answer_data["response"],
                    timestamp=answer_data["timestamp"],
                    confidence_level=answer_data["confidence_level"],
                    requires_followup=answer_data["requires_followup"]
                )
                context.answer_history.append(answer)
            
            # Reconstruct dimension summaries
            for dim_value, summary in context_data["dimension_summaries"].items():
                dimension = Dimension(dim_value)
                context.dimension_summaries[dimension] = summary
            
            # Restore other context data
            context.key_themes = context_data["key_themes"]
            context.stakeholder_mentions = context_data["stakeholder_mentions"]
            context.timeline_references = context_data["timeline_references"]
            context.constraint_mentions = context_data["constraint_mentions"]
            context.last_updated = datetime.fromisoformat(context_data["last_updated"])
            
            # Store context
            self.contexts[session_id] = context
            
            return True
            
        except (KeyError, ValueError, TypeError):
            return False