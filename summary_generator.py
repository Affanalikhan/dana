"""
Summary Generation System for CRISP-DM Business Understanding Specialist

This module implements comprehensive Business Understanding document generation
with structured summaries, gap highlighting, and downloadable report formatting.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from datetime import datetime
import json
import uuid

from crisp_dm_framework import Dimension, Question, Answer
from session_manager import SessionState
from analysis_engine import AnalysisEngine, Gap, Assumption, BusinessInsights, RiskLevel


@dataclass
class Stakeholder:
    """Represents a stakeholder in the business summary."""
    name: str
    role: str
    influence_level: str  # "high", "medium", "low"
    involvement: str  # "decision_maker", "end_user", "influencer", "observer"
    concerns: List[str] = field(default_factory=list)


@dataclass
class Constraint:
    """Represents a constraint in the business summary."""
    type: str  # "budget", "timeline", "resource", "regulatory", "technical"
    description: str
    impact_level: str  # "high", "medium", "low"
    mitigation_strategies: List[str] = field(default_factory=list)


@dataclass
class Metric:
    """Represents a success metric in the business summary."""
    name: str
    description: str
    target_value: Optional[str] = None
    measurement_method: str = ""
    frequency: str = ""  # "daily", "weekly", "monthly", "quarterly"
    owner: str = ""


@dataclass
class RiskAssumption:
    """Represents a risk or assumption in the business summary."""
    id: str
    type: str  # "risk", "assumption"
    category: str
    description: str
    impact_level: RiskLevel
    likelihood: str = "unknown"  # "high", "medium", "low", "unknown"
    mitigation_plan: str = ""
    validation_required: bool = False


@dataclass
class BusinessSummary:
    """
    Comprehensive Business Understanding Summary containing all required sections
    as specified in Requirements 5.1.
    """
    session_id: str
    business_question: str
    
    # Required sections per Requirements 5.1
    executive_summary: str
    business_objectives: List[str]
    current_situation: str
    stakeholders: List[Stakeholder]
    constraints: List[Constraint]
    success_metrics: List[Metric]
    risks_assumptions: List[RiskAssumption]
    recommended_next_steps: List[str]
    
    # Additional metadata
    identified_gaps: List[Gap] = field(default_factory=list)
    business_insights: Optional[BusinessInsights] = None
    generation_timestamp: datetime = field(default_factory=datetime.now)
    completeness_score: float = 0.0  # 0.0 to 1.0
    readiness_assessment: str = "unknown"
    
    def __post_init__(self):
        """Validate that all required sections are present."""
        required_sections = [
            'executive_summary', 'business_objectives', 'current_situation',
            'stakeholders', 'constraints', 'success_metrics', 
            'risks_assumptions', 'recommended_next_steps'
        ]
        
        for section in required_sections:
            value = getattr(self, section)
            if value is None or (isinstance(value, (list, str)) and len(value) == 0):
                raise ValueError(f"Required section '{section}' cannot be empty")


class SummaryGenerator:
    """
    Creates comprehensive Business Understanding documents from session data.
    Implements the interface specified in the design document.
    """
    
    def __init__(self, analysis_engine: Optional[AnalysisEngine] = None):
        """Initialize the summary generator with analysis capabilities."""
        self.analysis_engine = analysis_engine or AnalysisEngine()
        self.section_generators = self._initialize_section_generators()
        self.report_templates = self._initialize_report_templates()
    
    def _initialize_section_generators(self) -> Dict[str, Any]:
        """Initialize section-specific generation logic."""
        return {
            "executive_summary": self._generate_executive_summary,
            "business_objectives": self._extract_business_objectives,
            "current_situation": self._analyze_current_situation,
            "stakeholders": self._identify_stakeholders,
            "constraints": self._extract_constraints,
            "success_metrics": self._define_success_metrics,
            "risks_assumptions": self._compile_risks_assumptions,
            "recommended_next_steps": self._generate_recommendations
        }
    
    def _initialize_report_templates(self) -> Dict[str, str]:
        """Initialize downloadable report templates."""
        return {
            "markdown": self._get_markdown_template(),
            "text": self._get_text_template(),
            "json": self._get_json_template()
        }
    
    def generate_summary(self, session: SessionState) -> BusinessSummary:
        """
        Generate a comprehensive Business Understanding Summary from session data.
        
        Args:
            session: Complete session state with all questions and answers
            
        Returns:
            BusinessSummary: Structured summary with all required sections
        """
        if not session.answers_collected:
            raise ValueError("Cannot generate summary from session with no answers")
        
        # Perform analysis first
        assumptions = self.analysis_engine.identify_assumptions(session.answers_collected)
        gaps = self.analysis_engine.detect_gaps(session.answers_collected)
        insights = self.analysis_engine.generate_insights(session)
        
        # Generate each section
        summary = BusinessSummary(
            session_id=session.session_id,
            business_question=session.business_question,
            executive_summary=self._generate_executive_summary(session, insights),
            business_objectives=self._extract_business_objectives(session),
            current_situation=self._analyze_current_situation(session),
            stakeholders=self._identify_stakeholders(session),
            constraints=self._extract_constraints(session),
            success_metrics=self._define_success_metrics(session),
            risks_assumptions=self._compile_risks_assumptions(session, assumptions, gaps),
            recommended_next_steps=self._generate_recommendations(session, insights),
            identified_gaps=gaps,
            business_insights=insights,
            readiness_assessment=insights.overall_readiness if insights else "unknown"
        )
        
        # Calculate completeness score
        summary.completeness_score = self._calculate_completeness_score(summary, session)
        
        return summary
    
    def _generate_executive_summary(self, session: SessionState, insights: Optional[BusinessInsights] = None) -> str:
        """Generate executive summary section."""
        # Extract key information from answers
        answers_by_dimension = self._group_answers_by_dimension(session)
        
        # Build executive summary components
        problem_statement = self._extract_problem_statement(answers_by_dimension)
        key_objectives = self._extract_key_objectives(answers_by_dimension)
        main_stakeholders = self._extract_main_stakeholders(answers_by_dimension)
        critical_constraints = self._extract_critical_constraints(answers_by_dimension)
        
        # Construct executive summary
        summary_parts = [
            f"Business Challenge: {problem_statement}",
            f"Primary Objectives: {key_objectives}",
            f"Key Stakeholders: {main_stakeholders}",
            f"Critical Constraints: {critical_constraints}"
        ]
        
        if insights and insights.overall_readiness:
            summary_parts.append(f"Readiness Assessment: {insights.overall_readiness.title()} readiness for data mining phase")
        
        return " | ".join(summary_parts)
    
    def _extract_business_objectives(self, session: SessionState) -> List[str]:
        """Extract business objectives from session answers."""
        objectives = []
        
        # Look for objective-related answers
        for answer in session.answers_collected:
            # Find corresponding question
            question = self._find_question_by_id(session, answer.question_id)
            if question and question.dimension == Dimension.BUSINESS_OBJECTIVES:
                # Extract objectives from answer
                objective_text = self._clean_answer_text(answer.response)
                if objective_text and len(objective_text) > 10:
                    objectives.append(objective_text)
        
        # Ensure we have at least one objective
        if not objectives:
            objectives.append("Primary business objective to be further defined")
        
        return objectives[:5]  # Limit to top 5 objectives
    
    def _analyze_current_situation(self, session: SessionState) -> str:
        """Analyze and summarize current situation."""
        situation_parts = []
        
        # Extract current situation answers
        for answer in session.answers_collected:
            question = self._find_question_by_id(session, answer.question_id)
            if question and question.dimension == Dimension.CURRENT_SITUATION:
                cleaned_text = self._clean_answer_text(answer.response)
                if cleaned_text:
                    situation_parts.append(cleaned_text)
        
        if situation_parts:
            return " ".join(situation_parts)
        else:
            return "Current situation analysis requires additional information gathering"
    
    def _identify_stakeholders(self, session: SessionState) -> List[Stakeholder]:
        """Identify and categorize stakeholders from session data."""
        stakeholders = []
        stakeholder_mentions = {}
        
        # Extract stakeholder information
        for answer in session.answers_collected:
            question = self._find_question_by_id(session, answer.question_id)
            if question and question.dimension == Dimension.STAKEHOLDERS:
                # Parse stakeholder information from answer
                parsed_stakeholders = self._parse_stakeholders_from_text(answer.response)
                for stakeholder_info in parsed_stakeholders:
                    name = stakeholder_info.get("name", "Unnamed Stakeholder")
                    if name not in stakeholder_mentions:
                        stakeholder_mentions[name] = stakeholder_info
        
        # Convert to Stakeholder objects
        for name, info in stakeholder_mentions.items():
            stakeholder = Stakeholder(
                name=name,
                role=info.get("role", "Stakeholder"),
                influence_level=info.get("influence", "medium"),
                involvement=info.get("involvement", "observer"),
                concerns=info.get("concerns", [])
            )
            stakeholders.append(stakeholder)
        
        # Ensure we have at least one stakeholder
        if not stakeholders:
            stakeholders.append(Stakeholder(
                name="Primary Stakeholder",
                role="To be identified",
                influence_level="high",
                involvement="decision_maker"
            ))
        
        return stakeholders
    
    def _extract_constraints(self, session: SessionState) -> List[Constraint]:
        """Extract constraints from session data."""
        constraints = []
        
        # Look for constraint-related answers
        for answer in session.answers_collected:
            question = self._find_question_by_id(session, answer.question_id)
            if question and question.dimension == Dimension.CONSTRAINTS:
                # Parse constraints from answer
                parsed_constraints = self._parse_constraints_from_text(answer.response)
                constraints.extend(parsed_constraints)
        
        # Ensure we have at least one constraint
        if not constraints:
            constraints.append(Constraint(
                type="general",
                description="Constraints to be further defined",
                impact_level="medium"
            ))
        
        return constraints[:10]  # Limit to top 10 constraints
    
    def _define_success_metrics(self, session: SessionState) -> List[Metric]:
        """Define success metrics from session data."""
        metrics = []
        
        # Look for success criteria answers
        for answer in session.answers_collected:
            question = self._find_question_by_id(session, answer.question_id)
            if question and question.dimension == Dimension.SUCCESS_CRITERIA:
                # Parse metrics from answer
                parsed_metrics = self._parse_metrics_from_text(answer.response)
                metrics.extend(parsed_metrics)
        
        # Ensure we have at least one metric
        if not metrics:
            metrics.append(Metric(
                name="Primary Success Metric",
                description="Success measurement to be further defined",
                measurement_method="To be determined"
            ))
        
        return metrics[:8]  # Limit to top 8 metrics
    
    def _compile_risks_assumptions(self, session: SessionState, 
                                 assumptions: List[Assumption], 
                                 gaps: List[Gap]) -> List[RiskAssumption]:
        """Compile risks and assumptions from analysis results."""
        risk_assumptions = []
        
        # Convert assumptions to RiskAssumption objects
        for assumption in assumptions:
            risk_assumption = RiskAssumption(
                id=assumption.id,
                type="assumption",
                category=assumption.type.value,
                description=assumption.description,
                impact_level=assumption.risk_level,
                validation_required=True
            )
            risk_assumptions.append(risk_assumption)
        
        # Convert gaps to risks
        for gap in gaps:
            risk_assumption = RiskAssumption(
                id=gap.id,
                type="risk",
                category=gap.type.value,
                description=f"Information gap: {gap.description}",
                impact_level=gap.severity,
                validation_required=True
            )
            risk_assumptions.append(risk_assumption)
        
        # Ensure we have at least one item
        if not risk_assumptions:
            risk_assumptions.append(RiskAssumption(
                id="default_risk",
                type="risk",
                category="general",
                description="Risk assessment requires additional analysis",
                impact_level=RiskLevel.MEDIUM
            ))
        
        return risk_assumptions
    
    def _generate_recommendations(self, session: SessionState, 
                                insights: Optional[BusinessInsights] = None) -> List[str]:
        """Generate recommended next steps."""
        recommendations = []
        
        # Use insights recommendations if available
        if insights and insights.recommendations:
            recommendations.extend(insights.recommendations)
        
        # Add standard recommendations based on session completeness
        completeness_score = len(session.answers_collected) / 32  # Assuming ~32 total questions
        
        if completeness_score < 0.7:
            recommendations.append("Complete remaining business understanding questions to ensure comprehensive analysis")
        
        if len(session.dimensions_completed) < 8:
            missing_dimensions = 8 - len(session.dimensions_completed)
            recommendations.append(f"Address {missing_dimensions} remaining dimension(s) for complete CRISP-DM coverage")
        
        # Add phase-specific recommendations
        recommendations.extend([
            "Validate identified assumptions with relevant stakeholders",
            "Develop detailed project plan based on business understanding",
            "Proceed to CRISP-DM Data Understanding phase with clear objectives"
        ])
        
        return recommendations[:8]  # Limit to top 8 recommendations
    
    def format_for_download(self, summary: BusinessSummary, format_type: str = "markdown") -> str:
        """
        Format summary for download in specified format.
        
        Args:
            summary: BusinessSummary to format
            format_type: Format type ("markdown", "text", "json")
            
        Returns:
            str: Formatted document content
        """
        if format_type not in self.report_templates:
            raise ValueError(f"Unsupported format type: {format_type}")
        
        if format_type == "json":
            return self._format_as_json(summary)
        elif format_type == "markdown":
            return self._format_as_markdown(summary)
        elif format_type == "text":
            return self._format_as_text(summary)
        else:
            return self._format_as_text(summary)  # Default fallback
    
    def highlight_gaps(self, summary: BusinessSummary, gaps: List[Gap]) -> BusinessSummary:
        """
        Highlight gaps and assumptions in the summary.
        
        Args:
            summary: BusinessSummary to enhance
            gaps: List of gaps to highlight
            
        Returns:
            BusinessSummary: Enhanced summary with gap highlighting
        """
        # Update identified gaps
        summary.identified_gaps = gaps
        
        # Add gap information to executive summary
        if gaps:
            high_priority_gaps = [g for g in gaps if g.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]]
            if high_priority_gaps:
                gap_note = f" | Critical Gaps: {len(high_priority_gaps)} high-priority information gaps identified"
                summary.executive_summary += gap_note
        
        # Add gap-related recommendations
        gap_recommendations = []
        for gap in gaps:
            if gap.suggested_questions:
                gap_recommendations.append(f"Address {gap.type.value.replace('_', ' ')}: {gap.suggested_questions[0]}")
        
        if gap_recommendations:
            summary.recommended_next_steps.extend(gap_recommendations[:3])  # Add top 3 gap recommendations
        
        return summary
    
    # Helper methods
    
    def _group_answers_by_dimension(self, session: SessionState) -> Dict[Dimension, List[Answer]]:
        """Group answers by their question dimensions."""
        grouped = {}
        
        for answer in session.answers_collected:
            question = self._find_question_by_id(session, answer.question_id)
            if question:
                if question.dimension not in grouped:
                    grouped[question.dimension] = []
                grouped[question.dimension].append(answer)
        
        return grouped
    
    def _find_question_by_id(self, session: SessionState, question_id: str) -> Optional[Question]:
        """Find a question by its ID in the session."""
        for question in session.questions_asked:
            if question.id == question_id:
                return question
        return None
    
    def _clean_answer_text(self, text: str) -> str:
        """Clean and normalize answer text."""
        if not text:
            return ""
        
        # Remove extra whitespace and normalize
        cleaned = " ".join(text.split())
        
        # Truncate if too long
        if len(cleaned) > 500:
            cleaned = cleaned[:500] + "..."
        
        return cleaned
    
    def _extract_problem_statement(self, answers_by_dimension: Dict[Dimension, List[Answer]]) -> str:
        """Extract problem statement from problem definition answers."""
        if Dimension.PROBLEM_DEFINITION in answers_by_dimension:
            answers = answers_by_dimension[Dimension.PROBLEM_DEFINITION]
            if answers:
                return self._clean_answer_text(answers[0].response)
        return "Business problem to be further defined"
    
    def _extract_key_objectives(self, answers_by_dimension: Dict[Dimension, List[Answer]]) -> str:
        """Extract key objectives summary."""
        if Dimension.BUSINESS_OBJECTIVES in answers_by_dimension:
            answers = answers_by_dimension[Dimension.BUSINESS_OBJECTIVES]
            if answers:
                objectives_text = " ".join([self._clean_answer_text(a.response) for a in answers[:2]])
                return objectives_text[:200] + "..." if len(objectives_text) > 200 else objectives_text
        return "Objectives to be further defined"
    
    def _extract_main_stakeholders(self, answers_by_dimension: Dict[Dimension, List[Answer]]) -> str:
        """Extract main stakeholders summary."""
        if Dimension.STAKEHOLDERS in answers_by_dimension:
            answers = answers_by_dimension[Dimension.STAKEHOLDERS]
            if answers:
                stakeholders_text = self._clean_answer_text(answers[0].response)
                return stakeholders_text[:150] + "..." if len(stakeholders_text) > 150 else stakeholders_text
        return "Stakeholders to be identified"
    
    def _extract_critical_constraints(self, answers_by_dimension: Dict[Dimension, List[Answer]]) -> str:
        """Extract critical constraints summary."""
        if Dimension.CONSTRAINTS in answers_by_dimension:
            answers = answers_by_dimension[Dimension.CONSTRAINTS]
            if answers:
                constraints_text = self._clean_answer_text(answers[0].response)
                return constraints_text[:150] + "..." if len(constraints_text) > 150 else constraints_text
        return "Constraints to be assessed"
    
    def _parse_stakeholders_from_text(self, text: str) -> List[Dict[str, Any]]:
        """Parse stakeholder information from text."""
        # Simple parsing - in a real implementation, this could use NLP
        stakeholders = []
        text_lower = text.lower()
        
        # Look for specific stakeholder mentions with more sophisticated parsing
        stakeholder_patterns = {
            "customer success": {"role": "Customer Success Manager", "influence": "high", "involvement": "end_user"},
            "product manager": {"role": "Product Manager", "influence": "high", "involvement": "decision_maker"},
            "executive leadership": {"role": "Executive", "influence": "high", "involvement": "decision_maker"},
            "management": {"role": "Manager", "influence": "high", "involvement": "decision_maker"},
            "team": {"role": "Team Member", "influence": "medium", "involvement": "end_user"},
            "user": {"role": "End User", "influence": "medium", "involvement": "end_user"},
            "customer": {"role": "Customer", "influence": "high", "involvement": "end_user"},
            "client": {"role": "Client", "influence": "high", "involvement": "end_user"},
            "director": {"role": "Director", "influence": "high", "involvement": "decision_maker"},
            "executive": {"role": "Executive", "influence": "high", "involvement": "decision_maker"},
            "staff": {"role": "Staff Member", "influence": "low", "involvement": "observer"}
        }
        
        # Check for specific patterns first
        for pattern, attributes in stakeholder_patterns.items():
            if pattern in text_lower:
                stakeholders.append({
                    "name": pattern.title(),
                    "role": attributes["role"],
                    "influence": attributes["influence"],
                    "involvement": attributes["involvement"]
                })
        
        # If no specific stakeholders found, create a generic one
        if not stakeholders:
            stakeholders.append({
                "name": "Primary Stakeholder",
                "role": "Stakeholder",
                "influence": "high",
                "involvement": "decision_maker"
            })
        
        return stakeholders[:5]  # Limit to 5 stakeholders
    
    def _parse_constraints_from_text(self, text: str) -> List[Constraint]:
        """Parse constraints from text."""
        constraints = []
        text_lower = text.lower()
        
        # Budget constraints
        if any(word in text_lower for word in ["budget", "cost", "money", "funding"]):
            constraints.append(Constraint(
                type="budget",
                description="Budget constraints mentioned",
                impact_level="medium"
            ))
        
        # Timeline constraints
        if any(word in text_lower for word in ["time", "deadline", "schedule", "urgent"]):
            constraints.append(Constraint(
                type="timeline",
                description="Timeline constraints identified",
                impact_level="medium"
            ))
        
        # Resource constraints
        if any(word in text_lower for word in ["resource", "staff", "people", "capacity"]):
            constraints.append(Constraint(
                type="resource",
                description="Resource constraints noted",
                impact_level="medium"
            ))
        
        return constraints
    
    def _parse_metrics_from_text(self, text: str) -> List[Metric]:
        """Parse success metrics from text."""
        metrics = []
        text_lower = text.lower()
        
        # Common metric types
        if any(word in text_lower for word in ["revenue", "sales", "profit", "roi"]):
            metrics.append(Metric(
                name="Financial Performance",
                description="Financial metrics mentioned",
                measurement_method="Financial reporting"
            ))
        
        if any(word in text_lower for word in ["efficiency", "productivity", "performance"]):
            metrics.append(Metric(
                name="Operational Efficiency",
                description="Efficiency metrics identified",
                measurement_method="Operational reporting"
            ))
        
        if any(word in text_lower for word in ["customer", "satisfaction", "user", "experience"]):
            metrics.append(Metric(
                name="Customer Satisfaction",
                description="Customer-focused metrics noted",
                measurement_method="Customer feedback"
            ))
        
        return metrics
    
    def _calculate_completeness_score(self, summary: BusinessSummary, session: SessionState) -> float:
        """Calculate completeness score for the summary."""
        score = 0.0
        max_score = 10.0
        
        # Section completeness (4 points)
        required_sections = [
            summary.executive_summary, summary.business_objectives,
            summary.current_situation, summary.stakeholders,
            summary.constraints, summary.success_metrics,
            summary.risks_assumptions, summary.recommended_next_steps
        ]
        
        non_empty_sections = sum(1 for section in required_sections if section and len(str(section)) > 20)
        score += (non_empty_sections / len(required_sections)) * 4
        
        # Answer completeness (3 points)
        answer_ratio = min(len(session.answers_collected) / 24, 1.0)  # Assuming 24 typical answers
        score += answer_ratio * 3
        
        # Dimension coverage (2 points)
        dimension_ratio = len(session.dimensions_completed) / 8
        score += dimension_ratio * 2
        
        # Analysis depth (1 point)
        if summary.business_insights and len(summary.business_insights.insights) > 0:
            score += 1
        
        return min(score / max_score, 1.0)
    
    def _format_as_json(self, summary: BusinessSummary) -> str:
        """Format summary as JSON."""
        # Convert to serializable format
        summary_dict = {
            "session_id": summary.session_id,
            "business_question": summary.business_question,
            "executive_summary": summary.executive_summary,
            "business_objectives": summary.business_objectives,
            "current_situation": summary.current_situation,
            "stakeholders": [
                {
                    "name": s.name,
                    "role": s.role,
                    "influence_level": s.influence_level,
                    "involvement": s.involvement,
                    "concerns": s.concerns
                }
                for s in summary.stakeholders
            ],
            "constraints": [
                {
                    "type": c.type,
                    "description": c.description,
                    "impact_level": c.impact_level,
                    "mitigation_strategies": c.mitigation_strategies
                }
                for c in summary.constraints
            ],
            "success_metrics": [
                {
                    "name": m.name,
                    "description": m.description,
                    "target_value": m.target_value,
                    "measurement_method": m.measurement_method,
                    "frequency": m.frequency,
                    "owner": m.owner
                }
                for m in summary.success_metrics
            ],
            "risks_assumptions": [
                {
                    "id": ra.id,
                    "type": ra.type,
                    "category": ra.category,
                    "description": ra.description,
                    "impact_level": ra.impact_level.value,
                    "likelihood": ra.likelihood,
                    "mitigation_plan": ra.mitigation_plan,
                    "validation_required": ra.validation_required
                }
                for ra in summary.risks_assumptions
            ],
            "recommended_next_steps": summary.recommended_next_steps,
            "completeness_score": summary.completeness_score,
            "readiness_assessment": summary.readiness_assessment,
            "generation_timestamp": summary.generation_timestamp.isoformat()
        }
        
        return json.dumps(summary_dict, indent=2)
    
    def _format_as_markdown(self, summary: BusinessSummary) -> str:
        """Format summary as Markdown."""
        md_content = f"""# Business Understanding Summary

**Session ID:** {summary.session_id}
**Business Question:** {summary.business_question}
**Generated:** {summary.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
**Completeness Score:** {summary.completeness_score:.1%}
**Readiness Assessment:** {summary.readiness_assessment.title()}

## Executive Summary

{summary.executive_summary}

## Business Objectives

"""
        
        for i, objective in enumerate(summary.business_objectives, 1):
            md_content += f"{i}. {objective}\n"
        
        md_content += f"""
## Current Situation

{summary.current_situation}

## Stakeholders

"""
        
        for stakeholder in summary.stakeholders:
            md_content += f"- **{stakeholder.name}** ({stakeholder.role})\n"
            md_content += f"  - Influence: {stakeholder.influence_level}\n"
            md_content += f"  - Involvement: {stakeholder.involvement}\n"
            if stakeholder.concerns:
                md_content += f"  - Concerns: {', '.join(stakeholder.concerns)}\n"
            md_content += "\n"
        
        md_content += "## Constraints\n\n"
        
        for constraint in summary.constraints:
            md_content += f"- **{constraint.type.title()}:** {constraint.description}\n"
            md_content += f"  - Impact Level: {constraint.impact_level}\n"
            if constraint.mitigation_strategies:
                md_content += f"  - Mitigation: {', '.join(constraint.mitigation_strategies)}\n"
            md_content += "\n"
        
        md_content += "## Success Metrics\n\n"
        
        for metric in summary.success_metrics:
            md_content += f"- **{metric.name}:** {metric.description}\n"
            if metric.target_value:
                md_content += f"  - Target: {metric.target_value}\n"
            if metric.measurement_method:
                md_content += f"  - Measurement: {metric.measurement_method}\n"
            md_content += "\n"
        
        md_content += "## Risks & Assumptions\n\n"
        
        for ra in summary.risks_assumptions:
            md_content += f"- **{ra.type.title()}** ({ra.category}): {ra.description}\n"
            md_content += f"  - Impact Level: {ra.impact_level.value}\n"
            if ra.validation_required:
                md_content += f"  - Validation Required: Yes\n"
            md_content += "\n"
        
        md_content += "## Recommended Next Steps\n\n"
        
        for i, step in enumerate(summary.recommended_next_steps, 1):
            md_content += f"{i}. {step}\n"
        
        if summary.identified_gaps:
            md_content += "\n## Identified Gaps\n\n"
            for gap in summary.identified_gaps:
                md_content += f"- **{gap.type.value.replace('_', ' ').title()}:** {gap.description}\n"
                md_content += f"  - Severity: {gap.severity.value}\n"
                if gap.suggested_questions:
                    md_content += f"  - Suggested Questions: {gap.suggested_questions[0]}\n"
                md_content += "\n"
        
        return md_content
    
    def _format_as_text(self, summary: BusinessSummary) -> str:
        """Format summary as plain text."""
        text_content = f"""BUSINESS UNDERSTANDING SUMMARY
{'=' * 50}

Session ID: {summary.session_id}
Business Question: {summary.business_question}
Generated: {summary.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Completeness Score: {summary.completeness_score:.1%}
Readiness Assessment: {summary.readiness_assessment.title()}

EXECUTIVE SUMMARY
{'-' * 20}
{summary.executive_summary}

BUSINESS OBJECTIVES
{'-' * 20}
"""
        
        for i, objective in enumerate(summary.business_objectives, 1):
            text_content += f"{i}. {objective}\n"
        
        text_content += f"""
CURRENT SITUATION
{'-' * 20}
{summary.current_situation}

STAKEHOLDERS
{'-' * 20}
"""
        
        for stakeholder in summary.stakeholders:
            text_content += f"- {stakeholder.name} ({stakeholder.role})\n"
            text_content += f"  Influence: {stakeholder.influence_level}, Involvement: {stakeholder.involvement}\n"
        
        text_content += f"""
CONSTRAINTS
{'-' * 20}
"""
        
        for constraint in summary.constraints:
            text_content += f"- {constraint.type.title()}: {constraint.description}\n"
            text_content += f"  Impact Level: {constraint.impact_level}\n"
        
        text_content += f"""
SUCCESS METRICS
{'-' * 20}
"""
        
        for metric in summary.success_metrics:
            text_content += f"- {metric.name}: {metric.description}\n"
            if metric.target_value:
                text_content += f"  Target: {metric.target_value}\n"
        
        text_content += f"""
RISKS & ASSUMPTIONS
{'-' * 20}
"""
        
        for ra in summary.risks_assumptions:
            text_content += f"- {ra.type.title()} ({ra.category}): {ra.description}\n"
            text_content += f"  Impact Level: {ra.impact_level.value}\n"
        
        text_content += f"""
RECOMMENDED NEXT STEPS
{'-' * 20}
"""
        
        for i, step in enumerate(summary.recommended_next_steps, 1):
            text_content += f"{i}. {step}\n"
        
        return text_content
    
    def _get_markdown_template(self) -> str:
        """Get Markdown template structure."""
        return "markdown"
    
    def _get_text_template(self) -> str:
        """Get text template structure."""
        return "text"
    
    def _get_json_template(self) -> str:
        """Get JSON template structure."""
        return "json"