"""
Error Handling and Recovery System for CRISP-DM Business Understanding Specialist

This module provides comprehensive error handling, input validation, session recovery,
and graceful degradation capabilities for the CRISP-DM framework.
"""

import logging
import json
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import re
import pickle
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crisp_dm_errors.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


@dataclass
class ValidationError:
    """Represents a validation error with context."""
    field: str
    message: str
    severity: str  # 'error', 'warning', 'info'
    suggestion: Optional[str] = None


@dataclass
class RecoveryState:
    """Represents the state needed for session recovery."""
    session_id: str
    timestamp: datetime
    business_question: str
    current_dimension: str
    completed_dimensions: List[str]
    question_history: List[Dict[str, Any]]
    answers: Dict[str, Any]
    metadata: Dict[str, Any]


class InputValidator:
    """Validates business questions and answers with comprehensive checks."""
    
    def __init__(self):
        self.min_question_length = 10
        self.max_question_length = 1000
        self.min_answer_length = 2
        self.max_answer_length = 5000
        
        # Common business terms for validation
        self.business_keywords = {
            'revenue', 'profit', 'customer', 'market', 'sales', 'growth', 'cost',
            'efficiency', 'performance', 'strategy', 'competitive', 'analysis',
            'data', 'insight', 'trend', 'forecast', 'budget', 'roi', 'kpi'
        }
        
        # Suspicious patterns that might indicate invalid input
        self.suspicious_patterns = [
            r'(.)\1{10,}',  # Repeated characters
            r'[^\w\s\.,!?;:\-\'\"()]{5,}',  # Too many special characters
            r'^\s*test\s*$',  # Just "test"
            r'^\s*[a-z]\s*$',  # Single letter
            r'^\s*\d+\s*$',  # Just numbers
        ]
    
    def validate_business_question(self, question: str) -> List[ValidationError]:
        """Validate a business question with comprehensive checks."""
        errors = []
        
        if not question or not isinstance(question, str):
            errors.append(ValidationError(
                field="business_question",
                message="Business question is required and must be text",
                severity="error",
                suggestion="Please provide a clear business question describing what you want to analyze"
            ))
            return errors
        
        question = question.strip()
        
        # Length validation
        if len(question) < self.min_question_length:
            errors.append(ValidationError(
                field="business_question",
                message=f"Question too short (minimum {self.min_question_length} characters)",
                severity="error",
                suggestion="Please provide more detail about your business challenge"
            ))
        
        if len(question) > self.max_question_length:
            errors.append(ValidationError(
                field="business_question",
                message=f"Question too long (maximum {self.max_question_length} characters)",
                severity="error",  # Changed from warning to error
                suggestion="Consider breaking down your question into more focused parts"
            ))
        
        # Pattern validation
        for pattern in self.suspicious_patterns:
            if re.search(pattern, question, re.IGNORECASE):
                # Determine severity based on pattern type
                if question.strip().lower() in ['test', 'a', '1', '123']:
                    severity = "error"
                elif len(question.strip()) < 15 and re.match(r'^[^\w\s]+$', question.strip()):
                    severity = "error"  # Only special characters in short string
                else:
                    severity = "warning"
                
                errors.append(ValidationError(
                    field="business_question",
                    message="Question appears to contain invalid or test content",
                    severity=severity,
                    suggestion="Please provide a genuine business question"
                ))
                break
        
        # Business relevance check
        question_lower = question.lower()
        has_business_context = any(keyword in question_lower for keyword in self.business_keywords)
        
        if not has_business_context and len(question) > 20:
            errors.append(ValidationError(
                field="business_question",
                message="Question may not be business-related",
                severity="info",
                suggestion="Consider framing your question in business terms (revenue, customers, performance, etc.)"
            ))
        
        # Question mark check for questions
        if not question.endswith('?') and any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where', 'which']):
            errors.append(ValidationError(
                field="business_question",
                message="Question appears to be missing a question mark",
                severity="info",
                suggestion="Consider ending your question with '?' for clarity"
            ))
        
        return errors
    
    def validate_answer(self, answer: str, question_context: str = "") -> List[ValidationError]:
        """Validate an answer with context-aware checks."""
        errors = []
        
        if not answer or not isinstance(answer, str):
            errors.append(ValidationError(
                field="answer",
                message="Answer is required",
                severity="error",
                suggestion="Please provide an answer to continue"
            ))
            return errors
        
        answer = answer.strip()
        
        # Length validation
        if len(answer) < self.min_answer_length:
            errors.append(ValidationError(
                field="answer",
                message="Answer too short",
                severity="warning",
                suggestion="Please provide more detail in your answer"
            ))
        
        if len(answer) > self.max_answer_length:
            errors.append(ValidationError(
                field="answer",
                message=f"Answer too long (maximum {self.max_answer_length} characters)",
                severity="warning",
                suggestion="Consider providing a more concise answer"
            ))
        
        # Pattern validation
        for pattern in self.suspicious_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                errors.append(ValidationError(
                    field="answer",
                    message="Answer appears to contain invalid content",
                    severity="warning",
                    suggestion="Please provide a meaningful answer"
                ))
                break
        
        # Generic answer detection
        generic_answers = {
            'yes', 'no', 'maybe', 'idk', 'i don\'t know', 'not sure', 'n/a', 'na'
        }
        
        if answer.lower().strip() in generic_answers and len(question_context) > 0:
            errors.append(ValidationError(
                field="answer",
                message="Answer appears too generic",
                severity="info",
                suggestion="Consider providing more specific details about your situation"
            ))
        
        return errors
    
    def validate_session_data(self, session_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate complete session data for consistency."""
        errors = []
        
        required_fields = ['session_id', 'business_question', 'current_dimension']
        
        for field in required_fields:
            if field not in session_data or not session_data[field]:
                errors.append(ValidationError(
                    field=field,
                    message=f"Required field '{field}' is missing or empty",
                    severity="error",
                    suggestion=f"Please ensure {field} is properly set"
                ))
        
        # Validate dimension progression
        if 'completed_dimensions' in session_data and 'current_dimension' in session_data:
            completed = session_data['completed_dimensions']
            current = session_data['current_dimension']
            
            if current in completed:
                errors.append(ValidationError(
                    field="dimension_progression",
                    message="Current dimension is already marked as completed",
                    severity="warning",
                    suggestion="Check dimension progression logic"
                ))
        
        # Validate question-answer pairs
        if 'question_history' in session_data:
            for i, qa_pair in enumerate(session_data['question_history']):
                if not isinstance(qa_pair, (list, tuple)) or len(qa_pair) != 2:
                    errors.append(ValidationError(
                        field=f"question_history[{i}]",
                        message="Invalid question-answer pair format",
                        severity="error",
                        suggestion="Each entry should be a [question, answer] pair"
                    ))
        
        return errors


class SessionRecovery:
    """Handles session recovery and persistence."""
    
    def __init__(self, storage_path: str = ".crisp_dm_sessions"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        self.max_session_age_hours = 24
    
    def save_session_state(self, session_data: Dict[str, Any]) -> bool:
        """Save session state for recovery."""
        try:
            session_id = session_data.get('session_id')
            if not session_id:
                logger.error("Cannot save session: missing session_id")
                return False
            
            recovery_state = RecoveryState(
                session_id=session_id,
                timestamp=datetime.now(),
                business_question=session_data.get('business_question', ''),
                current_dimension=session_data.get('current_dimension', ''),
                completed_dimensions=session_data.get('completed_dimensions', []),
                question_history=session_data.get('question_history', []),
                answers=session_data.get('answers', {}),
                metadata=session_data.get('metadata', {})
            )
            
            file_path = self.storage_path / f"{session_id}.json"
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(recovery_state), f, indent=2, default=str)
            
            logger.info(f"Session {session_id} saved successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return False
    
    def load_session_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session state for recovery."""
        try:
            file_path = self.storage_path / f"{session_id}.json"
            
            if not file_path.exists():
                logger.warning(f"Session file not found: {session_id}")
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Check if session is too old
            timestamp = datetime.fromisoformat(data['timestamp'])
            if datetime.now() - timestamp > timedelta(hours=self.max_session_age_hours):
                logger.warning(f"Session {session_id} is too old, removing")
                file_path.unlink()
                return None
            
            logger.info(f"Session {session_id} loaded successfully")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load session {session_id}: {e}")
            return None
    
    def list_recoverable_sessions(self) -> List[Dict[str, Any]]:
        """List all recoverable sessions."""
        sessions = []
        
        try:
            for file_path in self.storage_path.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Check age
                    timestamp = datetime.fromisoformat(data['timestamp'])
                    if datetime.now() - timestamp > timedelta(hours=self.max_session_age_hours):
                        file_path.unlink()  # Remove old session
                        continue
                    
                    sessions.append({
                        'session_id': data['session_id'],
                        'timestamp': data['timestamp'],
                        'business_question': data['business_question'][:100] + '...' if len(data['business_question']) > 100 else data['business_question'],
                        'progress': f"{len(data['completed_dimensions'])}/8 dimensions"
                    })
                    
                except Exception as e:
                    logger.error(f"Error reading session file {file_path}: {e}")
                    continue
            
            # Sort by timestamp (newest first)
            sessions.sort(key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
        
        return sessions
    
    def cleanup_old_sessions(self) -> int:
        """Clean up old session files."""
        cleaned = 0
        
        try:
            cutoff_time = datetime.now() - timedelta(hours=self.max_session_age_hours)
            
            for file_path in self.storage_path.glob("*.json"):
                try:
                    # Check file modification time
                    if datetime.fromtimestamp(file_path.stat().st_mtime) < cutoff_time:
                        file_path.unlink()
                        cleaned += 1
                        
                except Exception as e:
                    logger.error(f"Error cleaning up {file_path}: {e}")
            
            if cleaned > 0:
                logger.info(f"Cleaned up {cleaned} old session files")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        
        return cleaned


class GracefulDegradation:
    """Handles graceful degradation when AI services fail."""
    
    def __init__(self):
        self.fallback_questions = {
            'problem_definition': [
                {
                    'question': 'What is the main business problem you want to solve?',
                    'options': ['Revenue decline', 'Customer churn', 'Operational efficiency', 'Market expansion', 'Other'],
                    'reasoning': 'Understanding the core problem helps scope the analysis'
                },
                {
                    'question': 'How urgent is this problem for your business?',
                    'options': ['Critical - needs immediate attention', 'Important - should be addressed soon', 'Moderate - can wait a few months', 'Low - nice to have'],
                    'reasoning': 'Urgency helps prioritize the analysis approach'
                }
            ],
            'business_objectives': [
                {
                    'question': 'What are your primary business objectives for this analysis?',
                    'options': ['Increase revenue', 'Reduce costs', 'Improve efficiency', 'Better decision making', 'Risk mitigation'],
                    'reasoning': 'Clear objectives guide the analysis direction'
                },
                {
                    'question': 'How will you measure success?',
                    'options': ['Financial metrics (ROI, revenue)', 'Operational metrics (efficiency, time)', 'Customer metrics (satisfaction, retention)', 'Strategic metrics (market share)', 'Multiple metrics'],
                    'reasoning': 'Success criteria ensure the analysis delivers value'
                }
            ],
            'stakeholders': [
                {
                    'question': 'Who are the key stakeholders for this project?',
                    'options': ['Executive leadership', 'Department managers', 'Analysts/Data team', 'End users', 'External partners'],
                    'reasoning': 'Stakeholder alignment is crucial for project success'
                }
            ],
            'current_situation': [
                {
                    'question': 'What is your current approach to this business area?',
                    'options': ['Manual processes', 'Basic reporting', 'Some analytics', 'Advanced analytics', 'No formal approach'],
                    'reasoning': 'Understanding the baseline helps measure improvement'
                }
            ],
            'constraints': [
                {
                    'question': 'What are your main constraints for this project?',
                    'options': ['Budget limitations', 'Time constraints', 'Data availability', 'Technical capabilities', 'Organizational readiness'],
                    'reasoning': 'Constraints shape what solutions are feasible'
                }
            ],
            'success_criteria': [
                {
                    'question': 'What would make this analysis successful?',
                    'options': ['Clear actionable insights', 'Improved decision making', 'Measurable business impact', 'Process improvements', 'Strategic clarity'],
                    'reasoning': 'Success criteria ensure we deliver what you need'
                }
            ],
            'business_domain': [
                {
                    'question': 'What industry or business domain is this for?',
                    'options': ['Technology/Software', 'Retail/E-commerce', 'Financial services', 'Healthcare', 'Manufacturing', 'Other'],
                    'reasoning': 'Industry context helps provide relevant insights'
                }
            ],
            'implementation': [
                {
                    'question': 'How do you plan to implement the analysis results?',
                    'options': ['Immediate action', 'Gradual rollout', 'Pilot program first', 'Further analysis needed', 'Not sure yet'],
                    'reasoning': 'Implementation planning ensures results are actionable'
                }
            ]
        }
    
    def get_fallback_questions(self, dimension: str, count: int = 2) -> List[Dict[str, Any]]:
        """Get fallback questions when AI services are unavailable."""
        questions = self.fallback_questions.get(dimension, [])
        return questions[:count] if questions else []
    
    def generate_fallback_summary(self, session_data: Dict[str, Any]) -> str:
        """Generate a basic summary when AI services are unavailable."""
        business_question = session_data.get('business_question', 'Business analysis')
        completed_dimensions = session_data.get('completed_dimensions', [])
        question_history = session_data.get('question_history', [])
        
        summary = f"""
# Business Understanding Summary

## Primary Business Question
{business_question}

## Analysis Progress
- Completed dimensions: {len(completed_dimensions)}/8
- Total questions answered: {len(question_history)}

## Key Insights Gathered
"""
        
        if question_history:
            for i, (question, answer) in enumerate(question_history[-5:], 1):  # Last 5 Q&As
                summary += f"\n**Q{i}:** {question}\n**A{i}:** {answer}\n"
        
        summary += """
## Recommended Next Steps
1. Complete any remaining business understanding questions
2. Gather relevant data sources for analysis
3. Define specific metrics and KPIs to track
4. Develop an implementation plan for insights

*Note: This summary was generated using fallback mode due to AI service unavailability.*
"""
        
        return summary


class DataConsistencyValidator:
    """Validates and repairs data consistency issues."""
    
    def __init__(self):
        self.valid_dimensions = {
            'problem_definition', 'business_objectives', 'stakeholders',
            'current_situation', 'constraints', 'success_criteria',
            'business_domain', 'implementation'
        }
    
    def validate_consistency(self, session_data: Dict[str, Any]) -> List[ValidationError]:
        """Validate data consistency across the session."""
        errors = []
        
        # Check dimension consistency
        current_dim = session_data.get('current_dimension')
        completed_dims = session_data.get('completed_dimensions', [])
        
        if current_dim and current_dim not in self.valid_dimensions:
            errors.append(ValidationError(
                field="current_dimension",
                message=f"Invalid dimension: {current_dim}",
                severity="error",
                suggestion="Use one of the valid CRISP-DM dimensions"
            ))
        
        for dim in completed_dims:
            if dim not in self.valid_dimensions:
                errors.append(ValidationError(
                    field="completed_dimensions",
                    message=f"Invalid completed dimension: {dim}",
                    severity="error",
                    suggestion="Remove invalid dimensions from completed list"
                ))
        
        # Check for duplicate dimensions
        if len(completed_dims) != len(set(completed_dims)):
            errors.append(ValidationError(
                field="completed_dimensions",
                message="Duplicate dimensions in completed list",
                severity="warning",
                suggestion="Remove duplicate entries"
            ))
        
        # Check question-answer consistency
        question_history = session_data.get('question_history', [])
        answers = session_data.get('answers', {})
        
        # Validate question history format
        for i, qa in enumerate(question_history):
            if not isinstance(qa, (list, tuple)) or len(qa) != 2:
                errors.append(ValidationError(
                    field=f"question_history[{i}]",
                    message="Invalid question-answer pair format",
                    severity="error",
                    suggestion="Each entry should be [question, answer]"
                ))
        
        return errors
    
    def repair_data(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to repair data consistency issues."""
        repaired_data = session_data.copy()
        
        # Fix dimension issues
        if 'completed_dimensions' in repaired_data:
            # Remove invalid dimensions
            valid_completed = [
                dim for dim in repaired_data['completed_dimensions']
                if dim in self.valid_dimensions
            ]
            # Remove duplicates while preserving order
            seen = set()
            repaired_data['completed_dimensions'] = [
                dim for dim in valid_completed
                if not (dim in seen or seen.add(dim))
            ]
        
        # Fix current dimension
        if 'current_dimension' in repaired_data:
            if repaired_data['current_dimension'] not in self.valid_dimensions:
                # Default to first dimension if invalid
                repaired_data['current_dimension'] = 'problem_definition'
        
        # Fix question history format
        if 'question_history' in repaired_data:
            fixed_history = []
            for qa in repaired_data['question_history']:
                if isinstance(qa, (list, tuple)) and len(qa) == 2:
                    fixed_history.append(qa)
                elif isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    fixed_history.append([qa['question'], qa['answer']])
            repaired_data['question_history'] = fixed_history
        
        return repaired_data


class ErrorHandler:
    """Main error handling coordinator."""
    
    def __init__(self):
        self.validator = InputValidator()
        self.session_recovery = SessionRecovery()
        self.graceful_degradation = GracefulDegradation()
        self.consistency_validator = DataConsistencyValidator()
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Handle errors with appropriate recovery strategies."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'timestamp': datetime.now().isoformat(),
            'context': context or {},
            'recovery_action': None,
            'user_message': None
        }
        
        # Log the error
        logger.error(f"Error occurred: {error_info['error_type']}: {error_info['error_message']}")
        if context:
            logger.error(f"Context: {context}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Determine recovery action based on error type
        if isinstance(error, (ConnectionError, TimeoutError)):
            error_info['recovery_action'] = 'use_fallback_mode'
            error_info['user_message'] = "AI services are temporarily unavailable. Using fallback mode to continue your session."
            
        elif isinstance(error, (ValueError, TypeError)):
            error_info['recovery_action'] = 'validate_and_repair'
            error_info['user_message'] = "Data validation issue detected. Attempting to repair and continue."
            
        elif isinstance(error, FileNotFoundError):
            error_info['recovery_action'] = 'create_missing_resources'
            error_info['user_message'] = "Missing resources detected. Creating necessary files to continue."
            
        elif isinstance(error, PermissionError):
            error_info['recovery_action'] = 'use_memory_storage'
            error_info['user_message'] = "File access issue. Using temporary storage for this session."
            
        else:
            error_info['recovery_action'] = 'graceful_degradation'
            error_info['user_message'] = "An unexpected error occurred. Switching to safe mode to continue your session."
        
        return error_info
    
    def validate_and_handle_input(self, input_data: Dict[str, Any]) -> Tuple[bool, List[ValidationError]]:
        """Validate input and return success status with any errors."""
        all_errors = []
        
        # Validate business question if present
        if 'business_question' in input_data:
            errors = self.validator.validate_business_question(input_data['business_question'])
            all_errors.extend(errors)
        
        # Validate answers if present
        if 'answers' in input_data:
            for question_id, answer in input_data['answers'].items():
                errors = self.validator.validate_answer(answer)
                all_errors.extend(errors)
        
        # Validate session data consistency
        if 'session_data' in input_data:
            errors = self.validator.validate_session_data(input_data['session_data'])
            all_errors.extend(errors)
            
            # Also check data consistency
            consistency_errors = self.consistency_validator.validate_consistency(input_data['session_data'])
            all_errors.extend(consistency_errors)
        
        # Determine if validation passed (no errors, only warnings/info allowed)
        has_errors = any(error.severity == 'error' for error in all_errors)
        
        return not has_errors, all_errors


# Utility functions for easy integration
def create_error_handler() -> ErrorHandler:
    """Create and return an error handler instance."""
    return ErrorHandler()


def validate_business_input(business_question: str, answers: Dict[str, str] = None) -> Tuple[bool, List[str]]:
    """Quick validation function for business inputs."""
    handler = create_error_handler()
    
    input_data = {'business_question': business_question}
    if answers:
        input_data['answers'] = answers
    
    is_valid, errors = handler.validate_and_handle_input(input_data)
    
    # Convert errors to simple messages
    error_messages = [f"{error.severity.upper()}: {error.message}" for error in errors]
    
    return is_valid, error_messages


def recover_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Quick session recovery function."""
    recovery = SessionRecovery()
    return recovery.load_session_state(session_id)


def save_session_for_recovery(session_data: Dict[str, Any]) -> bool:
    """Quick session save function."""
    recovery = SessionRecovery()
    return recovery.save_session_state(session_data)