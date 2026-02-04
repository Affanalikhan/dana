"""
Session Management System for CRISP-DM Business Understanding Specialist

This module implements session state management, progress tracking, answer collection,
and batch progression control logic as specified in the design document.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
import uuid
from enum import Enum

from crisp_dm_framework import Dimension, Question, Answer, QuestionBatch, CRISPDMFramework
from context_preservation import ContextPreservationEngine


class SessionPhase(Enum):
    """Phases of a CRISP-DM session."""
    INITIALIZATION = "initialization"
    QUESTIONING = "questioning"
    ANALYSIS = "analysis"
    COMPLETE = "complete"


@dataclass
class SessionState:
    """
    Represents the complete state of a CRISP-DM business understanding session.
    Tracks progress, questions, answers, and current position in the framework.
    """
    session_id: str
    business_question: str
    current_dimension: Dimension
    current_batch: int
    questions_asked: List[Question] = field(default_factory=list)
    answers_collected: List[Answer] = field(default_factory=list)
    dimensions_completed: List[Dimension] = field(default_factory=list)
    total_progress: float = 0.0
    adaptive_flags: Dict[str, Any] = field(default_factory=dict)
    phase: SessionPhase = SessionPhase.INITIALIZATION
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate session state after initialization."""
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        if not self.business_question.strip():
            raise ValueError("Business question cannot be empty")
        if self.total_progress < 0 or self.total_progress > 100:
            raise ValueError("Total progress must be between 0 and 100")


class SessionManager:
    """
    Manages CRISP-DM session state, progress tracking, and batch progression.
    Provides session persistence and recovery mechanisms.
    """
    
    def __init__(self, framework: Optional[CRISPDMFramework] = None):
        """Initialize session manager with CRISP-DM framework."""
        self.framework = framework or CRISPDMFramework()
        self.context_engine = ContextPreservationEngine()
        self._sessions: Dict[str, SessionState] = {}
        self._current_session_id: Optional[str] = None
    
    def create_session(self, business_question: str) -> SessionState:
        """
        Create a new CRISP-DM session with the given business question.
        
        Args:
            business_question: The main business question to explore
            
        Returns:
            SessionState: The newly created session
        """
        if not business_question.strip():
            raise ValueError("Business question cannot be empty")
        
        session_id = str(uuid.uuid4())
        
        # Start with the first dimension
        first_dimension = self.framework.get_all_dimensions()[0]
        
        session = SessionState(
            session_id=session_id,
            business_question=business_question.strip(),
            current_dimension=first_dimension,
            current_batch=0,
            phase=SessionPhase.INITIALIZATION
        )
        
        self._sessions[session_id] = session
        self._current_session_id = session_id
        
        # Create context for this session
        self.context_engine.create_context(session_id)
        
        return session
    
    def get_current_state(self) -> Optional[SessionState]:
        """Get the current session state."""
        if self._current_session_id and self._current_session_id in self._sessions:
            return self._sessions[self._current_session_id]
        return None
    
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get a specific session by ID."""
        return self._sessions.get(session_id)
    
    def set_current_session(self, session_id: str) -> bool:
        """Set the current active session."""
        if session_id in self._sessions:
            self._current_session_id = session_id
            return True
        return False
    
    def record_answer(self, question_id: str, answer_text: str, 
                     confidence_level: Optional[str] = None) -> bool:
        """
        Record an answer to a question in the current session.
        
        Args:
            question_id: ID of the question being answered
            answer_text: The user's answer
            confidence_level: Optional confidence level
            
        Returns:
            bool: True if answer was recorded successfully
        """
        current_session = self.get_current_state()
        if not current_session:
            return False
        
        if not answer_text.strip():
            raise ValueError("Answer cannot be empty")
        
        # Validate that the question was actually asked
        question_ids = [q.id for q in current_session.questions_asked]
        if question_id not in question_ids:
            raise ValueError(f"Question {question_id} was not asked in this session")
        
        # Check if answer already exists for this question
        existing_answer_ids = [a.question_id for a in current_session.answers_collected]
        if question_id in existing_answer_ids:
            # Update existing answer
            for answer in current_session.answers_collected:
                if answer.question_id == question_id:
                    answer.response = answer_text.strip()
                    answer.timestamp = datetime.now().isoformat()
                    answer.confidence_level = confidence_level
                    break
        else:
            # Create new answer
            answer = Answer(
                question_id=question_id,
                response=answer_text.strip(),
                timestamp=datetime.now().isoformat(),
                confidence_level=confidence_level
            )
            current_session.answers_collected.append(answer)
        
        # Update session timestamp
        current_session.last_updated = datetime.now()
        
        # Add answer to context preservation system
        question = next((q for q in current_session.questions_asked if q.id == question_id), None)
        if question:
            answer_obj = next((a for a in current_session.answers_collected if a.question_id == question_id), None)
            if answer_obj:
                self.context_engine.add_answer_to_context(current_session.session_id, answer_obj, question)
        
        return True
    
    def get_next_batch(self) -> Optional[QuestionBatch]:
        """
        Get the next batch of questions for the current session.
        Implements progressive batch delivery logic.
        
        Returns:
            QuestionBatch: Next batch of questions, or None if session complete
        """
        current_session = self.get_current_state()
        if not current_session:
            return None
        
        # Check if current batch is complete (all questions answered)
        if not self._is_current_batch_complete():
            return None  # Must complete current batch first
        
        # Get answered question IDs for current dimension
        answered_in_dimension = self._get_answered_questions_for_dimension(
            current_session.current_dimension
        )
        
        # Try to get next batch for current dimension
        batch = self.framework.get_next_batch(
            current_session.current_dimension, 
            answered_in_dimension
        )
        
        if batch:
            # Enhance questions with context preservation
            enhanced_questions = []
            for question in batch.questions:
                contextual_question = self.context_engine.generate_contextual_question(
                    question, current_session.session_id
                )
                enhanced_questions.append(contextual_question)
            
            # Update batch with enhanced questions
            batch.questions = enhanced_questions
            
            # Generate context bridge if transitioning dimensions
            if (current_session.dimensions_completed and 
                current_session.current_dimension not in current_session.dimensions_completed):
                previous_dimension = current_session.dimensions_completed[-1] if current_session.dimensions_completed else None
                context_bridge = self.context_engine.generate_context_bridge(
                    current_session.session_id,
                    current_session.current_dimension,
                    previous_dimension
                )
                batch.context_bridge = context_bridge
            
            # Add questions to session
            current_session.questions_asked.extend(batch.questions)
            current_session.current_batch += 1
            current_session.phase = SessionPhase.QUESTIONING
            current_session.last_updated = datetime.now()
            return batch
        
        # Current dimension is complete, move to next dimension
        if self._advance_to_next_dimension():
            return self.get_next_batch()  # Recursive call for next dimension
        
        # All dimensions complete
        current_session.phase = SessionPhase.COMPLETE
        current_session.total_progress = 100.0
        return None
    
    def _is_current_batch_complete(self) -> bool:
        """Check if all questions in the current batch have been answered."""
        current_session = self.get_current_state()
        if not current_session:
            return True
        
        # If no questions asked yet, batch is "complete" (ready for first batch)
        if not current_session.questions_asked:
            return True
        
        # Get questions from current batch
        current_batch_questions = self._get_current_batch_questions()
        if not current_batch_questions:
            return True
        
        # Check if all current batch questions are answered
        answered_question_ids = [a.question_id for a in current_session.answers_collected]
        
        for question in current_batch_questions:
            if question.id not in answered_question_ids:
                return False
        
        return True
    
    def _get_current_batch_questions(self) -> List[Question]:
        """Get questions from the current batch."""
        current_session = self.get_current_state()
        if not current_session or not current_session.questions_asked:
            return []
        
        # Find questions from current dimension and batch
        current_batch_questions = []
        for question in reversed(current_session.questions_asked):
            if question.dimension == current_session.current_dimension:
                current_batch_questions.insert(0, question)
            else:
                break  # Stop when we hit a different dimension
        
        return current_batch_questions
    
    def _get_answered_questions_for_dimension(self, dimension: Dimension) -> List[str]:
        """Get list of answered question IDs for a specific dimension."""
        current_session = self.get_current_state()
        if not current_session:
            return []
        
        # Get all questions for this dimension
        dimension_questions = [
            q for q in current_session.questions_asked 
            if q.dimension == dimension
        ]
        dimension_question_ids = [q.id for q in dimension_questions]
        
        # Get answered questions for this dimension
        answered_in_dimension = [
            a.question_id for a in current_session.answers_collected
            if a.question_id in dimension_question_ids
        ]
        
        return answered_in_dimension
    
    def _advance_to_next_dimension(self) -> bool:
        """
        Advance to the next dimension in the sequence.
        
        Returns:
            bool: True if advanced to next dimension, False if all complete
        """
        current_session = self.get_current_state()
        if not current_session:
            return False
        
        # Mark current dimension as complete
        if current_session.current_dimension not in current_session.dimensions_completed:
            current_session.dimensions_completed.append(current_session.current_dimension)
        
        # Get next dimension
        next_dimension = self.framework.get_next_dimension(current_session.current_dimension)
        
        if next_dimension:
            current_session.current_dimension = next_dimension
            current_session.current_batch = 0
            
            # Update progress
            total_dimensions = len(self.framework.get_all_dimensions())
            completed_dimensions = len(current_session.dimensions_completed)
            current_session.total_progress = (completed_dimensions / total_dimensions) * 100
            
            current_session.last_updated = datetime.now()
            return True
        
        return False
    
    def get_session_progress(self) -> Dict[str, Any]:
        """
        Get detailed progress information for the current session.
        
        Returns:
            Dict containing progress metrics and status
        """
        current_session = self.get_current_state()
        if not current_session:
            return {"error": "No active session"}
        
        total_dimensions = len(self.framework.get_all_dimensions())
        completed_dimensions = len(current_session.dimensions_completed)
        
        # Calculate more detailed progress
        current_dimension_progress = 0
        if current_session.current_dimension:
            answered_in_current = len(self._get_answered_questions_for_dimension(
                current_session.current_dimension
            ))
            total_in_current = len(self.framework.get_dimension_questions(
                current_session.current_dimension
            ))
            if total_in_current > 0:
                current_dimension_progress = answered_in_current / total_in_current
        
        overall_progress = (completed_dimensions + current_dimension_progress) / total_dimensions * 100
        
        return {
            "session_id": current_session.session_id,
            "business_question": current_session.business_question,
            "phase": current_session.phase.value,
            "current_dimension": current_session.current_dimension.value,
            "current_batch": current_session.current_batch,
            "total_dimensions": total_dimensions,
            "completed_dimensions": completed_dimensions,
            "dimensions_completed": [d.value for d in current_session.dimensions_completed],
            "total_questions_asked": len(current_session.questions_asked),
            "total_answers_collected": len(current_session.answers_collected),
            "overall_progress": round(overall_progress, 1),
            "current_dimension_progress": round(current_dimension_progress * 100, 1),
            "is_current_batch_complete": self._is_current_batch_complete(),
            "created_at": current_session.created_at.isoformat(),
            "last_updated": current_session.last_updated.isoformat()
        }
    
    def export_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export session data for persistence or sharing.
        
        Args:
            session_id: Session to export, or current session if None
            
        Returns:
            Dict containing complete session data
        """
        if session_id:
            session = self.get_session(session_id)
        else:
            session = self.get_current_state()
        
        if not session:
            return {"error": "Session not found"}
        
        # Convert to serializable format
        export_data = {
            "session_id": session.session_id,
            "business_question": session.business_question,
            "current_dimension": session.current_dimension.value,
            "current_batch": session.current_batch,
            "questions_asked": [
                {
                    "id": q.id,
                    "dimension": q.dimension.value,
                    "text": q.text,
                    "reasoning": q.reasoning,
                    "question_type": q.question_type.value,
                    "dependencies": q.dependencies,
                    "options": q.options
                }
                for q in session.questions_asked
            ],
            "answers_collected": [
                {
                    "question_id": a.question_id,
                    "response": a.response,
                    "timestamp": a.timestamp,
                    "confidence_level": a.confidence_level,
                    "requires_followup": a.requires_followup
                }
                for a in session.answers_collected
            ],
            "dimensions_completed": [d.value for d in session.dimensions_completed],
            "total_progress": session.total_progress,
            "adaptive_flags": session.adaptive_flags,
            "phase": session.phase.value,
            "created_at": session.created_at.isoformat(),
            "last_updated": session.last_updated.isoformat()
        }
        
        return export_data
    
    def import_session(self, session_data: Dict[str, Any]) -> bool:
        """
        Import session data from exported format.
        
        Args:
            session_data: Session data in export format
            
        Returns:
            bool: True if import successful
        """
        try:
            # Reconstruct session from exported data
            session = SessionState(
                session_id=session_data["session_id"],
                business_question=session_data["business_question"],
                current_dimension=Dimension(session_data["current_dimension"]),
                current_batch=session_data["current_batch"],
                total_progress=session_data["total_progress"],
                adaptive_flags=session_data["adaptive_flags"],
                phase=SessionPhase(session_data["phase"]),
                created_at=datetime.fromisoformat(session_data["created_at"]),
                last_updated=datetime.fromisoformat(session_data["last_updated"])
            )
            
            # Reconstruct questions
            from crisp_dm_framework import QuestionType
            for q_data in session_data["questions_asked"]:
                question = Question(
                    id=q_data["id"],
                    dimension=Dimension(q_data["dimension"]),
                    text=q_data["text"],
                    reasoning=q_data["reasoning"],
                    question_type=QuestionType(q_data["question_type"]),
                    dependencies=q_data["dependencies"],
                    options=q_data["options"]
                )
                session.questions_asked.append(question)
            
            # Reconstruct answers
            for a_data in session_data["answers_collected"]:
                answer = Answer(
                    question_id=a_data["question_id"],
                    response=a_data["response"],
                    timestamp=a_data["timestamp"],
                    confidence_level=a_data["confidence_level"],
                    requires_followup=a_data["requires_followup"]
                )
                session.answers_collected.append(answer)
            
            # Reconstruct completed dimensions
            session.dimensions_completed = [
                Dimension(d) for d in session_data["dimensions_completed"]
            ]
            
            # Store session
            self._sessions[session.session_id] = session
            self._current_session_id = session.session_id
            
            return True
            
        except (KeyError, ValueError, TypeError) as e:
            return False
    
    def save_session_to_file(self, filepath: str, session_id: Optional[str] = None) -> bool:
        """
        Save session to JSON file.
        
        Args:
            filepath: Path to save file
            session_id: Session to save, or current session if None
            
        Returns:
            bool: True if save successful
        """
        try:
            session_data = self.export_session(session_id)
            if "error" in session_data:
                return False
            
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            return True
        except Exception:
            return False
    
    def load_session_from_file(self, filepath: str) -> bool:
        """
        Load session from JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            bool: True if load successful
        """
        try:
            with open(filepath, 'r') as f:
                session_data = json.load(f)
            
            return self.import_session(session_data)
        except Exception:
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all sessions with basic information.
        
        Returns:
            List of session summaries
        """
        sessions = []
        for session_id, session in self._sessions.items():
            sessions.append({
                "session_id": session_id,
                "business_question": session.business_question,
                "phase": session.phase.value,
                "progress": session.total_progress,
                "created_at": session.created_at.isoformat(),
                "last_updated": session.last_updated.isoformat(),
                "is_current": session_id == self._current_session_id
            })
        
        return sessions
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: ID of session to delete
            
        Returns:
            bool: True if deletion successful
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            
            # Clear current session if it was deleted
            if self._current_session_id == session_id:
                self._current_session_id = None
            
            return True
        
        return False
    def get_context_summary(self) -> Dict[str, Any]:
        """
        Get context summary for the current session.
        
        Returns:
            Dict containing context summary and detected conflicts
        """
        current_session = self.get_current_state()
        if not current_session:
            return {"error": "No active session"}
        
        # Get conversation summary
        context_summary = self.context_engine.get_conversation_summary(current_session.session_id)
        
        # Detect conflicts
        conflicts = self.context_engine.detect_context_conflicts(current_session.session_id)
        
        return {
            "context_summary": context_summary,
            "conflicts": [
                {
                    "type": conflict.type.value,
                    "question_ids": conflict.question_ids,
                    "description": conflict.description,
                    "severity": conflict.severity
                }
                for conflict in conflicts
            ]
        }
    
    def export_context(self) -> Dict[str, Any]:
        """Export context data for the current session."""
        current_session = self.get_current_state()
        if not current_session:
            return {"error": "No active session"}
        
        return self.context_engine.export_context(current_session.session_id)
    
    def import_context(self, context_data: Dict[str, Any]) -> bool:
        """Import context data."""
        return self.context_engine.import_context(context_data)