from pydantic import BaseModel, Field
from typing import List, Dict, Any, Literal

class EvaluationReport(BaseModel):
    """Structured output for the LLM-as-a-Judge."""
    completeness_score: Literal[1, 2, 3, 4, 5] = Field(description="Score 1-5 for completeness against the source code.")
    accuracy_score: Literal[1, 2, 3, 4, 5] = Field(description="Score 1-5 for accuracy against the source code.")
    refinement_needed: bool = Field(description="True if refinement is required (e.g., if any score is < 4).")
    refinement_suggestion: str = Field(description="Detailed feedback on how to improve the documentation.")
    manual_cross_check: bool = Field(description="True if final score is low, indicating necessary human review.")

class CodeChunk(BaseModel):
    """Represents a segment of the legacy code and its associated artifacts."""
    file_path: str
    chunk_index: int
    original_content: str
    
    
    # Generated artifacts
    documentation: str = Field(default="", description="Markdown documentation for this chunk.")
    migrated_code: str = Field(default="", description="Java 21 Spring Boot code.")
    refined_code: str = Field(default="", description="Refactored and final Java code.")
    unit_tests: str = Field(default="", description="JUnit 5 test cases.")
    
    # Evaluation tracking
    eval_report: EvaluationReport | None = None
    refinement_count: int = 0
    
    status: Literal[
        "PENDING", "SPLIT", "DOC_COMPLETE", "EVAL_FAIL", "EVAL_PASS", 
        "MIG_COMPLETE", "TEST_COMPLETE", "REF_COMPLETE", "DONE" 
    ] = "PENDING"
    
class MigrationState(BaseModel):
    """The central state object for the LangGraph workflow."""
    chunks: List[CodeChunk] = Field(default_factory=list)
    unparsed_file_content: Dict[str, str] = Field(default_factory=dict)
    current_chunk_index: int = 0
    migration_summary: str = "" 
    overall_status: str = "INIT"

# The state dictionary required by LangGraph
# Note: LangGraph often uses a simple dictionary state, but Pydantic helps structure the content.
StateDict = Dict[str, Any]