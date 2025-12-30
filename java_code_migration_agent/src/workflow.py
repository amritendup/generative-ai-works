# src/workflow.py (Revised)

from langgraph.graph import StateGraph, END
from .state_models import MigrationState
from .agents import (
    parse_and_split_code, generate_documentation, evaluate_documentation, 
    generate_migrated_code, generate_unit_tests, refine_migrated_code, 
    dashboard_generation
)
from .workflow_router import chunk_batch_router


def build_workflow():
    """
    Constructs and compiles the Java Code Migration LangGraph workflow.
    """
    # 1. Define the Graph
    workflow = StateGraph(MigrationState)

    # 2. Add Nodes for all required steps
    workflow.add_node("parse_and_split_code", parse_and_split_code)
    workflow.add_node("generate_documentation", generate_documentation)
    workflow.add_node("evaluate_documentation", evaluate_documentation)
    workflow.add_node("chunk_batch_router", chunk_batch_router)
    workflow.add_node("generate_migrated_code", generate_migrated_code)
    workflow.add_node("generate_unit_tests", generate_unit_tests)
    workflow.add_node("refine_migrated_code", refine_migrated_code)
    workflow.add_node("dashboard_generation", dashboard_generation)

    # 3. Define the Entry Point
    workflow.set_entry_point("parse_and_split_code")

    # 4. Define Edges (using the logic provided previously)
    
    # --- Phase 0: Splitting ---
    workflow.add_edge("parse_and_split_code", "generate_documentation")

    # --- Phase 1: DOCUMENTATION & EVALUATION BATCH ---
    workflow.add_edge("generate_documentation", "evaluate_documentation")
    workflow.add_conditional_edges(
        "evaluate_documentation",
        lambda state: state.chunks[state.current_chunk_index].status,
        {
            "EVAL_FAIL": "generate_documentation",
            "EVAL_PASS": "chunk_batch_router",
        },
    )

    # --- Phase 2 & 3: CODE GEN, TEST, REFINE BATCH ---
    workflow.add_edge("generate_migrated_code", "chunk_batch_router")
    workflow.add_edge("generate_unit_tests", "refine_migrated_code")
    workflow.add_edge("refine_migrated_code", "chunk_batch_router") 
    # --- Phase 4: FINALIZATION ---
    workflow.add_edge("dashboard_generation", END)

    # --- Conditional Router (The core of the batch process) ---
    workflow.add_conditional_edges(
        "chunk_batch_router",
        lambda state: MigrationState.model_validate(state).current_phase,
        {
            "DOC_EVAL": "generate_documentation",
            "CODE_GEN": "generate_migrated_code",
            "TEST_REF": "generate_unit_tests", 
            "DASHBOARD": "dashboard_generation"
        }
    )

    # 5. Compile the workflow and return the app object
    app = workflow.compile()
    
    # Return the compiled app object
    return app