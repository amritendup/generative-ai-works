from langgraph.graph import StateGraph, END
from .agents import (
    parse_and_split_code, 
    generate_documentation, 
    evaluate_documentation, 
    generate_migrated_code, 
    generate_unit_tests
)
from .state_models import MigrationState, StateDict
from config import EVAL_REFINEMENT_ATTEMPTS

def should_refine(state: StateDict) -> str:
    """Conditional edge logic: Determine if documentation needs refinement or if we proceed."""
    migration_state = MigrationState(**state)
    chunk = migration_state.chunks[migration_state.current_chunk_index]
    
    if chunk.status == "EVAL_FAIL" and chunk.refinement_count < EVAL_REFINEMENT_ATTEMPTS:
        return "refine"  # Go back to documentation
    else:
        return "migrate" # Proceed to migration (either passed or max attempts reached)

def should_continue_loop(state: StateDict) -> str:
    """Conditional edge logic: Check if there are more chunks to process."""
    migration_state = MigrationState(**state)
    if migration_state.current_chunk_index < len(migration_state.chunks):
        return "process_chunk"
    else:
        return "end"

def build_workflow():
    """Builds the LangGraph workflow."""
    workflow = StateGraph(StateDict)

    # 1. Define Nodes (The Agents)
    workflow.add_node("splitter", parse_and_split_code)
    workflow.add_node("documenter", generate_documentation)
    workflow.add_node("evaluator", evaluate_documentation)
    workflow.add_node("migrator", generate_migrated_code)
    workflow.add_node("tester", generate_unit_tests)
    # This node simply updates the index to move to the next chunk
    workflow.add_node("next_chunk", lambda s: MigrationState(**s).model_dump()) 

    # 2. Define Edges (The Flow)
    
    # Initial Start: Only run split once, then enter the loop
    workflow.set_entry_point("splitter")
    workflow.add_edge("splitter", "documenter")

    # The Core Chunk Processing Loop: document -> evaluate -> (refine OR migrate) -> test -> next chunk
    workflow.add_edge("documenter", "evaluator")
    
    # Conditional Edge: Refine or Migrate
    workflow.add_conditional_edges(
        "evaluator",
        should_refine,
        {
            "refine": "documenter", # Loop back for refinement
            "migrate": "migrator"
        }
    )
    
    workflow.add_edge("migrator", "tester")
    
    # After testing, update the index and check if the loop should continue
    workflow.add_edge("tester", "next_chunk")
    
    # Loop Check: Continue processing the next chunk or END
    workflow.add_conditional_edges(
        "next_chunk",
        should_continue_loop,
        {
            "process_chunk": "documenter", # Start processing the new current chunk
            "end": END
        }
    )

    return workflow.compile()