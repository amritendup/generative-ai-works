# src/workflow.py

from langgraph.graph import StateGraph, END
from .agents import (
    parse_and_split_code, 
    generate_documentation, 
    evaluate_documentation, 
    generate_migrated_code, 
    generate_unit_tests,
    refine_migrated_code,
    update_chunk_index
)
from .state_models import MigrationState, StateDict
from config import EVAL_REFINEMENT_ATTEMPTS

# --- Conditional Edge Functions ---

def should_refine_docs(state: StateDict) -> str:
    """Step 3 & 4 Loop Condition: Re-run documentation or proceed to migration."""
    migration_state = MigrationState(**state)
    chunk = migration_state.chunks[migration_state.current_chunk_index]
    
    # Condition 1: Evaluation failed AND we are under the attempt limit (e.g., attempt 1 or 2)
    if chunk.status == "EVAL_FAIL" and chunk.refinement_count < EVAL_REFINEMENT_ATTEMPTS:
        print(f"Decision: Evaluation failed. Rerunning documentation (Attempt {chunk.refinement_count}/{EVAL_REFINEMENT_ATTEMPTS}).")
        return "rerun_documentation" 
    else:
        # Condition 2: Passed or limit reached (e.g., attempt 3)
        print("Decision: Documentation approved or refinement limit reached. Proceeding to migration.")
        chunk.status = "EVAL_PASS" # Ensure status is set correctly before migration
        migration_state.chunks[migration_state.current_chunk_index] = chunk
        return "proceed_migration" 

def should_continue_chunk_loop(state: StateDict) -> str:
    """Conditional edge logic: Check if there are more chunks to process."""
    migration_state = MigrationState(**state)
    
    # Increment the chunk index for the loop update node
    # Note: We must update the index after the final node of the inner loop completes
    #if migration_state.current_chunk_index < len(migration_state.chunks):
        #migration_state.current_chunk_index += 1
        
    if migration_state.current_chunk_index < len(migration_state.chunks):
        return "process_next_chunk"
    else:
        return "end_workflow"

def build_workflow():
    """Builds the LangGraph workflow with the corrected 8-step flow."""
    workflow = StateGraph(StateDict)

    # 1. Define Nodes
    workflow.add_node("splitter", parse_and_split_code)
    workflow.add_node("generate_documentation", generate_documentation) 
    workflow.add_node("evaluate_documentation", evaluate_documentation) 
    workflow.add_node("generate_migrated_code", generate_migrated_code) 
    workflow.add_node("generate_unit_tests", generate_unit_tests) 
    workflow.add_node("refine_code", refine_migrated_code) 
    workflow.add_node("update_index", update_chunk_index) # NEW NODE

    # 2. Define Edges
    
    # Initialization
    workflow.set_entry_point("splitter")
    workflow.add_edge("splitter", "generate_documentation") # 1 -> 2

    # Documentation/Evaluation Loop (Steps 2, 3, 4)
    workflow.add_edge("generate_documentation", "evaluate_documentation") # 2 -> 3
    
    # Conditional edge controls the refinement loop count
    workflow.add_conditional_edges(
        "evaluate_documentation",
        should_refine_docs, # This function checks if count < limit
        {
            "rerun_documentation": "generate_documentation", # Step 4: Loop back
            "proceed_migration": "generate_migrated_code"   # Step 5: Exit loop
        }
    )

    # Post-Migration Steps (Steps 5, 6, 7)
    workflow.add_edge("generate_migrated_code", "generate_unit_tests") # 5 -> 6
    workflow.add_edge("generate_unit_tests", "refine_code") # 6 -> 7
    
    # Chunk Completion and Index Update
    workflow.add_edge("refine_code", "update_index") # 7 -> Index Update

    # Final Loop Check: Process next chunk or END
    workflow.add_conditional_edges(
        "update_index",
        should_continue_chunk_loop,
        {
            "process_next_chunk": "generate_documentation", # Start Step 2 for next chunk
            "end_workflow": END
        }
    )

    return workflow.compile()