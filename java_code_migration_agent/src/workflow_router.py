# src/workflow_router.py
from .state_models import MigrationState

def chunk_batch_router(state: dict) -> dict:
    """
    Manages iteration through chunks within a phase and transitions between phases.
    It updates the state (index/phase) and returns the state dict for conditional routing.
    """
    # Use the robust Pydantic instantiation
    migration_state = MigrationState.model_validate(state) 
    
    current_index = migration_state.current_chunk_index
    total_chunks = migration_state.total_chunks
    current_phase = migration_state.current_phase

    # 1. Check for Iteration (Loop within the current phase)
    if current_index < total_chunks - 1:
        # Update index and remain in the current phase
        migration_state.current_chunk_index += 1
        print(f"Router: Advancing to Chunk {migration_state.current_chunk_index} in phase {current_phase}")
        
    # 2. Check for Phase Completion (Switch to the next phase)
    else:
        # Reset index for the next phase
        migration_state.current_chunk_index = 0
        
        if current_phase == 'DOC_EVAL':
            print("\nRouter: DOCUMENTATION/EVALUATION Phase COMPLETE. Starting CODE_GEN phase.")
            migration_state.current_phase = 'CODE_GEN'
            
        elif current_phase == 'CODE_GEN':
            print("\nRouter: CODE GENERATION Phase COMPLETE. Starting TEST_REF phase.")
            migration_state.current_phase = 'TEST_REF'
            
        elif current_phase == 'TEST_REF':
            print("\nRouter: TEST/REFINE Phase COMPLETE. Starting DASHBOARD phase.")
            migration_state.current_phase = 'DASHBOARD'
            
    # CRITICAL: Always return the updated state dictionary.
    return migration_state.model_dump()