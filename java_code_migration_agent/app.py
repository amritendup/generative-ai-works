import os
from typing import List
import streamlit as st
import pandas as pd
import zipfile
import io
from dotenv import load_dotenv
from langchain_core.exceptions import OutputParserException
from src.workflow import build_workflow
from src.state_models import MigrationState, CodeChunk
from config import JAVA_CODE_PLACEHOLDER 

load_dotenv()
workflow = build_workflow()

# --- Helper Functions ---
def run_full_migration(initial_state: MigrationState):
    """Executes the LangGraph workflow."""
    st.session_state['status'] = "Running..."
    st.session_state['chunks'] = initial_state.chunks
    
    # Reset internal state before running
    # Note: LangGraph often uses thread_id for persistence, but we use SessionState for Streamlit demo
    
    try:
        # Initial run of the splitter node
        initial_state_dict = initial_state.model_dump()
        config = {
        # CRITICAL: This line sets the recursion limit for the entire workflow run.
            "recursion_limit": 1000 
        }

        # We run the whole compiled graph from the splitter node to the END
        # LangGraph automatically handles the state transitions and loops.
        final_state = workflow.invoke(initial_state_dict, config=config)
        
        st.session_state['final_state'] = final_state
        st.session_state['status'] = "MIGRATION_COMPLETE"
        st.success("✅ Migration workflow completed successfully!")

    except OutputParserException as e:
        st.session_state['status'] = "EVALUATION_ERROR"
        st.error(f"LLM Structured Output Error (Validation Failed): {e}")
    except Exception as e:
        st.session_state['status'] = "FATAL_ERROR"
        msg = str(e).lower()
        if any(token in msg for token in ("content filter", "moderation", "has not provided the response", "content was filtered")):
            st.error("A fatal error occurred: the model response was blocked by content moderation. Try sanitizing sensitive terms or reduce input size. See logs/llm_filter.log for sanitized snapshots and diagnostics.")
        else:
            st.error(f"A fatal error occurred during the workflow: {e}")

# --- Streamlit UI Components ---

st.set_page_config(layout="wide", page_title="Java Legacy to Spring Boot 21 Migration Agent")

st.title("🤖 Legacy Java Code Modernization Agent")
st.markdown("Use this multi-agent solution to automate the conversion of legacy Java code to a modern Java 21 Spring Boot application.")

# --- 1. Input and Initiation ---
with st.sidebar:
    st.header("1. Input & Settings")
    
    uploaded_file = st.file_uploader("Upload Legacy Java Source Code (.zip or .java)", type=["zip", "java"])
    
    # Simple configuration check
    if st.button("Start Migration Workflow"):
        if not uploaded_file:
            st.error("Please upload a code file to start.")
        else:
            # Simple file handling (for demo, we simulate placing the file content)
            if uploaded_file.name.endswith('.java'):
                 file_content = uploaded_file.read().decode('utf-8')
                 # Write content to placeholder file for the splitter agent to read
                 with open(JAVA_CODE_PLACEHOLDER, 'w',encoding='utf-8') as f:
                     f.write(file_content)
                 
                 initial_state = MigrationState(
                     unparsed_file_content={uploaded_file.name: file_content},
                     overall_status="READY"
                 )
                 
                 # Kick off the workflow
                 run_full_migration(initial_state)
            else:
                 st.error("Only single .java file is supported in this demo.")


# Initialize session state for persistence
if 'status' not in st.session_state:
    st.session_state['status'] = "WAITING_FOR_INPUT"
if 'final_state' not in st.session_state:
    st.session_state['final_state'] = None

st.subheader("Current Status")
status_map = {
    "WAITING_FOR_INPUT": ("⏳", "Awaiting file upload..."),
    "READY": ("⚙️", "Configuration successful, ready to run."),
    "Running...": ("🔄", "Workflow in progress. Please wait..."),
    "MIGRATION_COMPLETE": ("✅", "Migration complete! Review results below."),
    "EVALUATION_ERROR": ("❌", "Evaluation failed due to structured output error."),
    "FATAL_ERROR": ("🚨", "A critical error stopped the process.")
}
status_icon, status_text = status_map.get(st.session_state['status'], ("❓", "Unknown status."))
st.metric(label="Workflow Status", value=status_icon + " " + status_text)

st.markdown("---")

# --- 2. Final Evaluation Dashboard (Activity 3) ---
st.header("📊 Final Evaluation Report Dashboard")

if st.session_state['final_state']:
    final_state = MigrationState(**st.session_state['final_state'])
    
    # Create the DataFrame for the dashboard
    data = []
    for chunk in final_state.chunks:
        report = chunk.eval_report
        data.append({
            "File / Component": f"{chunk.file_path} (Chunk {chunk.chunk_index})",
            "Completeness Score": report.completeness_score if report else 'N/A',
            "Accuracy Score": report.accuracy_score if report else 'N/A',
            "Refinement Attempts": chunk.refinement_count - 1 if chunk.refinement_count > 0 else 0,
            "Manual Cross-Check Required": "⚠️ YES" if report and report.manual_cross_check else "🟢 NO",
            "Final Status": chunk.status
        })

    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True)

    st.info("⚠️ **Manual Cross-Check Required** is flagged if the final LLM-as-a-Judge score was low (1 or 2), regardless of automated refinement attempts.")

    st.markdown("---")
    
    # --- 3. Generated Code and Download (Activities 4 & 5) ---
    st.header("📦 Generated Artifacts")

    # Download Button Logic
    @st.cache_data
    def create_zip(chunks: List[CodeChunk]):
        zip_io = io.BytesIO()
        with zipfile.ZipFile(zip_io, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for chunk in chunks:
                base_name = os.path.splitext(os.path.basename(chunk.file_path))[0]
                # 1. Migrated Code
                zip_file.writestr(f"src/main/java/com/migrated/{base_name}.java", chunk.migrated_code)
                # 2. Unit Tests
                zip_file.writestr(f"src/test/java/com/migrated/{base_name}Test.java", chunk.unit_tests)
                # 3. Documentation
                zip_file.writestr(f"docs/{base_name}.md", chunk.documentation)
        return zip_io.getvalue()

    zip_data = create_zip(final_state.chunks)
    
    st.download_button(
        label="Download Full Spring Boot Project (.zip)",
        data=zip_data,
        file_name="spring_boot_migration_project.zip",
        mime="application/zip",
        help="Includes all migrated Java files, JUnit 5 test cases, and Markdown documentation."
    )
    
    # Optional: Display content of the first chunk
    if final_state.chunks:
        first_chunk = final_state.chunks[0]
        with st.expander("Review Migrated Code (Example Chunk)"):
            st.code(first_chunk.migrated_code, language="java")
        with st.expander("Review Generated Unit Tests (Example Chunk)"):
            st.code(first_chunk.unit_tests, language="java")
        with st.expander("Review Final Documentation (Example Chunk)"):
            st.markdown(first_chunk.documentation)

else:
    st.info("The dashboard and code artifacts will appear here after the migration workflow is executed.")