import os
import zipfile
import io
import javalang
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from .state_models import MigrationState, CodeChunk, EvaluationReport, StateDict
from config import get_azure_llm, TARGET_ARCHITECTURE_PROMPT, CHUNK_SIZE_LIMIT, EVAL_REFINEMENT_ATTEMPTS

llm = get_azure_llm().with_structured_output(EvaluationReport)
parser = JsonOutputParser(pydantic_object=EvaluationReport)
JAVA_CODE_PLACEHOLDER = os.path.join("legacy_code", "temp_file.java")

# --- 1. Code Splitter Agent ---
def parse_and_split_code(state: StateDict) -> StateDict:
    """Activity 1: Parsing and Splitting Large Code Files."""
    print("Agent: Splitting code...")
    migration_state = MigrationState(**state)
    
    if migration_state.chunks:
        print("Code already split. Skipping splitter.")
        return state

    # Placeholder for actual Java parsing logic (e.g., using Tree-sitter)
    # For a simple demo, we load a placeholder file and split by class boundary (very simplified).
    with open(JAVA_CODE_PLACEHOLDER, 'r',encoding='utf-8') as f:
        file_content = f.read()

    # Simple logic: treat the entire file as one chunk or split by class definitions
    # In a real app, this would use AST parsing for intelligent splitting.
    
    new_chunks = []
    
    try:
        # 2. Parse the Java file content into an Abstract Syntax Tree (AST)
        tree = javalang.parse.parse(file_content)
        
        # 3. Iterate over the class declarations in the AST
        chunk_index = 0
        for path, node in tree:
            if isinstance(node, javalang.tree.ClassDeclaration) or \
               isinstance(node, javalang.tree.InterfaceDeclaration):
                
                # Javalang provides token information (start and end lines)
                # You need to extract the raw source code lines corresponding to this class.
                start_line = node.position.line
                # Finding the end line requires a bit more advanced AST traversal, 
                # but for simplicity, we'll mark the block here.
                
                # --- ACTUAL CODE EXTRACTION ---
                # This complex step involves mapping AST nodes back to the original source lines
                # and extracting the exact text block for the class.
                # Example: Extract the lines of the file content that define this class.
                
                # Mock extraction (Replace with actual extraction logic)
                class_name = node.name
                class_content = f"// Extracted content for {class_name}...\n" 
                
                new_chunks.append(CodeChunk(
                    file_path=JAVA_CODE_PLACEHOLDER,
                    chunk_index=chunk_index,
                    original_content=class_content, # The extracted code for the class
                    status="SPLIT"
                ))
                chunk_index += 1

    except javalang.tokenizer.LexerError as e:
        raise ValueError(f"Failed to parse Java file due to syntax error: {e}")
    
    # 4. Update the state with all generated chunks
    if not new_chunks:
         # Fallback: if no classes found (e.g., utility file), treat as one chunk
         new_chunks.append(CodeChunk(
            file_path=JAVA_CODE_PLACEHOLDER, chunk_index=0, original_content=file_content, status="SPLIT"
         ))
        
    migration_state.chunks = new_chunks

    print(f"Code Split into {len(migration_state.chunks)} chunks.")
    return migration_state.model_dump()

# --- 2. Documentation Agent ---
def generate_documentation(state: StateDict) -> StateDict:
    """Activity 2: Analyzing and Documenting Legacy Code."""
    print("Agent: Generating documentation...")
    migration_state = MigrationState(**state)
    chunk = migration_state.chunks[migration_state.current_chunk_index]
    
    refinement_instruction = ""
    if chunk.refinement_count > 0 and chunk.eval_report:
        # Refinement loop active
        refinement_instruction = f"IMPORTANT: Refine the existing documentation based on this feedback: '{chunk.eval_report.refinement_suggestion}'."
        refinement_instruction = refinement_instruction.replace('{', '{{').replace('}', '}}')
        print(f"**Refining documentation (Attempt {chunk.refinement_count})**")
    sanitized_documentation = chunk.documentation.replace('{', '{{').replace('}', '}}')
    sanitized_original_content = chunk.original_content.replace('{', '{{').replace('}', '}}')
    # Prompt for documentation
    doc_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a Senior Java Architect. Analyze the provided legacy Java code. "
         "Generate comprehensive documentation in **Markdown format** that includes:\n"
         "1. **Component Name/Class**\n"
         "2. **External Dependencies** (Libraries, other classes)\n"
         "3. **Processing Logic Summary** (Step-by-step description of business rules and function).\n"
         "The output must only be the Markdown content. "
         f"{refinement_instruction}"
        ),
        ("human", f"Legacy Java Code:\n```java\n{sanitized_original_content}\n```\nExisting Documentation (if refining):\n{sanitized_documentation}")
    ])
    
    chain = doc_prompt | get_azure_llm()
    
    # In a refinement, provide the old doc, otherwise start fresh
    input_data = {"chunk": chunk.original_content}
    if chunk.refinement_count > 0:
        input_data["documentation"] = chunk.documentation
        
    response = chain.invoke(input_data)
    
    # Update state
    chunk.documentation = response.content
    chunk.status = "DOC_COMPLETE"
    migration_state.chunks[migration_state.current_chunk_index] = chunk
    return migration_state.model_dump()

# --- 3. Evaluation Agent (LLM-as-a-Judge) ---
def evaluate_documentation(state: StateDict) -> StateDict:
    """Activity 3: Evaluating and Refining Generated Markdown using GPT-4o."""
    print("Agent: Evaluating documentation (LLM-as-a-Judge)...")
    migration_state = MigrationState(**state)
    chunk = migration_state.chunks[migration_state.current_chunk_index]
    
    # System prompt for LLM-as-a-Judge
    eval_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are an **Automated Documentation Verifier**. Your task is to compare the provided **Legacy Java Code** and its **Markdown Documentation**.\n"
         "You must output a strictly structured JSON object using the provided schema.\n"
         "CRITERIA:\n"
         "1. **Completeness Score (1-5):** Does the documentation cover ALL components, dependencies, and business logic described in the code?\n"
         "2. **Accuracy Score (1-5):** Is every statement in the documentation factually correct and free of contradiction with the code?\n"
         "Refinement is needed if any score is below 4. Set 'manual_cross_check' to true if any score is 1 or 2, regardless of refinement.\n"
         f"If refinement is needed, provide a detailed 'refinement_suggestion'."
        ),
        ("human", 
         f"--- Legacy Java Code ---\n```java\n{chunk.original_content}\n```\n\n"
         f"--- Markdown Documentation ---\n{chunk.documentation}"
        )
    ])
    
    # Use the LLM with Pydantic structured output
    eval_chain = eval_prompt | get_azure_llm().with_structured_output(EvaluationReport)

    # Invoke and parse the structured report
    raw_report = eval_chain.invoke({}) # Input is baked into the prompt
    report = EvaluationReport.model_validate(raw_report)
    
    # Update state
    chunk.eval_report = report
    chunk.refinement_count += 1
    
    if report.refinement_needed and chunk.refinement_count < EVAL_REFINEMENT_ATTEMPTS:
        chunk.status = "EVAL_FAIL"  # Trigger refinement
        print(f"Documentation failed evaluation. Refinement required: {report.refinement_suggestion}")
    else:
        # Final status after successful eval or exhausting attempts
        chunk.status = "EVAL_PASS" 
        print(f"Documentation passed or refinement limit reached. Status: {chunk.status}")

    migration_state.chunks[migration_state.current_chunk_index] = chunk
    return migration_state.model_dump()

# --- 4. Migration Agent ---
def generate_migrated_code(state: StateDict) -> StateDict:
    """Activity 4: Create code in Spring boot with Java 21."""
    print("Agent: Generating migrated code...")
    migration_state = MigrationState(**state)
    chunk = migration_state.chunks[migration_state.current_chunk_index]

    code_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         f"You are a Java 21 Spring Boot Code Generator. Convert the **Legacy Java Code** into a modern, runnable Spring Boot 3.x class. "
         f"The architecture context is: {TARGET_ARCHITECTURE_PROMPT}. "
         "The migrated class must use the features/logic detailed in the **Validated Documentation**.\n"
         "Output only the complete Java code block, including necessary imports. Do not include explanation."
        ),
        ("human", 
         f"--- Validated Documentation ---\n{chunk.documentation}\n\n"
         f"--- Legacy Java Code (for reference) ---\n```java\n{chunk.original_content}\n```"
        )
    ])

    chain = code_prompt | get_azure_llm()
    response = chain.invoke({})
    
    # Extract Java code block (simple regex or strip Markdown wrapper)
    migrated_code = response.content.strip().strip('```java').strip('```').strip()

    # Update state
    chunk.migrated_code = migrated_code.replace('{', '{{').replace('}', '}}')
    chunk.status = "MIG_COMPLETE"
    migration_state.chunks[migration_state.current_chunk_index] = chunk
    return migration_state.model_dump()

# --- 5. Unit Test Agent ---
def generate_unit_tests(state: StateDict) -> StateDict:
    """Activity 5: Create unit testcases in Junit 5."""
    print("Agent: Generating JUnit 5 unit tests...")
    migration_state = MigrationState(**state)
    chunk = migration_state.chunks[migration_state.current_chunk_index]

    test_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "You are a JUnit 5 Test Expert. Generate comprehensive unit tests for the provided **Migrated Java Code**. "
         "Tests must use **JUnit 5** and **Mockito** for mocking dependencies. "
         "Ensure high coverage: test the main logic, edge cases, and error paths as described in the **Documentation**."
         "Output only the complete Java test code block, including necessary imports. Do not include explanation."
        ),
        ("human", 
         f"--- Migrated Java Code ---\n```java\n{chunk.migrated_code}\n```\n\n"
         f"--- Documentation ---\n{chunk.documentation}"
        )
    ])

    chain = test_prompt | get_azure_llm()
    response = chain.invoke({})
    
    # Extract Java code block
    unit_tests = response.content.strip().strip('```java').strip('```').strip()

    # Update state
    chunk.unit_tests = unit_tests
    chunk.status = "TEST_COMPLETE"
    migration_state.chunks[migration_state.current_chunk_index] = chunk
    return migration_state.model_dump()

# --- Utility Node ---
def update_chunk_progress(state: StateDict) -> StateDict:
    """Moves the workflow to the next chunk or completes."""
    migration_state = MigrationState(**state)
    
    if migration_state.current_chunk_index < len(migration_state.chunks):
        chunk = migration_state.chunks[migration_state.current_chunk_index]
        chunk.status = "DONE"
        migration_state.current_chunk_index += 1
        
    if migration_state.current_chunk_index >= len(migration_state.chunks):
        migration_state.overall_status = "MIGRATION_COMPLETE"
        
    return migration_state.model_dump()