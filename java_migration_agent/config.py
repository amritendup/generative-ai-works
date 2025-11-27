import os
from dotenv import load_dotenv

load_dotenv()
# --- Java file placeholder ---
JAVA_CODE_PLACEHOLDER = "legacy_code/temp_file.java"

# --- Azure OpenAI Configuration ---
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# GPT-4o is recommended for complex reasoning/evaluation
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o") 
AZURE_OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION", "2024-12-01-preview")

# --- Agent Configuration ---
CHUNK_SIZE_LIMIT = 500  # Max characters per code chunk for the LLM
EVAL_REFINEMENT_ATTEMPTS = 1 # Max times to refine documentation

# --- Target Architecture Context ---
TARGET_ARCHITECTURE_PROMPT = (
    "The target is a Spring Boot 3.x application with Java 21. "
    "Use modern practices: constructor injection, JUnit 5 for tests, and "
    "Spring Data JPA for persistence (if applicable)."
)

def get_azure_llm():
    """Initializes the Azure ChatOpenAI model."""
    from langchain_openai import AzureChatOpenAI
    
    if not all([AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME]):
        raise ValueError("Azure OpenAI configuration missing in .env or config.py")

    return AzureChatOpenAI(
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_API_KEY,
        deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME,
        temperature=0.1,
    )