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

# --- Ollama Configuration for Evaluation ---
OLLAMA_MODEL = "llama3:8b-instruct-q4_K_M" #"phi3:mini"
OLLAMA_BASE_URL = "http://localhost:11434" # Default Ollama API endpoint

# --- Agent Configuration ---
CHUNK_SIZE_LIMIT = 6000  # Max characters per code chunk for the LLM (reduced to avoid filtering)
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
        temperature=0.0,
        top_p=0.1,
    )
    
'''

def get_ollama_llm():
    """Initializes the local Ollama LLM (used for Evaluation)."""
    from langchain_community.chat_models import ChatOllama
    
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
    )

'''