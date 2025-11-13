# -------------------------------------------------------------------
# An Email Compliance Agent for BFSI using GPT-4o + LangChain + RAG
# -------------------------------------------------------------------

import os
from dotenv import load_dotenv
import json
import pandas as pd
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.agents import create_agent
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceEmbeddings

# ========== CONFIG ==========
#===================LOAD ENV VARIABLES================
load_dotenv()
# Retrieve Azure OpenAI specific configuration from environment variables




EMAIL_FILE = "bfsi_email_data.csv"
OUTPUT_FILE = "compliance_results.csv"

# ========== STEP 1: LOAD AND NORMALIZE EMAILS ==========
def load_emails(file_path):
    df = pd.read_csv(file_path)
    df.fillna("", inplace=True)
    df["normalized_text"] = df["Subject"].str.strip() + "\n" + df["Body"].str.strip()
    return df

# ========== STEP 2: CREATE CATEGORY VECTOR DB ==========
categories = [
    {"category": "Secrecy", "priority": "High", "weightage": 0.9, "desc": "Disclosure of confidential client or trading information."},
    {"category": "Market manipulation/ Misconduct", "priority": "Critical", "weightage": 1.0, "desc": "Attempt to influence market unfairly or insider trading."},
    {"category": "Market bribery", "priority": "High", "weightage": 0.85, "desc": "Offering incentives for market influence."},
    {"category": "Change in communication", "priority": "Medium", "weightage": 0.6, "desc": "Using unauthorized or informal communication channels."},
    {"category": "Complaints", "priority": "Medium", "weightage": 0.5, "desc": "Customer or employee grievances."},
    {"category": "Employee ethics", "priority": "Medium", "weightage": 0.7, "desc": "Integrity, honesty, and professional misconduct."}
]

def build_vector_db():
    docs = [
        Document(
            page_content=f"Category: {c['category']}\nPriority: {c['priority']}\nWeightage: {c['weightage']}\nDescription: {c['desc']}"
        )
        for c in categories
    ]

    # LOAD A SMALL, OPENSOURCE EMBEDDING MODEL
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="category_db")
    return vectorstore.as_retriever(search_kwargs={"k": 3})

# ========== STEP 3: COMPLIANCE CHECKER USING GPT-4o ==========

llm = AzureChatOpenAI(
    model="gpt-4o",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),

    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_version=os.getenv("OPENAI_API_VERSION"),
    openai_api_type=os.getenv("OPENAI_API_TYPE"),
    deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    temperature=0.0 
)


compliance_prompt = ChatPromptTemplate.from_template("""
You are an email compliance auditor who thoroughly checks email contents against BFSI related regulatory compliance categories.

Email Content:
{email_text}

Category Context (Descriptions and Definitions):
{context}

Task:
1. Determine if the email is Compliant or Non-Compliant.
2. List applicable non-compliance categories (can be multiple).
3. Identify violating or suspicious lines if any.
4. Provide a confidence level (0 to 1) for your judgment.
5. There will not be any categories if the email is Compliant.

Return JSON in the following format strictly:
{{
  "compliance": "<Compliant or Non-Compliant>",
  "categories": ["<category1>", "<category2>"],
  "violations": ["<suspicious line>"],
  "confidence": <float between 0 and 1>
}}
""")

def check_compliance(email_text, context_text):
    full_prompt = compliance_prompt.format(email_text=email_text, context=context_text)
       
    response = llm.invoke(full_prompt)
    
    try:
        parsed = json.loads(response.content.removeprefix("```json").removesuffix("```").strip())
    except Exception:
        print("Error parsing LLM response:", response.content)
        parsed = {"compliance": "Error", "categories": [], "violations": [], "confidence": 0.0}
    return parsed

# ========== STEP 4: PROGRAMMATIC SCORING ==========
def compute_score(parsed_result):
    """
    Score = confidence × (1 - average(weightage of violating categories))
    Lower score = more risky
    """
    conf = parsed_result.get("confidence", 0)
    if not parsed_result.get("categories"):
        return round(conf, 2)
    weights = [c["weightage"] for c in categories if c["category"] in parsed_result["categories"]]
    avg_weight = sum(weights) / len(weights) if weights else 0
    return round(conf * (1 - avg_weight), 2)

# ========== STEP 5: TOOL DEFINITIONS ==========
def retrieve_category_context(query):
    print("Retrieving category context for query:\n", query)
    retriever = build_vector_db()
    docs = retriever._get_relevant_documents(query, run_manager=None)
    return "\n".join([d.page_content for d in docs])

# Tool 1: CSV Reader
def read_email_tool(_):
    df = load_emails(EMAIL_FILE)
    return df.to_json(orient="records")

# Tool 2: Category Retriever
def category_tool(email_text):
    return retrieve_category_context(email_text)

# Tool 3: Compliance Analyzer
def compliance_tool(email_text):
    context = retrieve_category_context(email_text)
    print("Category Context Retrieved:\n", context)
    return json.dumps(check_compliance(email_text, context), indent=2)

tools = [
    Tool(
        name="EmailReader",
        func=read_email_tool,
        description="Reads and returns all emails as JSON records."
    ),
    Tool(
        name="CategoryRetriever",
        func=category_tool,
        description="Fetches relevant compliance category descriptions for a given email."
    ),
    Tool(
        name="ComplianceChecker",
        func=compliance_tool,
        description="Analyzes email and returns JSON with compliance result, categories, violations, and confidence."
    )
]

# ========== STEP 6: INITIALIZE LANGCHAIN AGENT ==========
agent = create_agent(llm, tools)

# ========== STEP 7: EXECUTION PIPELINE ==========
df = load_emails(EMAIL_FILE)
results = []

for _, row in df.iterrows():
    email_text = row["normalized_text"]
    print(f"\n--- Analyzing Email ID: {row['From']} ---")
    print("Email Content:\n", email_text)

    # Step 1: Agent fetches category context
    context = category_tool(email_text)

    # Step 2: Agent performs compliance check
    parsed = check_compliance(email_text, context)

    # Step 3: Compute programmatic score
    score = compute_score(parsed)

    results.append({
        "From": row["From"],
        "compliance": parsed.get("compliance", ""),
        "categories": parsed.get("categories", []),
        "violations": parsed.get("violations", []),
        "confidence": parsed.get("confidence", 0),
        "score": score
    })

# ========== STEP 8: SAVE OUTPUT ==========
out_df = pd.DataFrame(results)
out_df.to_csv(OUTPUT_FILE, index=False)
print("\n Compliance results saved to:", OUTPUT_FILE)
print(out_df)
