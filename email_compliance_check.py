# ------------------------------------------------------
# BFSI Email Compliance Agent (Updated for LangChain v0.2+)
# ------------------------------------------------------

import os
import json
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.tools import tool  # Tool decorator
from langchain.agents import AgentExecutor  # Or proper agent class for your version
from langchain.tools import BaseTool  # if custom tools needed

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document

# ===== CONFIG =====
#os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_endpoint = os.getenv("OPENAI_API_ENDPOINT")
EMAIL_FILE = "emails.csv"
OUTPUT_FILE = "compliance_results.csv"

# ===== Category metadata =====
categories = [
    {"category": "Secrecy", "priority": "High", "weightage": 0.9,
     "desc": "Disclosure of confidential client or trading information."},
    {"category": "Market manipulation/ Misconduct", "priority": "Critical", "weightage": 1.0,
     "desc": "Attempt to influence market unfairly or insider trading."},
    {"category": "Market bribery", "priority": "High", "weightage": 0.85,
     "desc": "Offering incentives for market influence."},
    {"category": "Change in communication", "priority": "Medium", "weightage": 0.6,
     "desc": "Using unauthorized or informal communication channels."},
    {"category": "Complaints", "priority": "Medium", "weightage": 0.5,
     "desc": "Customer or employee grievances."},
    {"category": "Employee ethics", "priority": "Medium", "weightage": 0.7,
     "desc": "Integrity, honesty, and professional misconduct."}
]

# ===== Build vector retriever =====
def build_retriever():
    docs = [
        Document(page_content=(
            f"Category: {c['category']}\n"
            f"Priority: {c['priority']}\n"
            f"Weightage: {c['weightage']}\n"
            f"Description: {c['desc']}"
        ))
        for c in categories
    ]
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma.from_documents(docs, embeddings, collection_name="category_db")
    return vectordb.as_retriever(search_kwargs={"k": 3})

retriever = build_retriever()

# ===== LLM =====
llm = ChatOpenAI(model_name="gpt-4o", temperature=0.0)

# ===== Prompt Template =====
prompt_template = ChatPromptTemplate.from_template("""
You are a BFSI compliance auditor.

Email Content:
{email_text}

Category Context:
{context}

Task:
1. Determine if the email is Compliant or Non-Compliant.
2. List one or more applicable compliance categories.
3. Identify violating or suspicious lines if any.
4. Provide a confidence level (0 to 1) for your judgment.

Return JSON only in this format:
{
  "compliance": "<Compliant or Non-Compliant>",
  "categories": ["<category1>", "<category2>"],
  "violations": ["<suspicious line>"],
  "confidence": <float>
}
""")

# ===== Tools =====
@tool
def read_emails(_: str) -> str:
    """Read local CSV of emails and return as JSON string."""
    df = pd.read_csv(EMAIL_FILE)
    df.fillna("", inplace=True)
    df["normalized_text"] = df["subject"].str.strip() + "\n" + df["body"].str.strip()
    return df.to_json(orient="records")

@tool
def retrieve_context(email_text: str) -> str:
    """Retrieve category context from vector DB for given email text."""
    docs = retriever.get_relevant_documents(email_text)
    return "\n".join([d.page_content for d in docs])

@tool
def analyze_compliance(email_text: str) -> str:
    """Call LLM and return JSON string result."""
    context = retrieve_context(email_text)
    prompt = prompt_template.format(email_text=email_text, context=context)
    response = llm.invoke([{"role": "user", "content": prompt}])
    try:
        parsed = json.loads(response.content)
    except Exception:
        parsed = {"compliance": "Error", "categories": [], "violations": [], "confidence": 0.0}
    return json.dumps(parsed)

# ===== Scoring =====
def compute_score(parsed_result: dict) -> float:
    conf = parsed_result.get("confidence", 0.0)
    cats = parsed_result.get("categories", [])
    if not cats:
        return round(conf, 2)
    weights = [c["weightage"] for c in categories if c["category"] in cats]
    avg_w = sum(weights)/len(weights) if weights else 0
    return round(conf * (1 - avg_w), 2)

# ===== Agent Setup & Execution =====
tools = [read_emails, retrieve_context, analyze_compliance]

agent_executor = AgentExecutor.from_tools_and_llm(
    tools=tools,
    llm=llm,
    verbose=True
)

def run_pipeline():
    emails = json.loads(read_emails.invoke(""))
    results = []
    for rec in emails:
        email_id = rec.get("email_id")
        text = rec.get("normalized_text")
        print(f"Processing email_id={email_id}")
        raw = analyze_compliance.invoke(text)
        parsed = json.loads(raw)
        score = compute_score(parsed)
        results.append({
            "email_id": email_id,
            "compliance": parsed.get("compliance"),
            "categories": parsed.get("categories"),
            "violations": parsed.get("violations"),
            "confidence": parsed.get("confidence"),
            "score": score
        })
    df_out = pd.DataFrame(results)
    df_out.to_csv(OUTPUT_FILE, index=False)
    print("Results saved:", OUTPUT_FILE)
    print(df_out)

if __name__ == "__main__":
    run_pipeline()
