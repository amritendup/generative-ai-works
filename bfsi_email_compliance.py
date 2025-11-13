# ------------------------------------------------------
# BFSI Email Compliance Agent (Modern LangChain 0.2+)
# GPT-4o + RAG + Chroma + Programmatic Scoring
# ------------------------------------------------------

import os
import json
import pandas as pd
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_openai_functions_agent, tool
from langchain.tools import Tool
from langchain_core.messages import HumanMessage

# ==================== CONFIG ====================
#os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_endpoint = os.getenv("OPENAI_API_ENDPOINT")

print(f"OpenAI API Key: {openai_api_key}")
print(f"OpenAI API Endpoint: {openai_api_endpoint}")


EMAIL_FILE = "emails.csv"
OUTPUT_FILE = "compliance_results.csv"

# ==================== CATEGORY SETUP ====================
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

# ==================== VECTOR DB CREATION ====================
def build_vector_retriever():
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


retriever = build_vector_retriever()

# ==================== LLM ====================
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

# ==================== PROMPT ====================
compliance_prompt = ChatPromptTemplate.from_template("""
You are a BFSI compliance auditor.

Email Content:
{email_text}

Category Context (Descriptions and Definitions):
{context}

Task:
1. Determine if the email is Compliant or Non-Compliant.
2. List applicable compliance categories (can be multiple).
3. Identify violating or suspicious lines if any.
4. Provide a confidence level (0 to 1) for your judgment.

Return JSON only in the following format:
{{
  "compliance": "<Compliant or Non-Compliant>",
  "categories": ["<category1>", "<category2>"],
  "violations": ["<suspicious line>"],
  "confidence": <float between 0 and 1>
}}
""")

# ==================== TOOLS ====================

@tool
def read_emails_tool(_: str) -> str:
    """Reads and returns all emails from local CSV as JSON records."""
    df = pd.read_csv(EMAIL_FILE)
    df.fillna("", inplace=True)
    df["normalized_text"] = df["subject"].str.strip() + "\n" + df["body"].str.strip()
    return df.to_json(orient="records")

@tool
def category_retriever_tool(email_text: str) -> str:
    """Retrieves relevant compliance category descriptions for a given email."""
    docs = retriever.get_relevant_documents(email_text)
    return "\n".join([d.page_content for d in docs])

@tool
def compliance_checker_tool(email_text: str) -> str:
    """Analyzes email for compliance violations and returns JSON result."""
    context = category_retriever_tool(email_text)
    full_prompt = compliance_prompt.format(email_text=email_text, context=context)
    response = llm.invoke([HumanMessage(content=full_prompt)])
    try:
        parsed = json.loads(response.content)
    except Exception:
        parsed = {"compliance": "Error", "categories": [], "violations": [], "confidence": 0.0}
    return json.dumps(parsed, indent=2)

# ==================== PROGRAMMATIC SCORING ====================
def compute_score(parsed_result):
    conf = parsed_result.get("confidence", 0)
    cats = parsed_result.get("categories", [])
    if not cats:
        return round(conf, 2)
    weights = [c["weightage"] for c in categories if c["category"] in cats]
    avg_weight = sum(weights) / len(weights) if weights else 0
    return round(conf * (1 - avg_weight), 2)

# ==================== AGENT CREATION ====================
tools = [read_emails_tool, category_retriever_tool, compliance_checker_tool]
agent = create_openai_functions_agent(llm, tools)

# ==================== EXECUTION ====================
def run_compliance_pipeline():
    df = pd.read_csv(EMAIL_FILE)
    df.fillna("", inplace=True)
    df["normalized_text"] = df["subject"].str.strip() + "\n" + df["body"].str.strip()

    results = []
    for _, row in df.iterrows():
        email_text = row["normalized_text"]
        print(f"\n--- Analyzing Email ID: {row['email_id']} ---")

        # Agent uses compliance tool directly (you could call agent.invoke for autonomy)
        raw = compliance_checker_tool.invoke(email_text)
        parsed = json.loads(raw)
        score = compute_score(parsed)

        results.append({
            "email_id": row["email_id"],
            "compliance": parsed.get("compliance", ""),
            "categories": parsed.get("categories", []),
            "violations": parsed.get("violations", []),
            "confidence": parsed.get("confidence", 0),
            "score": score
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ Compliance results saved to: {OUTPUT_FILE}")
    print(out_df)


if __name__ == "__main__":
    run_compliance_pipeline()
