import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from agents.tools import (
    run_audit_tool,
    get_audit_summary_tool
)

load_dotenv()

SYSTEM_PROMPT = """
You are the DataSentinel Audit Agent.

Your job is to inspect the results of a tabular
training-data security audit.

Rules:
1. Use the supplied tools instead of inventing values.
2. Never claim a sample is definitely malicious.
3. Say "suspicious", "potentially poisoned", or
   "requires review".
4. Explain results in simple language.
5. Never create metrics that were not returned
   by DataSentinel.
6. If evidence is unavailable, say so explicitly.
"""


def create_audit_agent():
    api_key = os.getenv("GROQ_API_KEY") or "gsk_placeholder_key"
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0
    )
    return create_agent(
        model=llm,
        tools=[
            run_audit_tool,
            get_audit_summary_tool
        ],
        system_prompt=SYSTEM_PROMPT
    )


class LazyAuditAgent:
    def invoke(self, *args, **kwargs):
        agent = create_audit_agent()
        return agent.invoke(*args, **kwargs)


audit_agent = LazyAuditAgent()
