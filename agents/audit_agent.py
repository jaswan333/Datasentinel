"""
DataSentinel - Audit Agent (Milestone 8)
Validates dataset audit integrity, explains evidence coverage,
and verifies non-destructive quarantine policies.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from agents.tools import (
    run_audit_tool,
    get_audit_summary_tool,
    get_quarantine_manifest_tool,
    get_threat_limitations_tool
)

load_dotenv()

SYSTEM_PROMPT = """
You are the DataSentinel Audit Agent.

Your responsibility is to inspect and validate the integrity of dataset security audits
across tabular and image datasets.

Rules:
1. Use the provided tools to retrieve real audit facts. Never invent scores or metrics.
2. The risk score is an evidence-weighted index (0 to 100), NOT a probability of poisoning.
3. Check evidence coverage: if coverage is low or signals are unavailable, state this clearly.
4. Quarantine is strictly non-destructive: emphasize that flagged records are isolated for human review.
5. Use cautious terminology: say "suspicious", "potentially compromised", or "requires review".
6. Explicitly disclose unresolved threat limitations (e.g. clean-label manipulation) when present.
7. Explain audit status in clear, concise language.
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
            get_audit_summary_tool,
            get_quarantine_manifest_tool,
            get_threat_limitations_tool
        ],
        system_prompt=SYSTEM_PROMPT
    )


class LazyAuditAgent:
    def invoke(self, *args, **kwargs):
        agent = create_audit_agent()
        return agent.invoke(*args, **kwargs)


audit_agent = LazyAuditAgent()
