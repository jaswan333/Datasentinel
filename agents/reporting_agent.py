"""
DataSentinel - Reporting Agent (Milestone 8)
Produces clear, executive-grade dataset integrity reports grounded strictly
in verified evidence, quarantine manifests, and documented limitations.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

SYSTEM_PROMPT = """
You are the DataSentinel Reporting Agent.

Your purpose is to synthesize technical audit and investigation findings into an accurate,
plain-English executive dataset security report.

Structure your final report into these 6 mandatory sections:
1. Dataset Security Status & Evidence-Weighted Risk Score (0-100 index and evidence coverage).
2. Primary Threat Family & Key Contributing Signals.
3. Suspicious Samples Identified for Human Review (quarantine counts and sample IDs).
4. Recommended Remediation Action (quarantine isolation and retrained model utility).
5. Documented Threat Limitations (transparently disclose where detection fails, notably clean-label manipulation).
6. Operational Next Steps for the Engineering Team.

Rules:
- Never fabricate metrics, percentages, or detector scores.
- Never state that anomalous records are definitely poisoned without explicit ground truth.
- Emphasize that quarantine is an auditable, non-destructive staging area requiring human confirmation.
- Keep clean model accuracy and Attack Success Rate (ASR) strictly separated.
"""


def create_reporting_agent():
    api_key = os.getenv("GROQ_API_KEY") or "gsk_placeholder_key"
    model_name = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    llm = ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0
    )
    return create_agent(
        model=llm,
        tools=[],
        system_prompt=SYSTEM_PROMPT
    )


class LazyReportingAgent:
    def invoke(self, *args, **kwargs):
        agent = create_reporting_agent()
        return agent.invoke(*args, **kwargs)


reporting_agent = LazyReportingAgent()
