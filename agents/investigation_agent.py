import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from agents.tools import (
    get_audit_summary_tool,
    get_top_suspicious_samples_tool,
    get_sample_evidence_tool,
    get_detector_breakdown_tool
)

load_dotenv()

SYSTEM_PROMPT = """
You are the DataSentinel Investigation Agent.

Your purpose is to investigate suspicious training
data using actual detector evidence.

Use tools when the user asks:
- why a dataset is risky
- why a sample is suspicious
- which samples should be reviewed
- which detectors produced evidence

Rules:
- Never invent evidence.
- Never say that an anomalous record is definitely malicious.
- Distinguish anomaly from proven poisoning.
- Explain technical evidence in simple language.
"""


def create_investigation_agent():
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
            get_audit_summary_tool,
            get_top_suspicious_samples_tool,
            get_sample_evidence_tool,
            get_detector_breakdown_tool
        ],
        system_prompt=SYSTEM_PROMPT
    )


class LazyInvestigationAgent:
    def invoke(self, *args, **kwargs):
        agent = create_investigation_agent()
        return agent.invoke(*args, **kwargs)


investigation_agent = LazyInvestigationAgent()
