"""
DataSentinel - Investigation Agent (Milestone 8)
Investigates suspicious records and attributes findings strictly
to verified detector evidence without inventing certainty.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

from agents.tools import (
    get_audit_summary_tool,
    get_top_suspicious_samples_tool,
    get_sample_evidence_tool,
    get_detector_breakdown_tool,
    get_threat_limitations_tool,
    get_quarantine_manifest_tool
)

load_dotenv()

SYSTEM_PROMPT = """
You are the DataSentinel Investigation Agent.

Your purpose is to investigate suspicious records in machine learning training datasets
using recorded detector evidence across tabular and image pipelines.

Rules:
1. Always base findings on actual detector signals (e.g. label disagreement, prediction entropy,
   LOF embedding distance, activation clustering, spectral signatures, hash duplicates).
2. Distinguish statistical anomaly from confirmed poisoning: never claim a record is definitely
   malicious without external ground truth.
3. If investigating an individual sample, look up its actual contributing signals using tools.
4. When discussing clean-label manipulation or subtle blend threats, explicitly disclose that
   post-hoc detector recall was 0/12 under strict false positive ceilings and remains unresolved.
5. Explain technical evidence in clear, actionable terms suitable for data engineers and security analysts.
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
            get_detector_breakdown_tool,
            get_threat_limitations_tool,
            get_quarantine_manifest_tool
        ],
        system_prompt=SYSTEM_PROMPT
    )


class LazyInvestigationAgent:
    def invoke(self, *args, **kwargs):
        agent = create_investigation_agent()
        return agent.invoke(*args, **kwargs)


investigation_agent = LazyInvestigationAgent()
