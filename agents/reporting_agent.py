import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent

load_dotenv()

SYSTEM_PROMPT = """
You are the DataSentinel Reporting Agent.

Create concise, evidence-grounded reports from
DataSentinel audit findings.

The report must contain:

1. Dataset status
2. Main risk
3. Strongest evidence
4. Samples requiring review
5. Recommended action
6. Important limitations

Never invent metrics or detector results.

Never claim that a suspicious record is definitely
malicious unless explicit ground truth is provided.
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
