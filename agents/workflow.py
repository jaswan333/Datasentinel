from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from agents.audit_agent import audit_agent
from agents.investigation_agent import investigation_agent
from agents.reporting_agent import reporting_agent


class AgentState(TypedDict, total=False):
    dataset_id: str
    question: str
    audit_summary: str
    investigation: str
    final_report: str


def extract_final_message(result) -> str:
    messages = result.get(
        "messages",
        []
    )

    if not messages:
        return "No response generated."

    last_msg = messages[-1]
    content = getattr(last_msg, "content", None)
    if content is None and isinstance(last_msg, dict):
        content = last_msg.get("content", "")

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts) if parts else str(content)

    return str(content or "")


def audit_node(
    state: AgentState
):
    prompt = f"""
Audit dataset ID:

{state['dataset_id']}

Use the DataSentinel tools to retrieve
the audit summary.

Explain the current audit status.
"""

    result = audit_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    return {
        "audit_summary":
            extract_final_message(result)
    }


def investigation_node(
    state: AgentState
):
    prompt = f"""
Dataset ID:
{state['dataset_id']}

User question:
{state.get('question', 'Explain the audit.')}

Investigate this dataset using the DataSentinel
tools.

Focus on actual evidence and suspicious samples.
"""

    result = investigation_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    return {
        "investigation":
            extract_final_message(result)
    }


def reporting_node(
    state: AgentState
):
    prompt = f"""
Create the final DataSentinel explanation.

Audit Agent:
{state.get('audit_summary', '')}

Investigation Agent:
{state.get('investigation', '')}

Original User Question:
{state.get('question', '')}

Produce a clear non-technical explanation.
"""

    result = reporting_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    })

    return {
        "final_report":
            extract_final_message(result)
    }


workflow = StateGraph(
    AgentState
)

workflow.add_node(
    "audit_agent",
    audit_node
)

workflow.add_node(
    "investigation_agent",
    investigation_node
)

workflow.add_node(
    "reporting_agent",
    reporting_node
)

workflow.add_edge(
    START,
    "audit_agent"
)

workflow.add_edge(
    "audit_agent",
    "investigation_agent"
)

workflow.add_edge(
    "investigation_agent",
    "reporting_agent"
)

workflow.add_edge(
    "reporting_agent",
    END
)

agent_workflow = workflow.compile()
