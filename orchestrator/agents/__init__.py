# ============================================================
#  KriyaOS — orchestrator/agents/__init__.py
#  Exports all agents for easy import anywhere in KriyaOS.
#
#  Usage:
#    from orchestrator.agents import backend_agent, planner
#    backend_agent.write("Write a sort function")
#    planner.run("Plan a full REST API project")
# ============================================================

from orchestrator.agents import (
    planner,
    backend_agent,
    reviewer,
    docs_agent,
    design_agent,
    testing_agent,
)

# Map category names to agent modules
# Used by the orchestrator to dispatch tasks
AGENT_MAP = {
    "code":    backend_agent,
    "fix":     backend_agent,
    "explain": backend_agent,
    "review":  reviewer,
    "docs":    docs_agent,
    "design":  design_agent,
    "plan":    planner,
    "build":   planner,
    "test":    testing_agent,
}


def get_agent(category: str):
    """
    Get the agent module for a task category.

    Usage:
        agent = get_agent("code")
        agent.write("Write a FastAPI server")
    """
    return AGENT_MAP.get(category, backend_agent)