"""
Graph export for LangGraph Studio.

This file exports the earnings agent graph for use with LangGraph Studio.
LangGraph CLI will discover this file and use it to run the development server.
"""
from app.agents.earnings_agent import create_earnings_agent

# Export the graph - LangGraph Studio will use this
graph = create_earnings_agent()


