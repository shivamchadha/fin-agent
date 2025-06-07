from typing import List
from pydantic import BaseModel
import json
from utils.ollama import model  # Your raw model instance from commons.py
from functools import lru_cache


class ToolDecision(BaseModel):
    needs_retrieval: bool
    tools_needed: List[str]
    modified_query: str = None
    reasoning: str


@lru_cache(maxsize=256)  # Cache up to 256 unique queries
def cached_tool_decision(query: str) -> dict:
    """
    Cached version of select_tools with automatic query normalization
    """
    # Normalize query for better cache hits
    normalized_query = query.strip().lower()
    return select_tools(normalized_query)


def get_tool_decision(query: str, use_cache: bool = True) -> dict:
    """
    Public interface for getting tool decisions with cache control
    """
    normalized_query = query.strip().lower()
    if use_cache:
        return cached_tool_decision(normalized_query)
    return select_tools(normalized_query)


def select_tools(query: str) -> ToolDecision:
    """
    Direct model invocation for tool selection with enhanced prompt
    """
    prompt = f"""Analyze this query and select appropriate tools, only use the tool if you believe you cannot answer the question directly.
    Return optimized query for retrieval if needed.
    Available Tools:
    1. financial_retriever - Internal financial documents (fundamentals, historical data)
    2. web_search - Company news, recent events, or unknown information
    
    Respond ONLY with JSON:
    {{
        "needs_retrieval": bool,
        "tools_needed": ["financial_retriever", "web_search"],
        "rag_query": "optimized query or original",
        "search_query": "query for web search or original",
        "reasoning": "step-by-step logic"
    }}
    
    Query: {query}"""
    
    try:
        response = model.invoke(prompt)
        decision = json.loads(response.strip())
        print("Tool selection decision:", decision)  # Debug output
        return ToolDecision(**decision)
    except Exception as e:
        return ToolDecision(
            needs_retrieval=False,
            tools_needed=[],
            modified_query=query,
            reasoning=f"Error in Tool selection: {str(e)}"
        )


def decide_retrieval(query: str) -> bool:
    """
    Lightweight version for retrieval-only decisions
    Returns True/False for whether retrieval should occur
    """
    prompt = f"""Should we retrieve external context for this query? 
    Answer ONLY with JSON: {{"retrieve": bool, "reason": "..."}}
    
    Query: {query}"""
    
    try:
        response = model.invoke(prompt)
        return json.loads(response.strip())["retrieve"]
    except:
        return False  # Default fallback
    

if __name__ == '__main__':
    
    test_queries = [
        ("What's Apple's revenue?", True),
        ("Hi how are you?", False),
        ("Explain Q2 earnings", True)
    ]

    for query, expected in test_queries:
        decision = decide_retrieval(query)
        assert decision.needs_retrieval == expected