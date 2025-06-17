from typing import List
from pydantic import BaseModel
import json
from utils.ollama import model  # Your raw model instance from commons.py
from functools import lru_cache


class ToolDecision(BaseModel):
    needs_retrieval: bool
    tools_needed: List[str]
    financial_query: str = None
    search_query: str


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
    Enhanced tool selection with separate query optimization for each tool type.
    Returns distinct queries for RAG (financial data) and web search (news).
    """
    prompt = f"""Analyze this query and prepare optimized searches for:
    - financial_retriever: Optimize for document retrieval (fundamentals, financials)
    - web_search: Optimize for news/events search
    Only use the tools if aboslutely necessary.
    Respond EXACTLY with this JSON (no other text):
    {{
        "needs_retrieval": boolean,
        "tools_needed": ["financial_retriever", "web_search"],
        "financial_query": "optimized terms for financial docs",
        "search_query": "optimized terms for online search"
    }}
    Query: {query}"""
    
    try:
        response = model.invoke(prompt)
        cleaned = response.strip().replace('```json', '').replace('```', '').strip()
        decision = json.loads(cleaned)
        
        # Validate and normalize
        tools = decision.get("tools_needed", [])
        valid_tools = ["financial_retriever", "web_search"]
        
        # Default to original query if specific optimizations aren't provided
        financial_query = str(decision.get("financial_query", query))
        search_query = str(decision.get("search_query", query))
        
        return ToolDecision(
            needs_retrieval=bool(decision.get("needs_retrieval", bool(tools))),
            tools_needed=[t for t in tools if t in valid_tools],
            financial_query=financial_query, 
            search_query=search_query
        )
        
    except json.JSONDecodeError:
        return ToolDecision(
            needs_retrieval=False,
            tools_needed=[],
            financial_query=query,
            search_query=query
        )
    except Exception as e:
        return ToolDecision(
            needs_retrieval=False,
            tools_needed=[],
            financial_query=query,
            search_query=query
        )


if __name__ == '__main__':
    
    test_queries = [
        ("What's Apple's revenue?", True),
        ("Hi how are you?", False),
        ("Explain Q2 earnings", True)
    ]

    for query, expected in test_queries:
        decision = get_tool_decision(query)
        assert decision.needs_retrieval == expected