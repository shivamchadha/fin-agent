# fin_agent.py
import os
import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import pdb

from marketdata import MarketData
from ollama import *
from stockdata import *
from rag_pipeline import rag_answer
from conversation import *
from decision import *
from search_online import search_ddg, extract_article_text


def get_date_range(months_back: int) -> tuple[str, str]:
    """Calculate date range for preset buttons"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=30*months_back)).strftime("%Y-%m-%d")
    return start_date, end_date


def chat_agent_conversation(message: str, history: list, company: str = None, current_context: dict = None):
    global retriever, model, chain
    
    # Initialize context with proper structure
    current_context = current_context or {
        'company': company,
        'finance': '',  # Changed from 'financial' to 'finance' to match generate_response
        'news': ''
    }
    
    # Validate and clean input
    messages = []
    for entry in history:
        if isinstance(entry, tuple) and len(entry) == 2:
            messages.append(("user", entry[0]))
            messages.append(("assistant", entry[1]))
        elif isinstance(entry, dict):
            messages.append((entry.get("role", "user"), entry.get("content", "")))
            
    messages.append(("user", message))

    # Get tool decision with company context
    decision = get_tool_decision(message, company)

    # Process financial retriever
    if "financial_retriever" in decision.tools_needed:
        query = f"{company} {decision.modified_query}" if company else decision.modified_query
        retrieved_docs = retriever.get_relevant_documents(query)
        current_context['finance'] += "\n\nAdditional Documents:\n" + "\n".join([doc.page_content for doc in retrieved_docs])

    # Process web search
    if "web_search" in decision.tools_needed:
        search_query = f"{company} " if company else ""
        search_query += decision.modified_query or message
        search_results = search_ddg(f"{search_query} financial updates")
        
        web_results = []
        for result in search_results:
            article = extract_article_text(result['url'])
            if article:
                web_results.append(
                    f"- {article['title']}\n  Source: {result['url']}\n  Excerpt: {article['text'][:300]}..."
                )
        
        if web_results:
            current_context['news'] += "\n\nWeb Results:\n" + "\n".join(web_results)

    # Generate response with properly structured context
    response = generate_response(
        user_message=message,
        company=company or "No company selected",
        context={
            'finance': current_context['finance'] or 'No financial data available',
            'news': current_context['news'] or 'No recent news available.'
        },
        chat_history=messages
    )
   
    
    return response['text']


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("## Finance Agent Chat")
    
    with gr.Row():
        company_input = gr.Dropdown(
            choices=market.company_list, 
            label="Select Company",
            value=market.company_list[0]
        )
        
        with gr.Column():
            with gr.Row():
                start_date_input = gr.Textbox(
                    label="Start Date", 
                    value=get_date_range(3)[0],  # Default to 3 months back
                    placeholder="YYYY-MM-DD"
                )
                end_date_input = gr.Textbox(
                    label="End Date",
                    value=datetime.now().strftime("%Y-%m-%d"),  # Default to today
                    placeholder="YYYY-MM-DD"
                )
            
            with gr.Row():
                gr.Button("3 Months").click(
                    lambda: get_date_range(3),
                    outputs=[start_date_input, end_date_input]
                )
                gr.Button("6 Months").click(
                    lambda: get_date_range(6),
                    outputs=[start_date_input, end_date_input]
                )
                gr.Button("12 Months").click(
                    lambda: get_date_range(12),
                    outputs=[start_date_input, end_date_input]
                )
                gr.Button("YTD").click(
                    lambda: (datetime(datetime.now().year, 1, 1).strftime("%Y-%m-%d"), 
                            datetime.now().strftime("%Y-%m-%d")),
                    outputs=[start_date_input, end_date_input]
                )

    fetch_button = gr.Button("Fetch Data")
    
    with gr.Row():
        # Left Column - Financial Metrics
        with gr.Column(scale=1):
            gr.Markdown("### Financial Summary")
            metrics_output = gr.Dataframe(
                headers=["Metric", "Value"],
                interactive=False,
            )
        
        # Right Column - News
        with gr.Column(scale=1):
            gr.Markdown("### Recent Company News")
            news_output = gr.Textbox(
                lines=10,
                max_lines=15,
                show_label=False,
                interactive=False,
                elem_classes="news-box"
            )
    
    
    plot_output = gr.Image(label="Stock Chart", type="numpy")
    financial_summary_state = gr.State()
    
    fetch_button.click(
        fn=fetch_data,
        inputs=[company_input, start_date_input, end_date_input],
        outputs=[metrics_output, plot_output,news_output, financial_summary_state]
    )
    
    gr.Markdown("---")
 

    chat_interface = gr.ChatInterface(
        fn=lambda msg, hist, co, ctx: chat_agent_conversation(
        message=msg,
        history=hist,
        company=co,
        current_context=ctx),
        additional_inputs=[company_input, financial_summary_state],
        examples=[["What is volatility?"], ["Tell me about annual returns."]],
        title="Finance QnA Chat",
        description="Ask finance questions here. Your conversation will use the latest financial data as context.")
    
demo.launch()
