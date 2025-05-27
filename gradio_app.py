import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
from core.chat_agent import ChatAgent
from data.marketdata import MarketData
from data.stockdata import fetch_data
from utils.ollama import model
from tools.decision import get_tool_decision
from tools.search_online import search_ddg, extract_article_text
from rag.rag_pipeline import retrieve_context

class FinanceInterface:
    """Handles UI setup and user interactions"""
    
    def __init__(self):
        self.agent = ChatAgent(model)
        self.market = MarketData()
        # self.context = {
        #     'finance': '',
        #     'news': ''
        # }
        self.context_state = gr.State()


    def create_interface(self) -> gr.Blocks:
        """Build the complete Gradio interface"""
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            self._create_company_controls()
            self._create_data_display()
            self._create_chat_interface()
        return demo

    def _create_company_controls(self) -> None:
        """Company selection and date controls"""
        with gr.Row():
            self.company_input = gr.Dropdown(
                choices=self.market.company_list,
                label="Select Company",
                value=self.market.company_list[0]
            )
            
            with gr.Column():
                with gr.Row():
                    self.start_date = gr.Textbox(
                        label="Start Date", 
                        value=self._get_date(3),
                        placeholder="YYYY-MM-DD"
                    )
                    self.end_date = gr.Textbox(
                        label="End Date",
                        value=datetime.now().strftime("%Y-%m-%d"),
                        placeholder="YYYY-MM-DD"
                    )
                
                with gr.Row():
                    for months in [3, 6, 12]:
                        gr.Button(f"{months}M").click(
                            lambda: self._get_date_range(months),
                            inputs=[],
                            outputs=[self.start_date, self.end_date])
                    gr.Button("YTD").click(
                        self._get_ytd_range,
                        outputs=[self.start_date, self.end_date]
                    )

    def _create_data_display(self) -> None:
        """Data visualization components"""
        fetch_btn = gr.Button("Fetch Data")
        with gr.Row():
            self.metrics = gr.Dataframe(headers=["Metric", "Value"], interactive=False)
            self.news = gr.Textbox(label="Recent News", lines=10)
            self.plot = gr.Image(label="Price Chart", type="numpy")
        
        fetch_btn.click(
            fetch_data,
            inputs=[self.company_input, self.start_date, self.end_date],
            outputs=[self.metrics, self.plot, self.news, self.context_state]
        )

    def _create_chat_interface(self) -> None:
        """Chat conversation components"""
        gr.Markdown("---")
        self.chat = gr.ChatInterface(
            self._handle_chat,
            additional_inputs=[self.company_input, self.context_state],
            examples=[["What is volatility?"], ["Tell me about annual returns."]],
            title="Finance QnA Chat",
            description="Ask finance questions here. Your conversation will use the latest financial data as context.",
            additional_outputs=[self.context_state]
        )

    def _get_date(self, months_back: int) -> str:
        """Get formatted date string"""
        return (datetime.now() - timedelta(days=30*months_back)).strftime("%Y-%m-%d")

    def _get_date_range(self, months: int) -> Tuple[str, str]:
        """Update date range for buttons"""
        return (self._get_date(months), datetime.now().strftime("%Y-%m-%d"))

    def _get_ytd_range(self) -> Tuple[str, str]:
        """Get year-to-date range"""
        return (
            datetime(datetime.now().year, 1, 1).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )

    
    def _handle_chat(self, message: str, history: list, company: str, context: dict) -> str:
        """Complete chat processing matching original functionality"""
        # Initialize context if not provided
        if context is None:
            context = {
                'company': company,
                'finance': '',
                'news': ''
            }
        
        # Format history (same as original)
        messages = []
        for entry in history:
            if isinstance(entry, tuple) and len(entry) == 2:
                messages.append(("user", entry[0]))
                messages.append(("assistant", entry[1]))
            elif isinstance(entry, dict):
                messages.append((entry.get("role", "user"), entry.get("content", "")))
        messages.append(("user", message))

        # Get tool decision (same as original)
        decision = get_tool_decision(message, company)

        # Process financial retriever (added from original)
        if "financial_retriever" in decision.tools_needed:
            query = f"{company} {decision.modified_query}" if company else decision.modified_query
            retrieved_docs = retrieve_context(query)  # Now using self.retriever
            context['finance'] += "\n\nAdditional Documents:\n" + "\n".join([doc.page_content for doc in retrieved_docs])

        # Process web search (added from original)
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
                context['news'] += "\n\nWeb Results:\n" + "\n".join(web_results)

        # Generate response (using class's agent instead of global chain)
        response = self.agent.generate_response(
            user_message=message,
            company=company or "No company selected",
            context={
                'finance': context.get('finance', 'No financial data available'),
                'news': context.get('news', 'No recent news available.')
            },
            chat_history=messages
        )
        
        return response['text'], context

    def _handle_chat(self, message: str, history: list, company: str, context: dict) -> str:
        """Complete chat processing using memory-aware agent"""
        
        if context is None:
            context = {
                'company': company,
                'finance': '',
                'news': ''
            }

        decision = get_tool_decision(message, company)

        if "financial_retriever" in decision.tools_needed:
            query = f"{company} {decision.modified_query}" if company else decision.modified_query
            retrieved_docs = retrieve_context(query)
            context['finance'] += "\n\nAdditional Documents:\n" + "\n".join([doc.page_content for doc in retrieved_docs])

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
                context['news'] += "\n\nWeb Results:\n" + "\n".join(web_results)

        response = self.agent.generate_response(
            user_message=message,
            company=company or "No company selected",
            context={
                'finance': context.get('finance', 'No financial data available'),
                'news': context.get('news', 'No recent news available.')
            }
        )

        return response['text'], context

if __name__ == "__main__":
    interface = FinanceInterface()
    demo = interface.create_interface()
    demo.launch()