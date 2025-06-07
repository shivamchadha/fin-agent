import gradio as gr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple
from typing import List, Dict, Any
from core.chat_agent import ChatAgent
from data.marketdata import MarketData
from data.stockdata import fetch_data
from utils.ollama import model
from tools.decision import get_tool_decision
from tools.search_online import search_ddg, extract_article_text
from rag.rag_pipeline import retrieve_context
import pdb  # For debugging purposes, can be removed later

class FinanceInterface:
    """Handles UI setup and user interactions, using gr.ChatInterface."""

    def __init__(self):
        self.agent = ChatAgent(model)
        self.market = MarketData()
        self.context = {"finance": "", "news": ""}

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            self._create_company_controls()
            self._create_data_display()
            self._create_chat_interface()
        return demo



    def _create_company_controls(self) -> None:
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
                            fn=lambda m=months: self._get_date_range(m),
                            inputs=[],
                            outputs=[self.start_date, self.end_date]
                        )
                    gr.Button("YTD").click(
                        fn=self._get_ytd_range,
                        inputs=[],
                        outputs=[self.start_date, self.end_date]
                    )

    def _create_data_display(self) -> None:
        fetch_btn = gr.Button("Fetch Data")
        with gr.Row():
            self.metrics = gr.Dataframe(
                headers=["Metric", "Value"],
                interactive=False,
                label="Financial Metrics"
            )
            self.news = gr.Textbox(
                label="Recent News",
                lines=10,
                interactive=False
            )
            self.plot = gr.Image(
                label="Price Chart",
                type="numpy"
            )
        # fetch_btn updates UI, and we store `context` internally
        fetch_btn.click(
            fn=self._wrapped_fetch,
            inputs=[self.company_input, self.start_date, self.end_date],
            outputs=[self.metrics, self.plot, self.news]
        )

    def _create_chat_interface(self) -> None:
        gr.Markdown("---")
        self.chat_interface = gr.ChatInterface(
            fn=self._handle_chat,
            additional_inputs=[self.company_input],
            examples=[["What is volatility?"], ["Tell me about annual returns."]],
            title="Finance QnA Chat",
            description="Ask finance questions here. Your conversation will use the latest financial data as context."
        )

    def _get_date(self, months_back: int) -> str:
        return (datetime.now() - timedelta(days=30 * months_back)).strftime("%Y-%m-%d")

    def _get_date_range(self, months: int) -> Tuple[str, str]:
        return (
            self._get_date(months),
            datetime.now().strftime("%Y-%m-%d")
        )

    def _get_ytd_range(self) -> Tuple[str, str]:
        return (
            datetime(datetime.now().year, 1, 1).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d")
        )

    def _wrapped_fetch(self, company: str, start_date: str, end_date: str):
        metrics, plot, news, context = fetch_data(company, start_date, end_date)
        self.context.update(context)
        return metrics, plot, news

    def _handle_chat(
        self,
        message: str,
        history: List[Dict[str, Any]],
        company: str
    ) -> str:
        if not self.context:
            self.context = {"finance": "", "news": ""}

        # Extract chat history
        messages = [(entry["role"], entry["content"]) for entry in history if isinstance(entry, dict)]
        messages.append(("user", message))

        decision = get_tool_decision(message, company)

        if "financial_retriever" in decision.tools_needed:
            print("Retrieving financial documents...")
            query = f"{company} {decision.modified_query}" if company else decision.modified_query
            retrieved_docs = retrieve_context(query)
            snippet = "\n".join(doc.page_content for doc in retrieved_docs)
            self.context["finance"] += "\n\nAdditional Documents:\n" + snippet
            print("Retrieved financial documents:", snippet)

        if "web_search" in decision.tools_needed:
            print("Performing web search for news...")
            search_query = f"{company} " if company else ""
            search_query += (decision.modified_query or message)
            search_results = search_ddg(f"{search_query} financial updates")

            web_results = []
            for result in search_results:
                # pdb.set_trace()  # Debugging line to inspect search results
                article = extract_article_text(result)
                if article:
                    excerpt = article["text"][:300].replace("\n", " ")
                    web_results.append(
                        f"- {article['title']}\n  Source: {result['url']}\n  Excerpt: {excerpt}..."
                    )
            if web_results:
                self.context["news"] += "\n\nWeb Results:\n" + "\n".join(web_results)
            print("Web search results:", web_results)

        agent_out = self.agent.generate_response(
            user_message=message,
            company=company or "No company selected",
            context={
                "finance": self.context.get("finance", "No financial data available"),
                "news": self.context.get("news", "No recent news available.")
            }
        )
        return agent_out["text"]


if __name__ == "__main__":
    interface = FinanceInterface()
    demo = interface.create_interface()

    # demo.launch()
    demo.launch()