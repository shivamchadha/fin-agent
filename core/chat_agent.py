from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain.memory import ConversationBufferMemory
from typing import List, Dict, Any

class ChatAgent:
    """Handles conversation logic and response generation"""

    def __init__(self, model: Any):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.prompt = PromptTemplate(
            input_variables=["company", "financial_summary", "news_summary", "user_question", "chat_history"],
            template="""
                You are a knowledgeable finance agent specializing in analyzing data for Indian companies.
                Your role is to answer objective finance questions with factual analysis. When provided, use the summaries of key financial metrics and recent news to inform your answer. However, if no summary or context is available, please answer the question using your general financial knowledge.
                You do NOT provide personalized financial advice.
                If you are not certain about any information, do not provide an answer.

                Company Context: 
                {company}

                Data Summary:
                {financial_summary}

                News Summary:
                {news_summary}

                Conversation History:
                {chat_history}

                Based on the above, please answer the following question concisely:
                {user_question}

                Provide a detailed response using only relevant information from the context. 
                If using numbers or facts, cite their source.
            """
        )

        # Use RunnableSequence piping
        self.chain = self.prompt | model

    def format_history(self, chat_history: List) -> str:
        """Convert various chat history formats to string"""
        formatted = []
        for entry in chat_history:
            if isinstance(entry, tuple):
                user, bot = entry
                formatted.extend([f"User: {user}", f"Assistant: {bot}"])
            elif isinstance(entry, dict):
                formatted.append(f"{entry.get('role', 'user').capitalize()}: {entry.get('content', '')}")
        return "\n".join(formatted) or "No chat history"

    def generate_response(
        self,
        user_message: str,
        company: str,
        context: Dict[str, str]
    ) -> Dict:
        """Generate response with memory handling"""

        # Load history into prompt
        memory_vars = self.memory.load_memory_variables({})
        chat_history = memory_vars.get("chat_history", "")

        inputs = {
            "company": company,
            "financial_summary": context.get("finance", "No financial data"),
            "news_summary": context.get("news", "No recent news"),
            "user_question": user_message,
            "chat_history": chat_history
        }

        response = self.chain.invoke(inputs)

        # Update memory
        self.memory.save_context(
            {"input": user_message},
            {"output": response}
        )

        return {"text": response}
