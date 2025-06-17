from typing import List, Dict, Any

class ChatAgent:
    """Minimal chat agent for Llama 3.2 on Ollama"""
    
    def __init__(self, model: Any):
        self.model = model  # Ollama model with .invoke()
        self.history: List[Dict[str, str]] = []  # Stores {role, content} pairs

    def generate_response(
        self,
        user_message: str,
        company: str,
        context: Dict[str, str],
    ) -> Dict[str, Any]:
        """
        Generates a response using Ollama's .invoke().
        Input/Output matches original:
        - Input: `user_message`, `company`, `context` (keys: finance/news)
        - Output: `{"text": "model_reply"}`
        """
        # Format system prompt (static instructions)
        system_prompt = """
        You are a finance expert analyzing Indian companies. Answer questions factually.
        Use provided data if available, otherwise use general knowledge.
        Do NOT give personalized advice.
        Be Concise.
        Company: {company}
        Financial Data: {finance}
        News: {news}
        """.format(
            company=company,
            finance=context.get("finance", "No data"),
            news=context.get("news", "No news"),
        )

         # Build messages for Ollama (system + history + new question)
        messages = [
            {"role": "system", "content": system_prompt},
            *self.history,  # Previous turns
            {"role": "user", "content": user_message},
        ]

        # Call Llama 3.2 via Ollama (using .invoke())
        response = self.model.invoke(messages)

        # Update history (Ollama-style dicts)
        self.history.extend([
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": response},
        ])

        return {"text": response}
