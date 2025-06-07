from langchain.llms.base import LLM
from pydantic import Field, ValidationError
from typing import Optional, List
import json
import requests
import logging

logger = logging.getLogger(__name__)

class OllamaLLM(LLM):
    """Custom LLM wrapper for Ollama models."""
    
    model_name: str = Field(default="llama3.2", description="Ollama model name")
    base_url: str = Field(default="http://localhost:11434/api/generate", 
                         description="API endpoint URL")
    temperature: float = Field(default=0.2, ge=0, le=1, 
                              description="Model temperature")
    session: requests.Session = Field(default_factory=requests.Session, 
                                     exclude=True)

    @property
    def _llm_type(self) -> str:
        return "ollama"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Execute the LLM query with proper error handling and streaming."""
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "temperature": self.temperature,
            "stream": False,
            "options": {"stop": stop} if stop else None
        }

        try:
            response = self.session.post(
                self.base_url,
                json={k: v for k, v in payload.items() if v is not None},
                headers={"Content-Type": "application/json"},
                timeout=120
            )
            response.raise_for_status()

            # Handle both JSON and streaming responses
            if "application/json" in response.headers.get("Content-Type", ""):
                return response.json().get("response", "")

            return "".join(
                json.loads(line).get("response", "")
                for line in response.text.splitlines()
                if line.strip()
            )

        except requests.HTTPError as e:
            logger.error(f"Ollama API HTTP Error: {e.response.text}")
            return ""
        except (requests.RequestException, json.JSONDecodeError) as e:
            logger.error(f"Ollama API error: {str(e)}")
            return ""
        except ValidationError as ve:
            logger.error(f"Validation error: {ve.errors()}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return ""

    @property
    def _identifying_params(self) -> dict:
        """Get identifying parameters for caching."""
        return {"model_name": self.model_name, "base_url": self.base_url}

# Corrected initialization and usage
model = OllamaLLM(
    model_name="llama3.2",
    base_url="http://localhost:11434/api/generate"  # Fixed endpoint
)
if __name__ == '__main__':
    
    response = model.invoke("What is the capital of France?")
    print(response)