"""Chat client using LangChain OpenAI - optimized for LangGraph."""

from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


class ChatClient:
    """LangChain-based client for vLLM chat services."""
    
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        load_dotenv(override=True)
        
        self.base_url = base_url 
        self.api_key = api_key 
        self.model = model
        
        if not self.base_url:
            raise ValueError("base_url or INFERENCE_API_URL must be set")
        
        # Initialize chat LLM
        self.llm = ChatOpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "not-needed",
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def invoke(self, messages):
        """Invoke the model with messages (LangGraph-compatible)."""
        return self.llm.invoke(messages)
    
    def stream(self, messages):
        """Stream response from the model."""
        return self.llm.stream(messages)
    
    def send_message(self, message: str) -> str:
        """Send a single message (backward compatibility)."""
        from langchain_core.messages import HumanMessage
        response = self.llm.invoke([HumanMessage(content=message)])
        return response.content
