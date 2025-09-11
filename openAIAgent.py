from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
import os


class OpenAIAgent:
    
    def __init__(self, api_key: str, model_name: str = "openai:gpt-5"):
        self.api_key = api_key
        self.model_name = model_name        
        self.agent = self._initialize_agent()

    def _initialize_agent(self):
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.memory_saver = MemorySaver()
        self.model = init_chat_model(self.model_name)
        agent = create_react_agent(llm=self.model, memory_saver=self.memory_saver)
        return agent


    def run(self, input_text: str):
        return self.agent.run(input_text)