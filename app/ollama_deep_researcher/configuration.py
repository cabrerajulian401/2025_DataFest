import os
from enum import Enum
from typing import Any, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Explicitly load environment variables
load_dotenv()
print("Loaded environment variables in ollama_deep_researcher/configuration.py")
print(f"TAVILY_API_KEY present: {'TAVILY_API_KEY' in os.environ}")
print(f"FETCH_FULL_PAGE: {os.environ.get('FETCH_FULL_PAGE')}")
print(f"MAX_WEB_RESEARCH_LOOPS: {os.environ.get('MAX_WEB_RESEARCH_LOOPS')}")

from langchain_core.runnables import RunnableConfig

class SearchAPI(Enum):
    PERPLEXITY = "perplexity"
    TAVILY = "tavily"

class Configuration(BaseModel):
    """The configurable fields for the commercial real estate research assistant."""

    max_web_research_loops: int = Field(
        default=3,
        title="Research Depth",
        description="Number of research iterations to perform for CRE analysis"
    )
    local_llm: str = Field(
        default="llama3.2",
        title="LLM Model Name",
        description="Name of the LLM model to use for real estate analysis"
    )
    llm_provider: Literal["ollama"] = Field(
        default="ollama",
        title="LLM Provider",
        description="Provider for the LLM (only Ollama supported)"
    )
    search_api: Literal["perplexity", "tavily"] = Field(
        default="tavily",
        title="Search API",
        description="Web search API to use for real estate market research"
    )
    fetch_full_page: bool = Field(
        default=False,
        title="Fetch Full Page",
        description="Include the full page content in the search results for comprehensive analysis"
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434/",
        title="Ollama Base URL",
        description="Base URL for Ollama API"
    )
    strip_thinking_tokens: bool = Field(
        default=True,
        title="Strip Thinking Tokens",
        description="Whether to strip <think> tokens from model responses"
    )
    tavily_api_key: Optional[str] = Field(
        default=None,
        description="API key for Tavily search"
    )
    real_estate_focus: bool = Field(
        default=True,
        title="Real Estate Focus",
        description="Optimize searches and analysis for commercial real estate insights"
    )
    
    @classmethod
    def from_runnable_config(cls, config: RunnableConfig) -> "Configuration":
        """Create a Configuration from a RunnableConfig.
        
        Extracts configuration values from the RunnableConfig's configurable dict.
        
        Args:
            config: RunnableConfig object containing configuration values
            
        Returns:
            Configuration object with values from the RunnableConfig
        """
        configurable = {}
        
        if "configurable" in config:
            configurable = config["configurable"]
            
        # Load environment variables first
        env_fetch_full_page = os.getenv("FETCH_FULL_PAGE", "").lower() == "true"
        env_max_loops = os.getenv("MAX_WEB_RESEARCH_LOOPS")
        env_tavily_key = os.getenv("TAVILY_API_KEY")
        env_ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        env_local_llm = os.getenv("LOCAL_LLM")
        
        # Print debug information
        print(f"Creating Configuration with env vars: FETCH_FULL_PAGE={env_fetch_full_page}, MAX_LOOPS={env_max_loops}")
        
        # Start with defaults from the model
        result = cls()
        
        # Override with environment variables
        if env_fetch_full_page is not None:
            result.fetch_full_page = env_fetch_full_page
        if env_max_loops is not None:
            try:
                result.max_web_research_loops = int(env_max_loops)
            except ValueError:
                print(f"Warning: Invalid MAX_WEB_RESEARCH_LOOPS value: {env_max_loops}, using default")
        if env_tavily_key:
            result.tavily_api_key = env_tavily_key
        if env_ollama_base_url:
            # Strip any whitespace to prevent port number issues
            result.ollama_base_url = env_ollama_base_url.strip()
        if env_local_llm:
            result.local_llm = env_local_llm
            
        # Finally override with explicit config if provided
        if "local_llm" in configurable:
            result.local_llm = configurable["local_llm"]
        if "search_api" in configurable:
            result.search_api = configurable["search_api"]
        if "fetch_full_page" in configurable:
            result.fetch_full_page = configurable["fetch_full_page"]
        if "tavily_api_key" in configurable:
            result.tavily_api_key = configurable["tavily_api_key"]
        if "max_web_research_loops" in configurable:
            result.max_web_research_loops = configurable["max_web_research_loops"]
        if "ollama_base_url" in configurable:
            # Strip any whitespace to prevent port number issues
            result.ollama_base_url = configurable["ollama_base_url"].strip()
            
        print(f"Configuration created: fetch_full_page={result.fetch_full_page}, max_loops={result.max_web_research_loops}")
        return result