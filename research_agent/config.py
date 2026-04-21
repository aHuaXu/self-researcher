"""Configuration for the Research Agent system."""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SearchConfig:
    """Search engine configuration."""
    engine: str = "google"  # "google" or "bing"
    top_k: int = 10
    region: str = "us"
    lang: str = "en"
    viewport_size: int = 1024 * 40
    serper_api_key: str = ""
    azure_subscription_key: str = ""
    azure_mkt: str = "zh-CN"


@dataclass
class LLMConfig:
    """LLM configuration for different agents."""
    # Executor uses DeepResearcher model (the one being RL trained)
    executor_model: str = "deep-researcher"
    executor_base_url: str = "http://localhost:8000/v1"
    executor_api_key: str = "token-xxx"

    # Planner and Writer use small 3B model
    small_model: str = "qwen2.5-3b"
    small_base_url: str = "http://localhost:8000/v1"
    small_api_key: str = "token-xxx"

    # Quick summary model (for webpage content summarization)
    summary_model: str = "qwen-plus"
    summary_base_url: str = "http://localhost:8000/v1"
    summary_api_key: str = "token-xxx"


@dataclass
class AgentConfig:
    """Main agent configuration."""
    search: SearchConfig = field(default_factory=SearchConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Cache paths
    query_save_path: str = "./research_agent/cache/search_result.json"
    notes_save_path: str = "./research_agent/cache/notes.json"


def load_config() -> AgentConfig:
    """Load configuration from environment variables or defaults."""
    config = AgentConfig()

    # Search config from env
    config.search.engine = os.getenv("SEARCH_ENGINE", "google")
    config.search.top_k = int(os.getenv("SEARCH_TOP_K", "10"))
    config.search.region = os.getenv("SEARCH_REGION", "us")
    config.search.lang = os.getenv("SEARCH_LANG", "en")
    config.search.serper_api_key = os.getenv("SERPER_API_KEY", "")
    config.search.azure_subscription_key = os.getenv("AZURE_BING_KEY", "")

    # LLM config from env
    config.llm.executor_model = os.getenv("EXECUTOR_MODEL", "deep-researcher")
    config.llm.executor_base_url = os.getenv("EXECUTOR_BASE_URL", "http://localhost:8000/v1")
    config.llm.executor_api_key = os.getenv("EXECUTOR_API_KEY", "token-xxx")

    config.llm.small_model = os.getenv("SMALL_MODEL", "qwen2.5-3b")
    config.llm.small_base_url = os.getenv("SMALL_BASE_URL", "http://localhost:8000/v1")
    config.llm.small_api_key = os.getenv("SMALL_API_KEY", "token-xxx")

    config.llm.summary_model = os.getenv("SUMMARY_MODEL", "qwen-plus")
    config.llm.summary_base_url = os.getenv("SUMMARY_BASE_URL", "http://localhost:8000/v1")
    config.llm.summary_api_key = os.getenv("SUMMARY_API_KEY", "token-xxx")

    # Cache paths
    config.query_save_path = os.getenv("QUERY_SAVE_PATH", "./research_agent/cache/search_result.json")
    config.notes_save_path = os.getenv("NOTES_SAVE_PATH", "./research_agent/cache/notes.json")

    return config


# Global config instance
_config: Optional[AgentConfig] = None


def get_config() -> AgentConfig:
    """Get the global config instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def set_config(config: AgentConfig):
    """Set the global config instance."""
    global _config
    _config = config