"""Configuration for the Research Agent system."""

import os
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Optional

# 自动加载根目录的 .env 文件
# config.py 在 research_agent/config.py，所以根目录是 ../../.env
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))


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
    """LLM configuration."""
    model: str = "qwen2.5-7b"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "token-xxx"


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

    # LLM config: single model for all agents
    config.llm.model = os.getenv("LLM_MODEL", "qwen2.5-7b")
    config.llm.base_url = os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")
    config.llm.api_key = os.getenv("LLM_API_KEY", "token-xxx")

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