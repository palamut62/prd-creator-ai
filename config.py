"""
Uygulama konfigürasyon ayarları
"""
import os
from pathlib import Path

# API Ayarları
DEFAULT_MODEL = "openai/gpt-5"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

# Rate Limiting
DEFAULT_MAX_REQUESTS = 5
DEFAULT_RATE_WINDOW = 300  # 5 dakika

# Timeout Ayarları (saniye)
DEFAULT_API_TIMEOUT = 60
DEFAULT_CONNECT_TIMEOUT = 30
DEFAULT_TOTAL_TIMEOUT = 300  # 5 dakika

# Input Validation
MIN_INPUT_LENGTH = 10
MAX_INPUT_LENGTH = 5000
MAX_SANITIZED_LENGTH = 10000

# File Paths
PROJECT_ROOT = Path(__file__).parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"

# UI Ayarları
PAGE_TITLE = "PRD Creator - AI Ürün Doküman Üretici"
PAGE_ICON = "🚀"

# Desteklenen modeller - Güncel OpenRouter Listesi
SUPPORTED_MODELS = {
    "free": [
        "openai/gpt-oss-20b:free",
        "z-ai/glm-4.5-air:free", 
        "qwen/qwen3-coder:free",
    ],
    "economic": [
        "openai/gpt-3.5-turbo",
        "anthropic/claude-3-haiku",
        "google/gemini-flash-1.5",
        "mistralai/mistral-7b-instruct",
    ],
    "performance": [
        "openai/gpt-4o",
        "anthropic/claude-3.5-sonnet", 
        "google/gemini-2.5-pro",
        "qwen/qwen3-30b-a3b-instruct-2507",
    ],
    "premium": [
        "openai/gpt-5",
        "anthropic/claude-3-opus",
        "openai/o1-preview", 
        "x-ai/grok-beta",
    ]
}

def get_config_value(key: str, default=None):
    """Çevre değişkeninden config değeri al"""
    return os.getenv(key, default)

def get_model_name() -> str:
    """Kullanılacak model adını döndür"""
    return get_config_value("MODEL_NAME", DEFAULT_MODEL)

def get_rate_limits() -> tuple[int, int]:
    """Rate limit ayarlarını döndür"""
    max_requests = int(get_config_value("MAX_REQUESTS_PER_WINDOW", DEFAULT_MAX_REQUESTS))
    window_seconds = int(get_config_value("RATE_LIMIT_WINDOW_SECONDS", DEFAULT_RATE_WINDOW))
    return max_requests, window_seconds

def get_timeout_settings() -> dict:
    """Timeout ayarlarını döndür"""
    return {
        "api_timeout": int(get_config_value("DEFAULT_TIMEOUT_SECONDS", DEFAULT_API_TIMEOUT)),
        "connect_timeout": int(get_config_value("CONNECT_TIMEOUT_SECONDS", DEFAULT_CONNECT_TIMEOUT)),
        "total_timeout": int(get_config_value("TOTAL_TIMEOUT_SECONDS", DEFAULT_TOTAL_TIMEOUT)),
    }

def get_output_dir() -> Path:
    """Output dizinini döndür"""
    output_path = get_config_value("OUTPUT_DIR", str(DEFAULT_OUTPUT_DIR))
    return Path(output_path)