from livekit.plugins import openai

try:
    from livekit.plugins import google
except ImportError:
    google = None  # type: ignore

from ..config import KwamiVoiceConfig
from ..utils.provider import strip_model_prefix

# OpenAI models that only support temperature=1 (default); others support 0..2
_OPENAI_TEMPERATURE_FIXED_MODELS = ("gpt-5.1", "o1-", "o3-")


def _openai_temperature(config: KwamiVoiceConfig, model: str) -> float:
    """Use temperature=1 for models that only support the default (avoids API 400)."""
    if not model:
        return config.llm_temperature
    lower = model.lower()
    if any(prefix in lower for prefix in _OPENAI_TEMPERATURE_FIXED_MODELS):
        return 1.0
    return config.llm_temperature


def create_llm(config: KwamiVoiceConfig):
    """Create LLM instance based on configuration."""
    provider = config.llm_provider.lower()
    
    # Strip provider prefix if present (e.g. "openai/gpt-4.1-mini" -> "gpt-4.1-mini")
    model = strip_model_prefix(config.llm_model or "", provider)
    
    if provider == "openai":
        temp = _openai_temperature(config, model or "")
        try:
            return openai.LLM(
                model=model or "gpt-4o-mini",
                temperature=temp,
                max_tokens=config.llm_max_tokens,
            )
        except TypeError:
            # Older plugin versions may not expose max_tokens in constructor.
            return openai.LLM(
                model=model or "gpt-4o-mini",
                temperature=temp,
            )
    
    elif provider == "google" and google is not None:
        return google.LLM(
            model=model or "gemini-2.0-flash",
            temperature=config.llm_temperature,
        )
    
    elif provider == "anthropic":
        # Anthropic via OpenAI-compatible API
        return openai.LLM.with_anthropic(
            model=model or "claude-3-5-sonnet-latest",
            temperature=config.llm_temperature,
        )
    
    elif provider == "groq":
        # Groq via OpenAI-compatible API
        return openai.LLM.with_groq(
            model=model or "llama-3.1-70b-versatile",
            temperature=config.llm_temperature,
        )
    
    elif provider == "deepseek":
        # DeepSeek via OpenAI-compatible API
        return openai.LLM.with_deepseek(
            model=model or "deepseek-chat",
            temperature=config.llm_temperature,
        )
    
    elif provider == "mistral":
        # Mistral via OpenAI-compatible API  
        return openai.LLM.with_x_ai(
            model=model or "mistral-large-latest",
            temperature=config.llm_temperature,
            base_url="https://api.mistral.ai/v1",
        )
    
    elif provider == "cerebras":
        # Cerebras via OpenAI-compatible API
        return openai.LLM.with_cerebras(
            model=model or "llama3.1-70b",
            temperature=config.llm_temperature,
        )
    
    elif provider == "ollama":
        # Ollama local models
        return openai.LLM.with_ollama(
            model=model or "llama3.2",
            temperature=config.llm_temperature,
        )
    
    # Default to OpenAI
    return openai.LLM(
        model="gpt-4o-mini",
        temperature=0.7,
    )
