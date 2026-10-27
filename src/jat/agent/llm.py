"""Chat models behind OpenAI-compatible endpoints. Keys and URLs come from the environment (.env)."""

import os

from langchain_openai import ChatOpenAI

# provider -> (api key, base url, model id) environment variables
PROVIDERS = {
    "baseten": ("BASETEN_API_KEY", "BASETEN_BASE_URL", "BASETEN_MODEL"),
    "meta": ("META_API_KEY", "META_API_BASE_URL", "META_MODEL"),
}


def chat_model(provider: str, temperature: float) -> ChatOpenAI:
    key_var, url_var, model_var = PROVIDERS[provider]
    missing = [v for v in (key_var, url_var, model_var) if not os.environ.get(v)]
    if missing:
        raise RuntimeError(f"Set {', '.join(missing)} in .env (see .env.example).")
    return ChatOpenAI(
        model=os.environ[model_var],
        api_key=os.environ[key_var],
        base_url=os.environ[url_var],
        temperature=temperature,
        max_retries=4,
        timeout=180,
    )
