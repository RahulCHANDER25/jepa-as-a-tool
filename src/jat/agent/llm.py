"""Chat models behind OpenAI-compatible endpoints. Keys and URLs come from the environment (.env).

The "dry" provider is no model at all: it imagines two plans, pushes once and stops. It runs
every tool path of an episode with no API and no cost, for dry runs of the whole pipeline.
"""

import os

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

# provider -> (api key, base url, model id) environment variables
PROVIDERS = {
    "baseten": ("BASETEN_API_KEY", "BASETEN_BASE_URL", "BASETEN_MODEL"),
    "meta": ("META_API_KEY", "META_API_BASE_URL", "META_MODEL"),
}

_DRY_PUSHES = [
    {"start": [60.0, 60.0], "end": [160.0, 160.0]},
    {"start": [452.0, 452.0], "end": [352.0, 352.0]},
]


class DryModel(BaseChatModel):
    model_name: str = "dry"

    @property
    def _llm_type(self) -> str:
        return "dry"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        turn = sum(isinstance(m, AIMessage) for m in messages)
        script = [
            ("simulate", {"from_frame": "R0", "plans": [[p] for p in _DRY_PUSHES]}),
            ("act", {"push": _DRY_PUSHES[0]}),
            ("done", {}),
        ]
        name, args = script[min(turn, len(script) - 1)]
        reply = AIMessage(
            content=f"Dry run, step {turn + 1}.",
            tool_calls=[{"name": name, "args": args, "id": f"dry{turn}"}],
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])


def chat_model(provider: str, temperature: float) -> BaseChatModel:
    if provider == "dry":
        return DryModel()
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
