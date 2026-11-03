"""Trained runs on the Hugging Face Hub, so a fresh install can use them (MCP server, evals).

    jat publish --repo <owner>/jepa-as-a-tool tabletop=runs/tabletop billiards=runs/billiards
    jat mcp --world billiards --wm hf:<owner>/jepa-as-a-tool/billiards

A run on the Hub is a folder of the model repo: translator.pt, predictor.pt and their json files.
"""

from pathlib import Path

PREFIX = "hf:"
_FILES = ["translator.pt", "predictor.pt", "*.json"]


def resolve_run(value: str | Path) -> Path:
    """A run folder: a local path, or hf:<owner>/<repo>/<run> (downloaded once, then cached)."""
    text = str(value)
    if not text.startswith(PREFIX):
        return Path(text)
    owner, repo, *run = text[len(PREFIX) :].split("/")
    if not run:
        raise ValueError(f"expected hf:<owner>/<repo>/<run>, got {text!r}")
    from huggingface_hub import snapshot_download

    sub = "/".join(run)
    local = snapshot_download(f"{owner}/{repo}", allow_patterns=[f"{sub}/{f}" for f in _FILES])
    return Path(local) / sub


def publish(repo: str, runs: dict[str, Path], private: bool = False) -> str:
    """Uploads each run's model files to <repo>/<name>; returns the repo URL."""
    from huggingface_hub import HfApi

    api = HfApi()
    url = api.create_repo(repo, repo_type="model", private=private, exist_ok=True)
    for name, folder in runs.items():
        api.upload_folder(
            repo_id=repo, folder_path=folder, path_in_repo=name, allow_patterns=_FILES,
            commit_message=f"Upload the {name} world model",
        )  # fmt: skip
    return str(url)
