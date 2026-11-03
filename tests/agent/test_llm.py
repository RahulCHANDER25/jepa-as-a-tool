from jat.agent.llm import chat_model


def test_a_local_server_is_reached_like_the_meta_api_with_more_patience(monkeypatch):
    monkeypatch.setenv("LOCAL_API_KEY", "none")
    monkeypatch.setenv("LOCAL_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LOCAL_MODEL", "glimmer")
    model = chat_model("local", temperature=0.7)
    assert model.model_name == "glimmer" and model.request_timeout == 1200
    assert str(model.openai_api_base) == "http://localhost:8000/v1"
