import pytest


@pytest.fixture(scope="session")
def can_render():
    """Skip a test when this machine has no OpenGL for MuJoCo (CI has no GPU and no display)."""
    from jat.worlds import get_world

    try:
        get_world("tabletop").render([256.0, 200.0, 0.0], size=32)
    except Exception as error:  # noqa: BLE001 (any GL failure means "cannot render here")
        pytest.skip(f"MuJoCo cannot render here: {type(error).__name__}")
