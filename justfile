# Run `just` to list recipes. Pipeline stages get a recipe each as they land.

default:
    @just --list

# Install the Python env and dev tools
setup:
    uv sync

# Run the test suite
test *args:
    uv run pytest {{args}}

# Check lint and formatting
lint:
    uv run ruff check .
    uv run ruff format --check .

# Apply lint fixes and formatting
fmt:
    uv run ruff check --fix .
    uv run ruff format .
