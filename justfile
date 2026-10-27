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

# World-model pipeline. On a GPU box, notebooks/kaggle.ipynb runs the same steps.

# Generate random-push episodes
gen data="data/random-4k" *args="":
    uv run jat gen {{data}} {{args}}

# Encode every pose with V-JEPA 2
encode data="data/random-4k" *args="":
    uv run jat encode {{data}} {{args}}

# Train the translator, then the predictor
train data="data/random-4k" run="runs/wm":
    uv run jat train translator {{data}} --out {{run}}
    uv run jat train predictor {{data}} --out {{run}}

# Imagined vs true pose after 0..T pushes
wm-eval data="data/random-4k" run="runs/wm":
    uv run jat wm-eval {{data}} --wm {{run}}
