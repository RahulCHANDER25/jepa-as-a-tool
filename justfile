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

# World-model pipeline. On a GPU machine, notebooks/train_and_evaluate.ipynb runs them for every world.

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

# What the worlds look like: looks/<world>.png (what V-JEPA 2 sees) and .mp4 (random pushes)
look world="":
    uv run jat show {{ if world == "" { "" } else { "--world " + world } }}

# A 3D world in MuJoCo's interactive viewer (tabletop, tabletop-new-look, billiards)
look-3d world="billiards":
    uv run jat show --world {{world}} --interactive

# Website. Copy eval runs and world-model tables into web/public/data, each as name=folder:
#   just export-web --run pusht=results/pusht --run billiards=results/billiards --wm main=runs/wm
export-web *args:
    uv run jat export-web {{args}}

# Serve the website with live reload
web:
    cd web && npm install && npm run dev

# Build the static website into web/dist
web-build:
    cd web && npm ci && npm run build
