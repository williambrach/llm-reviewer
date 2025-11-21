## Usage local

### Create .env file
```bash
API_KEY=
API_BASE=
```

### Create env

(install uv if not installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
uv venv --python 3.13
```

### Acitvate env

```bash
source .venv/bin/activate
```

### Install requirements

```bash
uv sync --all-extras
```

### Run

```bash
gradio app.py
```

