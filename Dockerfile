FROM python:3.13-slim

# Copy uv binary from official image
COPY --from=ghcr.io/astral-sh/uv:0.5.14 /uv /uvx /bin/

# Environment variables for uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

WORKDIR /workspace

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock /workspace/

# Install dependencies (cached unless deps change)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

# Copy application code
COPY app.py logger.py .env /workspace/

EXPOSE 7799

CMD ["/workspace/.venv/bin/python", "/workspace/app.py"]
