FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN pip install uv \
    && uv sync --frozen --no-dev

COPY app ./app
COPY configs ./configs
COPY artifacts ./artifacts

EXPOSE 8000

CMD [".venv/bin/uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
