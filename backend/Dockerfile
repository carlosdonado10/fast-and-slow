# The runs API: FastAPI + pymongo only. The study's heavy dependencies (torch, SB3) are not installed: the API imports
# nothing from them, so the image is about 300 MB instead of several GB.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /bin/uv
ENV UV_SYSTEM_PYTHON=1 UV_NO_CACHE=1 PYTHONUNBUFFERED=1
WORKDIR /app

# dependencies first (pinned by uv.lock), so code changes don't reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv export --frozen --only-group api --no-hashes --no-emit-project -o /tmp/api.txt \
    && uv pip install -r /tmp/api.txt

COPY src ./src
RUN uv pip install --no-deps .

USER nobody
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=5 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status == 200 else 1)"
CMD ["uvicorn", "fast_and_slow.api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
