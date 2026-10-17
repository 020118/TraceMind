FROM node:22-alpine AS frontend-builder

WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AGENT_CONFIG=/app/config.toml \
    AGENT_ENABLE_TELEGRAM=0 \
    AGENT_ENABLE_PROACTIVE=0 \
    AGENT_WORKSPACE=/home/agent/.akashic/workspace

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 --shell /usr/sbin/nologin agent

COPY --chown=agent:agent . .
COPY --from=frontend-builder --chown=agent:agent /frontend/dist /app/frontend/dist
RUN cp /app/example_config.toml /app/config.toml
RUN mkdir -p /home/agent/.akashic/workspace \
    && chown -R agent:agent /home/agent/.akashic

USER agent
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
