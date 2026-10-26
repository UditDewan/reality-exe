# Pinned Ubuntu 24.04 (noble) manifest list: amd64 and arm64/Graviton. Python 3.12 from the distro.
FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 WARP_CACHE_PATH=/app/.warp-cache
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python3 -m venv /venv && /venv/bin/pip install --no-cache-dir -r requirements.txt
COPY *.py ./
COPY web web

# Compile Newton/Warp CPU kernels at build time so the first trial is not a cold start; then self-test.
RUN /venv/bin/python -c "import physics, scene; physics.newton_rollout(scene.Config(), [], [[300, 200, 30]], T=0.05)" \
    && /venv/bin/python test_core.py \
    && useradd --create-home app && mkdir -p traces && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD /venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["/venv/bin/uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
