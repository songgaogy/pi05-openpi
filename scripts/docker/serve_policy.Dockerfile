# Dockerfile for serving a PI policy.
# Based on UV's instructions: https://docs.astral.sh/uv/guides/integration/docker/#developing-in-a-container

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

# Needed because LeRobot uses git-lfs.
# Install Python via apt so uv does not download from GitHub during build.
RUN apt-get update && apt-get install -y \
    git git-lfs linux-headers-generic build-essential clang \
    python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=never

# Write the virtual environment outside of the project directory so it doesn't
# leak out of the container when we mount the application code.
ENV UV_PROJECT_ENVIRONMENT=/.venv

# GitHub/PyPI are often slow or blocked during image builds. Override at build time:
#   docker compose build --build-arg GIT_MIRROR= --build-arg UV_INDEX_URL=
# Or vendor lerobot first: scripts/docker/vendor_git_deps.sh
ARG GIT_MIRROR=https://ghfast.top/https://github.com
ARG UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ENV UV_HTTP_TIMEOUT=600 UV_INDEX_URL=${UV_INDEX_URL}

RUN if [ -n "$GIT_MIRROR" ]; then \
      git config --global url."${GIT_MIRROR%/}/".insteadOf "https://github.com/"; \
    fi && \
    git config --global http.lowSpeedLimit 1000 && \
    git config --global http.lowSpeedTime 600

# Install the project's dependencies using the lockfile and settings
RUN uv venv --python python3.11 $UV_PROJECT_ENVIRONMENT
# COPY (not bind-mount) so pyproject.toml can be patched for vendored lerobot.
# Run scripts/docker/vendor_git_deps.sh on the host before building.
COPY pyproject.toml uv.lock ./
COPY packages/openpi-client packages/openpi-client
COPY third_party/lerobot /app/third_party/lerobot
RUN --mount=type=cache,target=/root/.cache/uv \
    bash -euxo pipefail -c '\
      if [ -f third_party/lerobot/pyproject.toml ]; then \
        sed -i "s|lerobot = { git = \"https://github.com/huggingface/lerobot\", rev = \"[^\"]*\" }|lerobot = { path = \"third_party/lerobot\" }|" pyproject.toml; \
        GIT_LFS_SKIP_SMUDGE=1 uv lock --python python3.11; \
      fi; \
      GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev'

# Copy transformers_replace files while preserving directory structure
COPY src/openpi/models_pytorch/transformers_replace/ /tmp/transformers_replace/
RUN /.venv/bin/python -c "import transformers; print(transformers.__file__)" | xargs dirname | xargs -I{} cp -r /tmp/transformers_replace/* {} && rm -rf /tmp/transformers_replace

CMD /bin/bash -c "uv run scripts/serve_policy.py $SERVER_ARGS"
