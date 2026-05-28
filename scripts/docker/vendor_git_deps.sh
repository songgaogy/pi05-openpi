#!/usr/bin/env bash
# Vendor git dependencies for offline-friendly Docker builds.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEROBOT_REV=0cf864870cf29f4738d3ade893e6fd13fbd7cdb5
LEROBOT_DIR="${ROOT}/third_party/lerobot"

clone_repo() {
  local dest="$1"
  shift
  for url in "$@"; do
    echo "Trying: ${url}"
    if git clone --filter=blob:none "${url}" "${dest}"; then
      return 0
    fi
    rm -rf "${dest}"
  done
  return 1
}

mkdir -p "${ROOT}/third_party"
if [ ! -d "${LEROBOT_DIR}/.git" ]; then
  clone_repo "${LEROBOT_DIR}" \
    "https://github.com/huggingface/lerobot.git" \
    "https://ghfast.top/https://github.com/huggingface/lerobot.git" \
    "https://mirror.ghproxy.com/https://github.com/huggingface/lerobot.git"
fi

git -C "${LEROBOT_DIR}" fetch --depth 1 origin "${LEROBOT_REV}" 2>/dev/null || git -C "${LEROBOT_DIR}" fetch origin
git -C "${LEROBOT_DIR}" checkout "${LEROBOT_REV}"

echo "Vendored lerobot at ${LEROBOT_DIR} (${LEROBOT_REV})"
echo "Rebuild with: SERVER_ARGS=\"--env LIBERO\" docker compose -f examples/libero/compose.yml build openpi_server"
