#!/usr/bin/env bash
# Start OpenPI's official WebSocket policy server for an explicit checkpoint.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OPENPI_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OPENPI_ROOT="${OPENPI_ROOT:-${DEFAULT_OPENPI_ROOT}}"
OPENPI_CONFIG="${OPENPI_CONFIG:-pi05_arx_finetune_single_task}"
OPENPI_PORT="${OPENPI_PORT:-8000}"

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
    echo "CHECKPOINT_DIR is required" >&2
    echo "usage: CHECKPOINT_DIR=/absolute/path/to/checkpoint bash policy/pi05-openpi/pipeline/scripts/serve_openpi.sh" >&2
    exit 2
fi
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "CHECKPOINT_DIR does not exist or is not a directory: ${CHECKPOINT_DIR}" >&2
    exit 2
fi
if [[ ! -d "${OPENPI_ROOT}" ]]; then
    echo "OPENPI_ROOT does not exist or is not a directory: ${OPENPI_ROOT}" >&2
    exit 2
fi

CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}" && pwd)"

echo "[openpi] root=${OPENPI_ROOT}"
echo "[openpi] checkpoint=${CHECKPOINT_DIR}"
echo "[openpi] config=${OPENPI_CONFIG}"
echo "[openpi] bind=ws://0.0.0.0:${OPENPI_PORT}"

cd "${OPENPI_ROOT}"
exec uv run scripts/serve_policy.py \
    --port="${OPENPI_PORT}" \
    policy:checkpoint \
    --policy.config="${OPENPI_CONFIG}" \
    --policy.dir="${CHECKPOINT_DIR}"
