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

# OpenPI expects the step directory (e.g. .../10000), not .../10000/params.
if [[ -f "${CHECKPOINT_DIR}/_METADATA" && -d "${CHECKPOINT_DIR}/../assets" ]]; then
    echo "[openpi] CHECKPOINT_DIR points at params/; using parent step directory instead" >&2
    CHECKPOINT_DIR="$(cd "${CHECKPOINT_DIR}/.." && pwd)"
fi

# Real-time chunking (RTC) is opt-in. Set RTC_ENABLE=1 to wrap the policy with server-side RTC.
RTC_ENABLE="${RTC_ENABLE:-0}"
RTC_EXECUTE_HORIZON="${RTC_EXECUTE_HORIZON:-25}"
RTC_INFERENCE_DELAY="${RTC_INFERENCE_DELAY:-1}"
RTC_METHOD="${RTC_METHOD:-auto}"
RTC_PREFIX_ATTENTION_SCHEDULE="${RTC_PREFIX_ATTENTION_SCHEDULE:-exp}"
RTC_MAX_GUIDANCE_WEIGHT="${RTC_MAX_GUIDANCE_WEIGHT:-5.0}"

echo "[openpi] root=${OPENPI_ROOT}"
echo "[openpi] checkpoint=${CHECKPOINT_DIR}"
echo "[openpi] config=${OPENPI_CONFIG}"
echo "[openpi] bind=ws://0.0.0.0:${OPENPI_PORT}"
echo "[openpi] rtc_enable=${RTC_ENABLE}"

# tyro parses the optional `realtime` config as a subcommand that must follow the `policy:checkpoint` subcommand.
RTC_ARGS=()
if [[ "${RTC_ENABLE}" != "0" ]]; then
    echo "[openpi] rtc execute_horizon=${RTC_EXECUTE_HORIZON} inference_delay=${RTC_INFERENCE_DELAY} method=${RTC_METHOD}"
    RTC_ARGS=(
        realtime:realtime
        --realtime.execute-horizon="${RTC_EXECUTE_HORIZON}"
        --realtime.inference-delay="${RTC_INFERENCE_DELAY}"
        --realtime.method="${RTC_METHOD}"
        --realtime.prefix-attention-schedule="${RTC_PREFIX_ATTENTION_SCHEDULE}"
        --realtime.max-guidance-weight="${RTC_MAX_GUIDANCE_WEIGHT}"
    )
fi

cd "${OPENPI_ROOT}"
exec uv run scripts/serve_policy.py \
    --port="${OPENPI_PORT}" \
    policy:checkpoint \
    --policy.config="${OPENPI_CONFIG}" \
    --policy.dir="${CHECKPOINT_DIR}" \
    "${RTC_ARGS[@]}"
